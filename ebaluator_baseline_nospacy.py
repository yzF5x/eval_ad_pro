#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from configs.config_loader import build_stage_namespace
from configs.dataset_config import DATASET_DEFAULTS
from evaluator import _parse_pred_answer, run_anomaly_metrics
from models.factory import HandlerFactory
from utils import build_model_name, compute_seg_metrics
from utils.visual_tools import get_attention_from_saved_ablation_nospacy


def _normalize_threshold_index(value, name: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer, got: {value}") from exc
    if index < 0:
        raise ValueError(f"{name} must be >= 0, got: {index}")
    return index


def main(args):
    threshold_start = _normalize_threshold_index(getattr(args, "threshold_start", 0), "threshold_start")
    threshold_end = _normalize_threshold_index(getattr(args, "threshold_end", 1), "threshold_end")
    if threshold_end < threshold_start:
        raise ValueError(
            f"threshold_end must be >= threshold_start, got {threshold_start}, {threshold_end}"
        )

    model_type = HandlerFactory.infer_model_type(args.model_path, args.model_type)
    handler = HandlerFactory.create(
        model_type=model_type,
        model_path=args.model_path,
        use_monkey_patch=False,
        device="auto",
        torch_dtype="bfloat16",
        attn_implementation="eager",
        load_model_weights=False,
    )

    model_name = build_model_name(args.model_path, args.with_tag)
    eval_variant_tag = "baseline_ablation_nospacy"
    save_dir = os.path.join(args.generated_dir, model_name)
    out_path = os.path.join(save_dir, "output_attentions")
    base_result_dir = os.path.join(save_dir, "results")
    out_img_path = os.path.join(save_dir, "images", eval_variant_tag)
    out_model_dir = os.path.join(base_result_dir, eval_variant_tag)
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_model_dir, exist_ok=True)

    result_json_path = os.path.join(base_result_dir, "result.json")
    run_anomaly_metrics(args, result_json_path=result_json_path, result_dir=out_model_dir, model_type=model_type)

    if not os.path.isdir(out_path):
        raise FileNotFoundError(f"Generated dir not found: {out_path}")

    pixel_dct = {}
    pixel_dct_zero = {}
    outlier_tokens_num, all_tokens_num = 0, 0

    files = [
        os.path.join(out_path, f)
        for f in os.listdir(out_path)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    files.sort()

    for i, fpath in enumerate(files, start=1):
        if args.debug_mode and i > 10:
            break

        data = torch.load(fpath, map_location="cpu")
        compressed_attn = data.get("compressed_attn", data.get("filtered_attn"))
        if isinstance(compressed_attn, dict):
            for k in ("flatten_text2vision_attn", "flatten_text2text_attn", "vlm_attn", "prompt2text_attn"):
                v = compressed_attn.get(k, None)
                if torch.is_tensor(v) and v.dtype in (torch.float16, torch.bfloat16):
                    compressed_attn[k] = v.float()

        sequences = data["sequence"]
        meta = data["meta"]
        img_path = meta["image_path"]
        question = meta["question"]
        category = meta.get("category", "")
        gt_image = meta.get("gt_image", "")
        input_token_len = int(meta.get("input_token_len", 0))
        output_token_len = int(meta.get("output_token_len", max(int(sequences.shape[-1] - input_token_len), 0)))
        output_text = meta.get("output_text", "")

        print(output_text)
        print(f"Evaluating {fpath} | image: {img_path}")

        save_name = img_path.replace(args.replace_path, "")
        save_name = os.path.join(out_img_path, save_name)
        target_suffix = "_baseline_nospacy_aggregated.png" if args.return_aggregate else "_baseline_nospacy_final.png"
        target_path = os.path.splitext(save_name)[0] + target_suffix
        if args.global_save_fig and os.path.exists(target_path) and not args.overwrite:
            print(f"Skipping existing {target_path}")
            args.save_fig = False
        else:
            args.save_fig = bool(args.global_save_fig)

        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        processed = handler.preprocess(
            img_path=img_path,
            question=question,
            use_structured_prompt=args.with_tag,
        )
        processed_image = processed["processed_image"]
        width, height = processed_image[0].size

        pred_has_anomaly = bool(_parse_pred_answer(output_text, result_json_path, model_type))

        pred_mask, sc, sample_outlier_tokens_num, sample_all_tokens_num = get_attention_from_saved_ablation_nospacy(
            tokenizer=handler.tokenizer,
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            model_type=meta.get("model_type", model_type),
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=args.save_fig,
            with_tag=args.with_tag,
            return_aggregate=args.return_aggregate,
            patch_size=int(meta.get("patch_size", args.patch_size)),
            merge_size=int(meta.get("merge_size", args.merge_size)),
            vision_token_id=int(meta.get("vision_token_id", args.vision_token_id)),
            grid_height=meta.get("grid_height"),
            grid_width=meta.get("grid_width"),
            threshold_start=threshold_start,
            threshold_end=threshold_end,
        )

        if np.isnan(pred_mask).any():
            print(f"\n[ERROR] Found NaN in prediction mask!")
            print(f"[ERROR] Problematic file path: {fpath}")
            print(f"[ERROR] Image path: {img_path}")
            raise ValueError(f"NaN detected in mask for {fpath}")

        outlier_tokens_num += int(sample_outlier_tokens_num)
        all_tokens_num += int(sample_all_tokens_num)

        pred_mask_zero = pred_mask if pred_has_anomaly else np.zeros((height, width), dtype=int)

        try:
            gt_img = Image.open(gt_image)
            gt_img = gt_img.resize((width, height))
            gt_img = gt_img.convert("L")
            gt_mask = (np.array(gt_img) > 128).astype(int)
        except Exception:
            gt_mask = np.zeros((height, width), dtype=int)

        gt_flat = gt_mask.flatten().tolist()
        pred_flat = pred_mask.flatten().tolist()
        pred_flat_zero = pred_mask_zero.flatten().tolist()

        if category not in pixel_dct:
            pixel_dct[category] = {"pred": [], "true": []}
        if category not in pixel_dct_zero:
            pixel_dct_zero[category] = {"pred": [], "true": []}
        pixel_dct[category]["pred"].append(pred_flat)
        pixel_dct[category]["true"].append(gt_flat)
        pixel_dct_zero[category]["pred"].append(pred_flat_zero)
        pixel_dct_zero[category]["true"].append(gt_flat)

    seg_metrics = compute_seg_metrics(pixel_dct)
    seg_metrics_zero = compute_seg_metrics(pixel_dct_zero)

    score_tag = "baseline_ablation_nospacy"
    if args.return_aggregate:
        seg_metrics.to_excel(
            os.path.join(out_model_dir, f"seg_score_aggreated_{score_tag}.xlsx"),
            index=False,
            float_format="%.3f",
        )
        seg_metrics_zero.to_excel(
            os.path.join(out_model_dir, f"seg_score_aggreated_zero_{score_tag}.xlsx"),
            index=False,
            float_format="%.3f",
        )
    else:
        seg_metrics.to_excel(
            os.path.join(out_model_dir, f"seg_score_median_new_{score_tag}.xlsx"),
            index=False,
            float_format="%.3f",
        )
        seg_metrics_zero.to_excel(
            os.path.join(out_model_dir, f"seg_score_median_zero_new_{score_tag}.xlsx"),
            index=False,
            float_format="%.3f",
        )

    print("No-spacy baseline evaluation complete. Results saved to:", out_model_dir)
    rate = 0.0 if all_tokens_num == 0 else outlier_tokens_num / all_tokens_num
    print(f"outlier_tokens_num : {outlier_tokens_num}\n all_tokens_num : {all_tokens_num}\n rate : {rate}")
    token_stats = {
        "outlier_tokens_num": int(outlier_tokens_num),
        "all_tokens_num": int(all_tokens_num),
        "outlier_ratio": float(rate),
    }
    token_stats_path = os.path.join(out_model_dir, f"outlier_token_stats_{score_tag}.json")
    with open(token_stats_path, "w", encoding="utf-8") as fp:
        json.dump(token_stats, fp, ensure_ascii=False, indent=2)
    print(f"Outlier token stats saved to: {token_stats_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    p.add_argument("--threshold_start", type=int, default=0)
    p.add_argument("--threshold_end", type=int, default=1)
    p.add_argument("--debug_mode", action="store_true", default=False)
    cli_args = p.parse_args()
    stage_args = build_stage_namespace(cli_args.config, stage="evaluator", dataset=cli_args.dataset)
    stage_args.threshold_start = _normalize_threshold_index(cli_args.threshold_start, "threshold_start")
    stage_args.threshold_end = _normalize_threshold_index(cli_args.threshold_end, "threshold_end")
    stage_args.debug_mode = cli_args.debug_mode
    main(stage_args)
