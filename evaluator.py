#!/usr/bin/env python3
import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image

from configs.config_loader import build_stage_namespace
from configs.dataset_config import DATASET_DEFAULTS
from models.factory import HandlerFactory
from utils import (
    build_model_name,
    compute_classify_matrics,
    compute_seg_metrics,
    evaluate_saved_attention_fast,
    evaluate_saved_attention_sink_first,
    send2api,
)


def _extract_tag_content(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip().lower()


def _fallback_yes_no(text: str) -> str:
    lowered = (text or "").lower()
    yes_match = re.search(r"\byes\b", lowered)
    no_match = re.search(r"\bno\b", lowered)
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    if yes_match:
        return "yes"
    if no_match:
        return "no"
    return "no"


def _parse_pred_answer(pred_reasoning: str, result_path: str, model_type: str, openrouter_api_key: str = "") -> int:
    text = (pred_reasoning or "").lower()

    # if "nothinking" in result_path.lower():
    #     pred_yes_or_no = text.replace("addcriterion", "")
    #     pred_yes_or_no = "no" if "no" in pred_yes_or_no else "yes"
    #     return 1 if "yes" in pred_yes_or_no else 0

    if model_type == "glm" or "glm" in result_path.lower():
        pred_yes_or_no = _extract_tag_content(text, r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>")
    else:
        pred_yes_or_no = _extract_tag_content(text, r"<answer>(.*?)</answer>")
    if not pred_yes_or_no:
        try:
            pred_yes_or_no = send2api(pred_reasoning, openrouter_api_key=openrouter_api_key).strip().lower()
        except RuntimeError as exc:
            print(f"Warning: API fallback failed, using local parsing result. {exc}")
            pred_yes_or_no = _fallback_yes_no(text)
    return 1 if "yes" in pred_yes_or_no else 0


def _parse_gt_answer(sample: dict) -> int:
    raw_answer = str(sample.get("answer", sample.get("gt_reasoning", ""))).strip()
    if not raw_answer:
        raise ValueError(f"Missing 'answer' field in sample: {sample.get('id', '')}")

    answer_text = _extract_tag_content(raw_answer, r"<answer>\s*(.*?)\s*</answer>")
    if not answer_text:
        answer_text = raw_answer.lower()

    if re.search(r"\byes\b", answer_text):
        return 1
    if re.search(r"\bno\b", answer_text):
        return 0
    raise ValueError(f"Cannot parse image-level label from answer: {raw_answer}")


def _normalize_attention_eval_mode(mode: str) -> str:
    aliases = {
        "fast": "fast",
        "sink_first": "sink_first",
        "sink-first": "sink_first",
        "sinkfirst": "sink_first",
    }
    normalized = aliases.get(str(mode).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported attention_eval_mode: {mode}. Use 'fast' or 'sink_first'.")
    return normalized


def _build_eval_variant_tag(attention_eval_mode: str, topk_spike_patches: int) -> str:
    return f"{attention_eval_mode}_topk_spike_patches_{topk_spike_patches}"


def run_anomaly_metrics(args, result_json_path: str, result_dir: str, model_type: str):
    if not os.path.isfile(result_json_path):
        raise FileNotFoundError(f"Result json not found for anomaly metrics: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    anomaly_dct = {}
    openrouter_api_key = getattr(args, "OPENROUTER_API_KEY", "")
    for _, sample in results.items():
        category = sample.get("category", "")
        if category not in anomaly_dct:
            anomaly_dct[category] = {"pred": [], "true": []}

        pred_answer = _parse_pred_answer(
            sample.get("pred_reasoning", ""),
            result_json_path,
            model_type,
            openrouter_api_key=openrouter_api_key,
        )
        gt_answer = _parse_gt_answer(sample)

        anomaly_dct[category]["pred"].append(pred_answer)
        anomaly_dct[category]["true"].append(gt_answer)

    anomaly_scores = compute_classify_matrics(anomaly_dct)
    anomaly_score_path = os.path.join(result_dir, "anomaly_score.xlsx")
    anomaly_scores.to_excel(anomaly_score_path, index=False, float_format="%.3f")
    print(f"Saved anomaly metrics to {anomaly_score_path}")


def main(args):
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
    attention_eval_mode = _normalize_attention_eval_mode(getattr(args, "attention_eval_mode", "fast"))
    topk_spike_patches = int(getattr(args, "topk_spike_patches", 3))
    eval_variant_tag = _build_eval_variant_tag(attention_eval_mode, topk_spike_patches)
    evaluate_attention_fn = (
        evaluate_saved_attention_sink_first
        if attention_eval_mode == "sink_first"
        else evaluate_saved_attention_fast
    )
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

    pixel_dct_median = {}
    pixel_dct_median_zero = {}
    outlier_tokens_num, all_tokens_num = 0, 0

    files = [
        os.path.join(out_path, f)
        for f in os.listdir(out_path)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    files.sort()

    for fpath in files:
        data = torch.load(fpath, map_location="cpu")
        compressed_attn = data.get("compressed_attn", data.get("filtered_attn"))
        if isinstance(compressed_attn, dict):
            for k in ("vlm_attn", "prompt2text_attn"):
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
        if attention_eval_mode == "sink_first":
            target_suffix = (
                "_final_aggreated_image_fast_sink_first.png"
                if args.return_aggregate
                else "_final_valid_image_fast_sink_first.png"
            )
        else:
            target_suffix = (
                "_final_aggreated_image_fast.png"
                if args.return_aggregate
                else "_final_valid_image_fast.png"
            )
        target_path = save_name.replace(".png", target_suffix)
        if args.global_save_fig and os.path.exists(target_path) and not args.overwrite:
            print(f"Skipping existing {target_path}")
            args.save_fig = False
        else:
            args.save_fig = bool(args.global_save_fig)

        directory = os.path.dirname(save_name)
        os.makedirs(directory, exist_ok=True)

        processed = handler.preprocess(
            img_path=img_path,
            question=question,
            use_structured_prompt=args.with_tag,
        )
        processed_text = processed["prompt_text"]
        processed_image = processed["processed_image"]
        width, height = processed_image[0].size

        pred_has_anomaly = bool(_parse_pred_answer(output_text, result_json_path, model_type))

        pred_mask_median, sc, sample_outlier_tokens_num, sample_all_tokens_num = evaluate_attention_fn(
            tokenizer=handler.tokenizer,
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_text,
            model_type=meta.get("model_type", model_type),
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=args.save_fig,
            with_tag=args.with_tag,
            return_aggregate=args.return_aggregate,
            patch_size=int(meta.get("patch_size", args.patch_size)),
            merge_size=int(meta.get("merge_size", args.merge_size)),
            layers_num=int(meta.get("layers_num", args.layers_num)),
            heads_num=int(meta.get("heads_num", args.heads_num)),
            vision_token_id=int(meta.get("vision_token_id", args.vision_token_id)),
            topk_spike_patches=topk_spike_patches,
        )

        outlier_tokens_num += sample_outlier_tokens_num
        all_tokens_num += sample_all_tokens_num

        if pred_has_anomaly or not args.normal_set_zero:
            pred_mask_median_zero = pred_mask_median
        else:
            pred_mask_median_zero = np.zeros((height, width), dtype=int)

        try:
            gt_img = Image.open(gt_image)
            gt_img = gt_img.resize((width, height))
            gt_img = gt_img.convert("L")
            gt_mask = (np.array(gt_img) > 128).astype(int)
        except Exception:
            gt_mask = np.zeros((height, width), dtype=int)

        gt_flat = gt_mask.flatten().tolist()
        pred_flat_median = pred_mask_median.flatten().tolist()
        pred_flat_median_zero = pred_mask_median_zero.flatten().tolist()

        if category not in pixel_dct_median:
            pixel_dct_median[category] = {"pred": [], "true": []}
        if category not in pixel_dct_median_zero:
            pixel_dct_median_zero[category] = {"pred": [], "true": []}
        pixel_dct_median[category]["pred"].append(pred_flat_median)
        pixel_dct_median[category]["true"].append(gt_flat)
        pixel_dct_median_zero[category]["pred"].append(pred_flat_median_zero)
        pixel_dct_median_zero[category]["true"].append(gt_flat)

    seg_metrics_median = compute_seg_metrics(pixel_dct_median)
    seg_metrics_median_zero = compute_seg_metrics(pixel_dct_median_zero)

    score_tag = "sink_first" if attention_eval_mode == "sink_first" else "fast"
    if args.return_aggregate:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, f"seg_score_aggreated_{score_tag}.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, f"seg_score_aggreated_zero_{score_tag}.xlsx"), index=False, float_format="%.3f")
    else:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, f"seg_score_median_new_{score_tag}.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, f"seg_score_median_zero_new_{score_tag}.xlsx"), index=False, float_format="%.3f")

    print("Evaluation complete. Results saved to:", out_model_dir)
    rate = 0.0 if all_tokens_num == 0 else outlier_tokens_num / all_tokens_num
    print(f"outlier_tokens_num : {outlier_tokens_num}\n all_tokens_num : {all_tokens_num}\n rate : {rate}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    p.add_argument("--topk_spike_patches", type=int, default=None)
    cli_args = p.parse_args()
    stage_args = build_stage_namespace(cli_args.config, stage="evaluator", dataset=cli_args.dataset)
    if cli_args.topk_spike_patches is not None:
        if cli_args.topk_spike_patches <= 0:
            raise ValueError(f"--topk_spike_patches must be > 0, got: {cli_args.topk_spike_patches}")
        stage_args.topk_spike_patches = cli_args.topk_spike_patches
    main(stage_args)
