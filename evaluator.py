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
    compute_classify_matrics,
    compute_seg_metrics,
    send2api,
)


def _resolve_model_name(model_path: str, with_tag: bool) -> str:
    model_name = os.path.basename(model_path.rstrip("/\\"))
    if with_tag:
        model_name += "with-tag"
    return model_name


def _parse_pred_answer(pred_reasoning: str, result_path: str, model_type: str) -> int:
    text = (pred_reasoning or "").lower()

    if "nothinking" in result_path.lower():
        pred_yes_or_no = text.replace("addcriterion", "")
        pred_yes_or_no = "no" if "no" in pred_yes_or_no else "yes"
        return 1 if "yes" in pred_yes_or_no else 0

    if model_type == "glm" or "glm" in result_path.lower():
        try:
            pred_content_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, re.DOTALL)
            pred_yes_or_no = pred_content_match.group(1).strip()
        except Exception:
            pred_yes_or_no = send2api(pred_reasoning).strip().lower()
        return 1 if "yes" in pred_yes_or_no else 0

    try:
        pred_content_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        pred_yes_or_no = pred_content_match.group(1).strip()
    except Exception:
        pred_yes_or_no = send2api(pred_reasoning).strip().lower()
    return 1 if "yes" in pred_yes_or_no else 0


def _parse_gt_answer(sample_id: str, dataset: str) -> int:
    if dataset == "btad":
        return 0 if "ok" in sample_id else 1
    return 0 if "good" in sample_id else 1


def run_anomaly_metrics(args, result_json_path: str, result_dir: str, model_type: str):
    if not os.path.isfile(result_json_path):
        raise FileNotFoundError(f"Result json not found for anomaly metrics: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    anomaly_dct = {}
    for _, sample in results.items():
        category = sample.get("category", "")
        if category not in anomaly_dct:
            anomaly_dct[category] = {"pred": [], "true": []}

        sample_id = sample.get("id", "")
        pred_answer = _parse_pred_answer(sample.get("pred_reasoning", ""), result_json_path, model_type)
        gt_answer = _parse_gt_answer(sample_id, args.dataset)

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

    model_name = _resolve_model_name(args.model_path, args.with_tag)
    save_dir = os.path.join(args.generated_dir, model_name)
    out_path = os.path.join(save_dir, "output_attentions")
    out_img_path = os.path.join(save_dir, "images")
    out_model_dir = os.path.join(save_dir, "results")
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_model_dir, exist_ok=True)

    result_json_path = os.path.join(out_model_dir, "result.json")
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
        target_suffix = "_final_aggreated_image_fast.png" if args.return_aggreagate else "_final_valid_image_fast.png"
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

        try:
            pred_content_match = re.search(r"<answer>(.*?)</answer>", output_text.lower(), re.DOTALL)
            pred_yes_or_no = pred_content_match.group(1).strip()
        except Exception:
            pred_yes_or_no = send2api(output_text.strip())
            pred_yes_or_no = pred_yes_or_no.strip().lower()
        pred_yes_or_no = "yes" if "yes" in pred_yes_or_no else "no"
        pred_has_anomaly = pred_yes_or_no == "yes"

        pred_mask_median, sc, sample_outlier_tokens_num, sample_all_tokens_num = handler.evaluate_saved_attention(
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_text,
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=args.save_fig,
            with_tag=args.with_tag,
            return_aggreagate=args.return_aggreagate,
            patch_size=int(meta.get("patch_size", args.patch_size)),
            merge_size=int(meta.get("merge_size", args.merge_size)),
            layers_num=int(meta.get("layers_num", args.layers_num)),
            heads_num=int(meta.get("heads_num", args.heads_num)),
            vision_token_id=int(meta.get("vision_token_id", args.vision_token_id)),
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

    if args.return_aggreagate:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, "seg_score_aggreated_fast.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, "seg_score_aggreated_zero_fast.xlsx"), index=False, float_format="%.3f")
    else:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, "seg_score_median_new_fast.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, "seg_score_median_zero_new_fast.xlsx"), index=False, float_format="%.3f")

    print("Evaluation complete. Results saved to:", out_model_dir)
    rate = 0.0 if all_tokens_num == 0 else outlier_tokens_num / all_tokens_num
    print(f"outlier_tokens_num : {outlier_tokens_num}\n all_tokens_num : {all_tokens_num}\n rate : {rate}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    cli_args = p.parse_args()
    main(build_stage_namespace(cli_args.config, stage="evaluator", dataset=cli_args.dataset))

