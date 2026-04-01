#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from qwen_vl_utils import process_vision_info
from util import (
    toliststr,
    load_dataset,
    compute_seg_metrics,
    get_resize_info,
    resize_image,
    send2api,
    optimized_get_attention_from_saved_per_layer_head_fast,
)
from PIL import Image
import re
import torch
import numpy as np
from modelscope import AutoProcessor, AutoTokenizer

question_with_tag = """Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. Before answering, perform a structured visual assessment in the following order:
Overview: Briefly describe the overall content, context, and general appearance of the image.
Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, explain why the image appears normal and consistent with expected standards.
Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.
Output your response strictly in this format—without any additional text or tags outside the specified structure:
<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer> """


def main(args):
    outlier_tokens_num, all_tokens_num = 0, 0
    merged_patch_size, max_size = get_resize_info(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    if args.with_tag:
        model_name += "with-tag"
    save_dir = os.path.join(args.generated_dir, model_name)
    out_path = os.path.join(save_dir, "output_attentions")
    out_img_path = os.path.join(save_dir, "images")
    os.makedirs(out_img_path, exist_ok=True)
    if not os.path.isdir(out_path):
        raise FileNotFoundError(f"Generated dir not found: {out_path}")

    pixel_dct_median = {}
    pixel_dct_median_zero = {}

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
            for k in ("vlm_attn", "prompt2text_attn", "filtered_vlm_attn", "filtered_prompt2text_attn"):
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
        output_token_len = int(meta.get("output_token_len", 0))
        output_text = meta.get("output_text", "")
        print(output_text)
        print(f"Evaluating {fpath} | image: {img_path}")

        with_tag = False
        if "with-tag" in model_name:
            with_tag = True
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question_with_tag},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }]

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
        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        processed_image = resize_image(processed_image, max_size, merged_patch_size)
        width, height = processed_image[0].size

        try:
            pred_content_match = re.search(r"<answer>(.*?)</answer>", output_text.lower(), re.DOTALL)
            pred_yes_or_no = pred_content_match.group(1).strip()
        except Exception:
            pred_yes_or_no = send2api(output_text.strip())
            pred_yes_or_no = pred_yes_or_no.strip().lower()
        pred_yes_or_no = "yes" if "yes" in pred_yes_or_no else "no"
        pred_has_anomaly = pred_yes_or_no == "yes"

        pred_mask_median, sc, sample_outlier_tokens_num, sample_all_tokens_num = optimized_get_attention_from_saved_per_layer_head_fast(
            tokenizer=tokenizer,
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_text,
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=args.save_fig,
            with_tag=with_tag,
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

    out_model_dir = os.path.join(save_dir, "results")
    os.makedirs(out_model_dir, exist_ok=True)
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
    p.add_argument("--generated_dir", default="/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot")
    p.add_argument("--dataset", "-dd", default="mvtec")
    p.add_argument("--model_path", "-m", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--global_save_fig", action="store_true")
    p.add_argument("--normal_set_zero", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--return_aggreagate", action="store_true")
    p.add_argument("--with_tag", action="store_true")
    p.add_argument("--vision_token_id", type=int, default=151655)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--merge_size", type=int, default=2)
    p.add_argument("--layers_num", type=int, default=36)
    p.add_argument("--heads_num", type=int, default=32)
    args = p.parse_args()
    args.global_save_fig = False
    args.return_aggreagate = True

    if args.dataset == "mvtec":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MVTecAD_seg_0shot.tsv"
        args.generated_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot"
        args.replace_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/dataset/anomaly/MVTec-AD/"
    elif args.dataset == "sdd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/SDD_seg_0shot.tsv"
        args.generated_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/SDD_seg_0shot"
        args.replace_path = "/home/yizhou/LVLM/dataset/TEST_DATASET/SDD/SDD/"
    elif args.dataset == "dtd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DTD_seg_0shot.tsv"
        args.generated_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/DTD_seg_0shot"
        args.replace_path = "/home/yizhou/LVLM/dataset/TEST_DATASET/dtd/"
    elif args.dataset == "wfdd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv"
        args.generated_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot"
        args.replace_path = "/home/yizhou/LVLM/dataset/TEST_DATASET/wfdd/"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    main(args)
