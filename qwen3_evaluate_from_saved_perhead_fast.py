#!/usr/bin/env python3
"""
evaluate_from_saved.py
第二轮读取第一轮保存的 .npz 文件，重建 processed_image/processed_prompt，调用 get_median_outputtoken_vision_attn
并统计 pixel-level dict，最后 compute_seg_metrics 并导出 Excel / JSON。
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from qwen_vl_utils import process_vision_info
from util import (get_attention_from_saved_per_layer_head_fast,get_attention_from_saved_new,get_attention_from_saved_tag,toliststr, load_dataset, compute_seg_metrics,get_resize_info,resize_image,get_attention_from_saved_per_layer_head,send2api,
                   load_model)
from PIL import Image
import re
import torch
import time
import pickle
from modelscope import AutoProcessor,AutoTokenizer
question_with_tag = """Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. Before answering, perform a structured visual assessment in the following order:
Overview: Briefly describe the overall content, context, and general appearance of the image.
Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, explain why the image appears normal and consistent with expected standards.
Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.
Output your response strictly in this format—without any additional text or tags outside the specified structure:
<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer> """ 

def main(args):
    outlier_tokens_num , all_tokens_num = 0 , 0
    merged_patch_size , max_size = get_resize_info(args.model_path)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if args.with_tag:
        model_name += 'with-tag'
    save_dir = os.path.join(args.generated_dir, model_name)
    out_path = os.path.join(save_dir,'output_attentions')
    out_img_path = os.path.join(save_dir, 'images')
    os.makedirs(out_img_path, exist_ok=True)
    if not os.path.isdir(out_path):
        raise FileNotFoundError(f"Generated dir not found: {out_path}")

    pixel_dct_orig = {}
    pixel_dct_filtered = {}
    pixel_dct_median = {}
    pixel_dct_median_zero = {}

    # iterate all .npz files
    files = [os.path.join(out_path, f) for f in os.listdir(out_path) if f.endswith('.pth')]
    files.sort()

    for i,fpath in enumerate(files):
        with open(fpath, "rb") as f:
            data = torch.load(f)
        llm_attn_matrix = data['filtered_attn']
        sequences = data['sequence']
        meta = data["meta"]
        img_path = meta['image_path']
        question = meta['question']
        category = meta.get('category', '')
        # if category != 'leather':
        #     continue
        gt_image = meta.get('gt_image', '')
        input_token_len = int(meta.get('input_token_len', 0))
        output_token_len = int(meta.get('output_token_len',0))
        output_text = meta.get('output_text', '')
        print(output_text)
        with_tag = False
        print(f"Evaluating {fpath} | image: {img_path}")
        if 'with-tag' in model_name:
            with_tag = True
            print('q with tag!!!!!!!!!!!!!!!!!!!!!!!')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question_with_tag}
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question}
                ],
            }]
        save_name = img_path.replace(args.replace_path, "")
        save_name = os.path.join(out_img_path, save_name)
        if args.return_aggreagate:
            if args.global_save_fig:
                if os.path.exists(save_name.replace('.png','_final_aggreated_image.png')) and not args.overwrite:
                    print(f"Skipping existing {save_name.replace('.png','_final_aggreated_image.png')}")
                    args.save_fig=False
                else:
                    args.save_fig=True
            else:
                args.save_fig=False
        else:
            if args.global_save_fig:
                if os.path.exists(save_name.replace('.png','_final_valid_image.png')) and not args.overwrite:
                    print(f"Skipping existing {save_name.replace('.png','_final_valid_image.png')}")
                    args.save_fig=False
                else:
                    args.save_fig=True
            else:
                args.save_fig=False
        directory = os.path.dirname(save_name)
        os.makedirs(directory, exist_ok=True)
        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        processed_image = resize_image(processed_image, max_size , merged_patch_size)
        width, height = processed_image[0].size

        # Determine pred_yes_or_no same as before
        try:
            pred_content_match = re.search(r'<answer>(.*?)</answer>', output_text.lower(), re.DOTALL)
            pred_yes_or_no = pred_content_match.group(1).strip()
        except:
            pred_yes_or_no = send2api(output_text.strip())
            pred_yes_or_no = pred_yes_or_no.strip().lower()
        pred_yes_or_no = "yes" if "yes" in pred_yes_or_no else "no"
        pred_has_anomaly = True if pred_yes_or_no == "yes" else False

        # 2. 加载 + 直接渲染（无过滤计算）
        result = get_attention_from_saved_per_layer_head_fast(
            
            sequences, tokenizer, img, ...
        )  # 内部不再执行 outlier 检测，速度提升 2-3 倍
        
        
    #     pred_mask_median, sc ,sample_outlier_tokens_num ,sample_all_tokens_num= get_attention_from_saved_per_layer_head_fast(
    #     tokenizer=tokenizer,
    #     llm_attn_matrix=llm_attn_matrix,
    #     sequences=sequences,
    #     input_token_len=input_token_len,
    #     output_token_len=output_token_len,
    #     processed_image=processed_image,
    #     processed_prompt=processed_text,
    #     save_name=save_name,    # we can skip saving heatmap here or configure a folder
    #     pred_has_anomaly=pred_has_anomaly,
    #     save_fig=args.save_fig,
    #     with_tag=with_tag,
    #     return_aggreagate=args.return_aggreagate,
    #     patch_size = 16,
    #     layers_num=36,
    #     heads_num=32,
    #     vision_token_id = 151655
    # )
        outlier_tokens_num += sample_outlier_tokens_num 
        all_tokens_num += sample_all_tokens_num
        if pred_has_anomaly:
            pred_mask_median_zero = pred_mask_median
        else:
            pred_mask_median_zero = np.zeros((height, width), dtype=int)

        # Load GT mask
        try:
            gt_img = Image.open(gt_image)
            gt_img = gt_img.resize((width, height))
            gt_img = gt_img.convert('L')
            gt_mask = (np.array(gt_img) > 128).astype(int)
        except Exception:
            gt_mask = np.zeros((height, width)).astype(int)

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

        # exit(0)
        # create id from filename


    # After all files processed, compute metrics and save
    # os.makedirs(args.output_path, exist_ok=True)
    seg_metrics_median = compute_seg_metrics(pixel_dct_median)
    seg_metrics_median_zero = compute_seg_metrics(pixel_dct_median_zero)
    

    # save Excel
    out_model_dir = os.path.join(save_dir, 'results')
    os.makedirs(out_model_dir, exist_ok=True)
    if args.return_aggreagate:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, "seg_score_aggreated_fast.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, "seg_score_aggreated_zero_fast.xlsx"), index=False, float_format="%.3f")
    else:
        seg_metrics_median.to_excel(os.path.join(out_model_dir, "seg_score_median_new_fast.xlsx"), index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(os.path.join(out_model_dir, "seg_score_median_zero_new_fast.xlsx"), index=False, float_format="%.3f")
    # # save jsons
    # with open(os.path.join(out_model_dir, "seg_metrics_median.json"), 'w') as f:
    #     json.dump(pixel_dct_median, f, indent=4, ensure_ascii=False)
    # with open(os.path.join(out_model_dir, "seg_metrics_orig.json"), 'w') as f:
    #     json.dump(pixel_dct_orig, f, indent=4, ensure_ascii=False)


    print("Evaluation complete. Results saved to:", out_model_dir)
    rate = outlier_tokens_num / all_tokens_num
    print(f"outlier_tokens_num : {outlier_tokens_num}\n all_tokens_num : {all_tokens_num}\n rate : {rate}")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--generated_dir', default='/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot', help='where to save .npz files')
    p.add_argument('--dataset', '-dd', default='mvtec')
    p.add_argument('--model_path', '-m', default='Qwen/Qwen3-VL-8B-Thinking',help='')
    # p.add_argument('--output_path', '-o', default='/share/home/yizhou_lustre/LVLM-results/results/MVTecAD_seg_0shot')
    p.add_argument('--global_save_fig', action='store_true', help='if get_median_outputtoken_vision_attn should save visualization')
    p.add_argument('--normal_set_zero', action='store_true', help='set anomaly maps to zero for predicted normal images')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--return_aggreagate', action='store_true')
    p.add_argument('--with_tag', action='store_true')
    args = p.parse_args()
    args.global_save_fig = False
    args.return_aggreagate = True
    if args.dataset == 'mvtec':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MVTecAD_seg_0shot.tsv'
        args.generated_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot'
        args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/dataset/anomaly/MVTec-AD/'
    # elif args.dataset == 'mpdd':
    #     args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MPDD_seg_0shot.tsv'
    #     args.generated_dir = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/results/MPDD_seg_0shot'
    #     args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/mpdd/'
    # elif args.dataset == 'dagm':
    #     args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DAGM_seg_0shot.tsv'
    #     args.generated_dir = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/results/DAGM_seg_0shot'
    #     args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/dagm-new/'
    elif args.dataset == 'sdd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/SDD_seg_0shot.tsv'
        args.generated_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/SDD_seg_0shot'
        args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/SDD/SDD/'
    # elif args.dataset =='btad':
    #     args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/BTAD_seg_0shot.tsv'
    #     args.generated_dir = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/results/BTAD_seg_0shot'
    #     args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/btad/'
    elif args.dataset =='dtd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DTD_seg_0shot.tsv'
        args.generated_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/DTD_seg_0shot'
        args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/dtd/'
    elif args.dataset == 'wfdd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv'
        args.generated_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot'
        args.replace_path = '/home/yizhou/LVLM/dataset/TEST_DATASET/wfdd/'
    main(args)
