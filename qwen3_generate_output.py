#!/usr/bin/env python3
"""
generate_outputs.py
第一轮遍历：对每张图片 run model.generate(...)，把生成结果（sequences）和必要 metadata 保存到 disk。
可选保存 attentions（--save_attentions），注意文件会非常大。
输出目录结构:
  <save_dir>/
    <model_name>/
      <id_0001>.npz   # numpy .npz 包含 sequences (ints), input_token_len (int), output_text (str), image_path (str), question (str)
"""

import os
import argparse
import json
import numpy as np
from qwen_vl_utils import process_vision_info
from util import (toliststr, load_dataset, send2api,
                  load_model,get_saved_attention,resize_image,get_saved_per_layer_head_attention,
                  get_resize_info)
from PIL import Image
import re
import torch
import pickle
from transformers import Qwen3VLForConditionalGeneration,AutoProcessor,AutoTokenizer

question_with_tag = """Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. Before answering, perform a structured visual assessment in the following order:
Overview: Briefly describe the overall content, context, and general appearance of the image.
Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, explain why the image appears normal and consistent with expected standards.
Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.
Output your response strictly in this format—without any additional text or tags outside the specified structure:
<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer> """ 

# question_with_tag = """Inspect the provided image and respond strictly in the following exact format—no additional text, explanations, whitespace, or characters before, between, or after the tags:

# <overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer>

# - In <overview>, briefly describe the overall scene, including the main subject(s), context, lighting, and general visual appearance.
# - In <analyze>, carefully examine the image for any visible defects, anomalies, or irregularities (such as cracks, stains, discolorations, distortions, foreign objects, or structural inconsistencies). If no clear defects are found, explain why the image appears normal and consistent with expected standards.
# - In <conclusion>, provide a single, clear sentence stating whether any defects or anomalies are present.
# - In <answer>, output only "Yes" or "No"—nothing else.

# Your entire response must begin with "<overview>" and end with "</answer>", contain exactly these four tags in this order, and include no markdown, line breaks outside tag content, or extra punctuation. Do not add any commentary, disclaimers, or reasoning beyond what is required within each tag."""

# output_path = "../results/1013-median/MVTecAD_seg_0shot/"
def move_to_cpu(obj):
    """Recursively move all torch.Tensors in obj to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(v) for v in obj)
    else:
        return obj
def main(args):
    ret = {}
    q_with_tag = False
    merged_patch_size , max_size = get_resize_info(args.model_path)
    eval_dataset = load_dataset(args.dataset_path)
    # model, processor, tokenizer = load_model(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if ('with-tag' in model_name) or args.with_tag:
        q_with_tag = True
        print('q with tag!!!!!!!!!!!!!!!!!!!!!!!')
        model_name += 'with-tag'
    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir,'output_attentions')
    os.makedirs(out_path, exist_ok=True)
    for i, data in eval_dataset.iterrows():

        img_path = toliststr(data["image_path"])[0]
        save_name = img_path.replace(args.replace_path, "")
        # sanitize file id (no slashes)
        id = save_name
        file_id = save_name.replace("/", "__")
        
        save_path = os.path.join(out_path, f"{file_id}.pt")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Skipping existing {save_path}")
            continue
        prompt = question_with_tag if q_with_tag else data["question"]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text":prompt}
            ],
        }]
        
        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        processed_image = resize_image(processed_image, max_size , merged_patch_size)
        inputs = processor(
            text=[processed_text],
            images=processed_image,
            return_tensors="pt",
        ).to(model.device)

        # generate
        generated = model.generate(
            **inputs,  
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=args.save_attentions,   # if we want attentions, set True
            output_attentions=args.save_attentions
        )

        sequences = generated["sequences"]
   
        # trimmed ids (what user used previously) can be reconstructed in evaluation if needed.
        input_token_len = len(inputs.input_ids[0])
    
        trimmed = sequences[0][input_token_len:]
        output_text = processor.tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(output_text)
        ret[id] = {
                "id": id,
                "category":data['category'], 
                "pred_reasoning":output_text ,
                "gt_reasoning":data['answer']
            }

        attn_compressed, meta = optimized_get_saved_per_layer_head_attention(
            tokenizer, sequences, input_token_len, processed_image, processed_text,
            sequences=sequences,           # ← 新增
            vision_token_id=151655,        # ← 新增
            save_path="attn_compressed.pt" # ← 直接保存
        )
        torch.save({
            "filtered_attn": compressed_attn.cpu(),  # 可进一步用 .half() 压缩
            "meta": meta
        }, save_path)
        # llm_attn_matrix,output_token_len = get_saved_per_layer_head_attention(tokenizer = tokenizer,
        #                                                 output_ids = generated,
        #                                                 input_token_len = input_token_len,
        #                                                 processed_image = processed_image,
        #                                                 processed_prompt = processed_text,
        #                                                 save_name = save_name,
        #                                                 patch_size = 16
        #                                                 )
        # # metadata
        # meta = {
        #     "image_path": img_path,
        #     "question": data["question"],
        #     "category": data.get("category", ""),
        #     "gt_image": data.get("gt_image", ""),
        #     "input_token_len": input_token_len,
        #     "output_text": output_text,
        #     'output_token_len':output_token_len
        # }

        # save_dict = {
        #      "llm_attn_matrix": llm_attn_matrix,
        #      "sequence": sequences,
        #     "meta": meta
        # }
        # save_dict_cpu = move_to_cpu(save_dict)
        # # 安全保存
        # with open(save_path, "wb") as f:
        #     pickle.dump(save_dict_cpu, f)
        # torch.save(save_dict, out_path)

        print(f"Saved {save_path}")
    json_path = os.path.join(save_dir, 'results')
    os.makedirs(json_path, exist_ok=True)
    json_path = os.path.join(json_path, 'result.json')
    with open(json_path , 'w') as f:
            json.dump(ret, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', '-d', default='/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv',help='')
    p.add_argument('--dataset', '-dd', default='wfdd')
    p.add_argument('--model_path', '-m', default='/gpfsdata/home/yizhou/yizhou_lustre/modelscope/models/Qwen3/Qwen3-VL-4B-Thinking',help='')
    p.add_argument('--save_dir', '-s', default='/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot', help='where to save .npz files')
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--save_attentions', action='store_false', help='also save attentions from generate (very large)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--with_tag', action='store_true')
    args = p.parse_args()
    if args.dataset == 'mvtec':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MVTecAD_seg_0shot.tsv'
        args.save_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot'
        args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/MVTec-AD/'
    elif args.dataset == 'sdd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/SDD_seg_0shot.tsv'
        args.save_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/SDD_seg_0shot'
        args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/SDD/SDD/'
    # elif args.dataset =='btad':
    #     args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/BTAD_seg_0shot.tsv'
    #     args.save_dir = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/results/BTAD_seg_0shot'
    #     args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/btad/'
    elif args.dataset =='dtd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DTD_seg_0shot.tsv'
        args.save_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/DTD_seg_0shot'
        args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/DTD/'
    elif args.dataset == 'wfdd':
        args.dataset_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv'
        args.save_dir = '/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot'
        args.replace_path = '/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/WFDD/'
    else:
        exit(0)
    main(args)
