#!/usr/bin/env python3
import os
import argparse
import json
import torch
from qwen_vl_utils import process_vision_info
from util import (
    toliststr,
    load_dataset,
    resize_image,
    get_resize_info,
    move_to_cpu,
    optimized_get_saved_per_layer_head_attention,
)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

question_with_tag = """Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. Before answering, perform a structured visual assessment in the following order:
Overview: Briefly describe the overall content, context, and general appearance of the image.
Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, explain why the image appears normal and consistent with expected standards.
Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.
Output your response strictly in this format—without any additional text or tags outside the specified structure:
<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer> """


def main(args):
    ret = {}
    q_with_tag = False
    merged_patch_size, max_size = get_resize_info(args.model_path)
    eval_dataset = load_dataset(args.dataset_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    if ("with-tag" in model_name) or args.with_tag:
        q_with_tag = True
        model_name += "with-tag"
    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "output_attentions")
    os.makedirs(out_path, exist_ok=True)

    for _, data in eval_dataset.iterrows():
        img_path = toliststr(data["image_path"])[0]
        save_name = img_path.replace(args.replace_path, "")
        sample_id = save_name
        file_id = save_name.replace("/", "__").replace("\\", "__")
        save_path = os.path.join(out_path, f"{file_id}.pt")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Skipping existing {save_path}")
            continue

        question = data["question"]
        prompt = question_with_tag if q_with_tag else question
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }]

        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        processed_image = resize_image(processed_image, max_size, merged_patch_size)
        inputs = processor(
            text=[processed_text],
            images=processed_image,
            return_tensors="pt",
        ).to(model.device)

        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_attentions=True,
        )

        sequences = generated["sequences"]
        input_token_len = len(inputs.input_ids[0])
        trimmed = sequences[0][input_token_len:]
        output_text = processor.tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        print(output_text)

        compressed_attn, attn_meta = optimized_get_saved_per_layer_head_attention(
            tokenizer=tokenizer,
            output_ids=generated,
            input_token_len=input_token_len,
            processed_image=processed_image,
            processed_prompt=processed_text,
            sequences=sequences,
            vision_token_id=args.vision_token_id,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
            outlier_ratio=args.outlier_ratio,
            dominance_ratio=args.dominance_ratio,
            outlier_share_thr=args.outlier_share_thr,
        )

        compressed_attn_to_save = move_to_cpu(compressed_attn)
        if isinstance(compressed_attn_to_save, dict):
            for k in ("vlm_attn", "prompt2text_attn", "filtered_vlm_attn", "filtered_prompt2text_attn"):
                v = compressed_attn_to_save.get(k, None)
                if torch.is_tensor(v):
                    compressed_attn_to_save[k] = v.to(torch.float16)

        meta = {
            "image_path": img_path,
            "question": question,
            "category": data.get("category", ""),
            "gt_image": data.get("gt_image", ""),
            "input_token_len": input_token_len,
            "output_text": output_text,
            **attn_meta,
        }

        torch.save(
            {
                "compressed_attn": compressed_attn_to_save,
                "sequence": move_to_cpu(sequences),
                "meta": move_to_cpu(meta),
            },
            save_path,
        )

        ret[sample_id] = {
            "id": sample_id,
            "category": data["category"],
            "pred_reasoning": output_text,
            "gt_reasoning": data["answer"],
        }
        print(f"Saved {save_path}")

    json_path = os.path.join(save_dir, "results")
    os.makedirs(json_path, exist_ok=True)
    json_path = os.path.join(json_path, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", "-d", default="/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv")
    p.add_argument("--dataset", "-dd", default="wfdd")
    p.add_argument("--model_path", "-m", default="/gpfsdata/home/yizhou/yizhou_lustre/modelscope/models/Qwen3/Qwen3-VL-4B-Thinking")
    p.add_argument("--save_dir", "-s", default="/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--with_tag", action="store_true")
    p.add_argument("--vision_token_id", type=int, default=151655)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--merge_size", type=int, default=2)
    p.add_argument("--outlier_ratio", type=float, default=50.0)
    p.add_argument("--dominance_ratio", type=float, default=5.0)
    p.add_argument("--outlier_share_thr", type=float, default=0.3)
    args = p.parse_args()

    if args.dataset == "mvtec":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/MVTecAD_seg_0shot.tsv"
        args.save_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot"
        args.replace_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/MVTec-AD/"
    elif args.dataset == "sdd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/SDD_seg_0shot.tsv"
        args.save_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/SDD_seg_0shot"
        args.replace_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/SDD/SDD/"
    elif args.dataset == "dtd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/DTD_seg_0shot.tsv"
        args.save_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/DTD_seg_0shot"
        args.replace_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/DTD/"
    elif args.dataset == "wfdd":
        args.dataset_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/eval_dataset/WFDD_seg_0shot.tsv"
        args.save_dir = "/gpfsdata/home/yizhou/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot"
        args.replace_path = "/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/TEST_DATASET/WFDD/"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    main(args)
