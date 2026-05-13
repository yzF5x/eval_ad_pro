#!/usr/bin/env python3
import argparse
import json
import os

import torch

from configs.config_loader import build_stage_namespace
from configs.dataset_config import DATASET_DEFAULTS, QUESTION_WITH_TAG
from models.factory import HandlerFactory
from utils import build_model_name, load_dataset, move_to_cpu, toliststr

import re

# 1. 将正则表达式预编译为全局常量（模块加载时只编译一次，极其高效）
# re.IGNORECASE 标志直接让匹配忽略大小写，省去了 path.lower() 的开销
_ABNORMAL_RE = re.compile(
    r'ungood|bad|defect|defective|anomaly|fault|error|broken|broken_large'
    r'|damaged|crack|scratch|scratch_large|abnormal'
    r'|bent|stain|rust|dent|chip|flip|contamination|misaligned|missing|blob'
    r'|color|glue|translucent|fog',
    re.IGNORECASE
)

_NORMAL_RE = re.compile(
    r'(?:^|[/\\])(normal|good|ok|healthy|perfect|undamaged)',
    re.IGNORECASE
)

def _normalize_tsv_answer(answer) -> str:
    answer_text = str(answer).strip().lower()
    if re.search(r"\byes\b", answer_text):
        return "yes"
    if re.search(r"\bno\b", answer_text):
        return "no"
    raise ValueError(f"Cannot parse GT answer from TSV answer field: {answer}")


def parse_gt_answer(path: str, data=None, dataset: str = "") -> str:
    if str(dataset).strip().lower() == "mcbt":
        if data is None or "answer" not in data:
            raise ValueError("MCBT requires reading GT from the TSV 'answer' field.")
        return _normalize_tsv_answer(data["answer"])

    # 2. 直接在原始字符串上进行 C 层级的单次扫描匹配     
    if _NORMAL_RE.search(path):
        return "no"
    return "yes"
    # raise ValueError(f"无法识别路径中的 GT 标签: '{path}'")

def _get_input_token_len(inputs) -> int:
    if hasattr(inputs, "input_ids"):
        return len(inputs.input_ids[0])
    if isinstance(inputs, dict) and "input_ids" in inputs:
        return len(inputs["input_ids"][0])
    raise ValueError("Cannot infer input token length from model inputs.")


def main(args):
    eval_dataset = load_dataset(args.dataset_path)

    model_type = HandlerFactory.infer_model_type(args.model_path, args.model_type)
    handler = HandlerFactory.create(
        model_type=model_type,
        model_path=args.model_path,
        use_monkey_patch=(not args.disable_monkey_patch) if model_type == "qwen" else False,
        device="auto",
        torch_dtype="bfloat16",
        attn_implementation="eager",
    )

    model_name = build_model_name(args.model_path, args.with_tag)

    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "output_attentions")
    os.makedirs(out_path, exist_ok=True)

    ret = {}
    for i, data in eval_dataset.iterrows():
        img_path = toliststr(data["image_path"])[0]
        save_name = img_path.replace(args.replace_path, "")
        sample_id = save_name
        file_id = save_name.replace("/", "__").replace("\\", "__")
        save_path = os.path.join(out_path, f"{file_id}.pt")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Skipping existing {save_path}")
            continue

        question = str(data.get("question", ""))
        text_prompt = QUESTION_WITH_TAG if args.with_tag else question

        processed = handler.preprocess(
            img_path=img_path,
            question=question,
            use_structured_prompt=args.with_tag,
        )
        inputs = processed["inputs"]
        processed_image = processed["processed_image"]

        generated = handler.generate(
            inputs=inputs,
            max_new_tokens=args.max_new_tokens,
            return_attentions=True,
        )

        sequences = generated["sequences"]
        input_token_len = _get_input_token_len(inputs)
        output_text = handler.decode_output(sequences=sequences, input_len=input_token_len)
        print(output_text)

        compressed_attn, attn_meta = handler.extract_attention(
            generated=generated,
            input_len=input_token_len,
            processed_image=processed_image,
            model_type=model_type,
            vision_token_id=args.vision_token_id,
            patch_size=args.patch_size,
            merge_size=args.merge_size,
        )

        compressed_attn_to_save = move_to_cpu(compressed_attn)
        if isinstance(compressed_attn_to_save, dict):
            for k in (
                "flatten_text2vision_attn",
                "flatten_text2text_attn",
                "vlm_attn",
                "prompt2text_attn",
                "filtered_vlm_attn",
                "filtered_prompt2text_attn",
            ):
                v = compressed_attn_to_save.get(k, None)
                if torch.is_tensor(v):
                    compressed_attn_to_save[k] = v.to(torch.float16)

        meta = {
            "image_path": img_path,
            "question": text_prompt,
            "category": data.get("category", ""),
            "gt_image": data.get("gt_image", ""),
            "input_token_len": input_token_len,
            "output_text": output_text,
            "model_type": model_type,
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
        gt_answer = parse_gt_answer(img_path, data=data, dataset=args.dataset)
        ret[sample_id] = {
            "id": sample_id,
            "category": data.get("category", ""),
            "pred_reasoning": output_text,
            "answer": gt_answer,
            "gt_reasoning": data.get("answer", ""),
        }
        print(f"Saved {save_path}")

    json_path = os.path.join(save_dir, "results")
    os.makedirs(json_path, exist_ok=True)
    json_path = os.path.join(json_path, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS.keys()))
    cli_args = p.parse_args()
    main(build_stage_namespace(cli_args.config, stage="generator", dataset=cli_args.dataset))
