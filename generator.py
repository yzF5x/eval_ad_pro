#!/usr/bin/env python3
import argparse
import json
import os

import torch

from configs.config_loader import build_stage_namespace
from configs.dataset_config import DATASET_DEFAULTS, QUESTION_WITH_TAG
from models.factory import HandlerFactory
from utils import build_model_name, load_dataset, move_to_cpu, toliststr


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
    for _, data in eval_dataset.iterrows():
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
        processed_prompt = processed["prompt_text"]
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
            prompt=processed_prompt,
            model_type=model_type,
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

        ret[sample_id] = {
            "id": sample_id,
            "category": data.get("category", ""),
            "pred_reasoning": output_text,
            "answer": data.get("answer", ""),
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
