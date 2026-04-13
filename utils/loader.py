import pandas as pd
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import torch


def load_dataset(dataset_path):
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".xlsx"):
        dataset = pd.read_excel(dataset_path)
    elif dataset_path.endswith(".tsv"):
        dataset = pd.read_csv(dataset_path, sep="\t")
    return dataset


def load_model(
    model_path: str,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
    use_fast: bool = False,
    load_model_weights: bool = True,
    **kwargs,
):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

    if attn_implementation != "eager":
        attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=use_fast
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=use_fast
    )
    model = None
    if load_model_weights:
        lower_path = model_path.lower()
        model_cls = AutoModel
        if any(k in lower_path for k in ("qwen3", "qwen-3")):
            from transformers import Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        elif any(k in lower_path for k in ("qwen2.5-vl", "qwen2_5", "vision-r1")):
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
        elif any(k in lower_path for k in ("internvl3", "internvl3_5", "internvl3.5")):
            from transformers import InternVLForConditionalGeneration

            model_cls = InternVLForConditionalGeneration
        elif "glm" in lower_path:
            from transformers import Glm4vForConditionalGeneration

            model_cls = Glm4vForConditionalGeneration

        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        model.eval()
    return model, processor, tokenizer
