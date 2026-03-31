import os
import pandas as pd
import json
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,Qwen3VLForConditionalGeneration,AutoTokenizer,AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer,AutoModel

import torch

def load_dataset(dataset_path):
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".xlsx"):
        dataset = pd.read_excel(dataset_path)
    elif dataset_path.endswith(".tsv"):
        dataset = pd.read_csv(dataset_path, sep="\t")
 
    return dataset


def load_model(model_path):
    
    if "qwen2.5-vl" or "vision-r1" or "qwen2_5" in model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    # elif "intern" in model_path.lower():
    #     model = AutoModel.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #         use_flash_attn=True,
    #         trust_remote_code=True).eval().cuda()
    # elif "glm" in model_path.lower():
    #     from transformers import Glm4vForConditionalGeneration
    #     model = Glm4vForConditionalGeneration.from_pretrained(
    #         model_path,
    #         torch_dtype = torch.bfloat16,
    #         device_map="auto"
    #     )
    #     processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    return model , processor , tokenizer