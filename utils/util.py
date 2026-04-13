import ast
import os

from PIL import Image
import requests
import torch


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


def get_resize_info(model_path):
    if 'qwen3' in model_path.lower():
        merged_patch_size = 32
        max_size = 416
    elif 'qwen2' in model_path.lower():
        merged_patch_size = 28
        max_size = 420
    elif 'glm' in model_path.lower():
        merged_patch_size = 28
        max_size = 420
    else:
        return -1 , -1
    return merged_patch_size , max_size

def resize_image(images , max_size , merged_patch_size):
    if max_size == -1:
        return images
    processed_images = []
    for image in images:
        width, height = image.size
        long_side = max(width, height)
        scale = max_size / long_side
        short_side = min(width, height)
        new_short = short_side * scale
        new_short = (int(new_short) // merged_patch_size) * merged_patch_size
        new_short = max(new_short, merged_patch_size)  # 防止短边为 0
        new_size = (max_size , new_short) if  width >= height else (new_short , max_size)
        processed_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
        processed_images.append(processed_image)
    return processed_images

def send2api(prediction , prompt = "" , model = "google/gemini-flash-1.5"):
    """
        无法直接提取答案时 调api 让api模型根据测试模型的文本输出得到答案
        prediction是测试模型的文本输出
        prompt是给API model的问题
        发送给API的内容是 prompt + prediction
        可以参考的prompt:
        prompt = "Determine whether there is an anomaly or defect by the semantics of the following paragraph.  If yes, answer \"yes\", otherwise answer \"no\".  No other words are allowed.  No punctuation is allowed. The paragraph is : "
    """
    prompt = "Determine whether there is an anomaly or defect by the semantics of the following paragraph.  If there is, answer \"yes\", otherwise answer \"no\".  No other words are allowed except the number.  No punctuation is allowed. The paragraph is : "
    url = "https://openrouter.ai/api/v1/chat/completions"
    ssh_key = os.getenv("OPENROUTER_API_KEY", "")
    if not ssh_key:
        return "Something Wrong with API"
    headers= {
            "Authorization": f"Bearer " + ssh_key
    }
    text = {
        "type": "text",
        "text": prompt + f" '{prediction}' "
    }
    content = [text]
    payload = {
        "model": model, 
        "messages": [
          {
            "role": "user",
            "content": content
          }
        ]
    }
    response = requests.post(url = url, headers=headers, json=payload)
    if response.status_code == 200:
        json_llm_answer = response.json()
        choices = json_llm_answer.get('choices',[])
        d = choices[0]
        messages = d.get('message',{})
        content = messages.get('content','')
        print(prediction,"\n" ,content,"\n")
        return content
    else:
        return "Something Wrong with API"
    
def toliststr(s):
    """
        字符串形式的列表 转换为 真实列表
    """
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in ast.literal_eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

