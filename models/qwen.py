from qwen_vl_utils import process_vision_info

from configs.dataset_config import QUESTION_WITH_TAG
from utils import (
    get_resize_info,
    resize_image,
    use_monkey_patch_qwen2_5vl_qkvfp32_eager_encoderselfattn,
    use_monkey_patch_qwen2_5vl_qkvfp32_eager_visionattn,
)

from .base import BaseModelHandler


class QwenHandler(BaseModelHandler):
    def __init__(self, model_path: str, use_monkey_patch: bool = True, **kwargs):
        if use_monkey_patch:
            use_monkey_patch_qwen2_5vl_qkvfp32_eager_visionattn()
            use_monkey_patch_qwen2_5vl_qkvfp32_eager_encoderselfattn()
        super().__init__(model_path, **kwargs)

    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> dict:
        text = QUESTION_WITH_TAG if use_structured_prompt else question
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        merged_patch_size, max_size = get_resize_info(self.model_path)
        processed_image = resize_image(processed_image, max_size, merged_patch_size)
        inputs = self.processor(text=[prompt], images=processed_image, return_tensors="pt")
        return {"inputs": inputs, "prompt_text": prompt, "processed_image": processed_image}

