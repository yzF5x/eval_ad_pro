from PIL import Image

from configs.dataset_config import QUESTION_WITH_TAG
from utils import get_resize_info, resize_image

from .base import BaseModelHandler


class GlmHandler(BaseModelHandler):
    def _attention_runtime_defaults(self):
        return {
            "patch_size": 14,
            "merge_size": 2,
            "layers_num": 40,
            "heads_num": 32,
            "vision_token_id": 151343,
            "outlier_ratio": 50.0,
            "dominance_ratio": 5.0,
            "outlier_share_thr": 0.3,
        }

    def _load_components(self):
        super()._load_components()
        self.merged_patch_size, self.max_size = get_resize_info(self.model_path)

    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> dict:
        text = QUESTION_WITH_TAG if use_structured_prompt else question
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image = Image.open(img_path).convert("RGB")
        processed_image = resize_image([image], self.max_size, self.merged_patch_size)
        inputs = self.processor(text=prompt, images=processed_image, return_tensors="pt")
        return {"inputs": inputs, "prompt_text": prompt, "processed_image": processed_image}

