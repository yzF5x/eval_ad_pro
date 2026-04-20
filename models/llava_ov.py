from PIL import Image

from configs.dataset_config import QUESTION_WITH_TAG
from utils import get_resize_info, resize_image

from .base import BaseModelHandler


class LlavaOvHandler(BaseModelHandler):
    def _load_components(self):
        super()._load_components()
        self.merged_patch_size, self.max_size = get_resize_info(self.model_path)
        if self.max_size <= 0 or self.merged_patch_size <= 0:
            raise ValueError(f"Invalid max_size from get_resize_info for model path: {self.model_path}")
        self.patch_size = 14

    def _attention_runtime_defaults(self):
        return {
            "patch_size": 14,
            "merge_size": 2,
            "layers_num": 36,
            "heads_num": 32,
            "vision_token_id": 151655,
            "outlier_ratio": 50.0,
            "dominance_ratio": 5.0,
            "outlier_share_thr": 0.3,
        }

    def _resolve_attention_params(self, **kwargs):
        cfg = super()._resolve_attention_params(**kwargs)
        cfg["patch_size"] = 14
        cfg["merge_size"] = 2
        cfg["layers_num"] = 36
        cfg["heads_num"] = 32
        cfg["vision_token_id"] = 151655
        return cfg

    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> dict:
        text = QUESTION_WITH_TAG if use_structured_prompt else question
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        img = Image.open(img_path).convert("RGB")
        processed_image = resize_image([img], max_size=self.max_size, merged_patch_size=self.merged_patch_size)

        inputs = self.processor(
            text=prompt,
            images=processed_image,
            return_tensors="pt",
        )
        if self.model is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        return {"inputs": inputs, "prompt_text": prompt, "processed_image": processed_image}
