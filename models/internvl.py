from PIL import Image

from configs.dataset_config import QUESTION_WITH_TAG
from utils import dynamic_preprocess

from .base import BaseModelHandler


class InternVLHandler(BaseModelHandler):
    def _attention_runtime_defaults(self):
        lower = self.model_path.lower()
        if "internvl3_5" in lower or "internvl3.5" in lower:
            return {
                "patch_size": 14,
                "merge_size": 2,
                "layers_num": 36,
                "heads_num": 32,
                "vision_token_id": 151671,
                "outlier_ratio": 50.0,
                "dominance_ratio": 5.0,
                "outlier_share_thr": 0.3,
            }
        return {
            "patch_size": 14,
            "merge_size": 2,
            "layers_num": 28,
            "heads_num": 28,
            "vision_token_id": 151667,
            "outlier_ratio": 50.0,
            "dominance_ratio": 5.0,
            "outlier_share_thr": 0.3,
        }

    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> dict:
        text = QUESTION_WITH_TAG if use_structured_prompt else question
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(img_path).convert("RGB")
        processed_image = dynamic_preprocess(image, max_num=1)
        inputs = self.processor(text=[prompt], images=processed_image, return_tensors="pt")
        return {"inputs": inputs, "prompt_text": prompt, "processed_image": processed_image}

    def _grid_shape_for_image(self, width, height, patch_size, merge_size, num_patches=None):
        return 16, 16

    def _vision_token_span(self, flat_ids, vision_token_id, fallback_num_patches):
        vision_positions = (flat_ids == vision_token_id).nonzero(as_tuple=True)[0]
        if vision_positions.numel() == 0:
            raise ValueError(f"Vision token id {vision_token_id} not found in prompt sequence.")
        start = int(vision_positions[0].item())
        end = start + 256
        if end > flat_ids.numel():
            raise ValueError("InternVL fixed vision span exceeds prompt token length.")
        return start, end

