from PIL import Image

from configs.dataset_config import QUESTION_WITH_TAG
from utils import get_resize_info, resize_image

from .base import BaseModelHandler


class LlavaNextHandler(BaseModelHandler):
    @staticmethod
    def _to_int(value, attr_name: str) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError(f"processor.{attr_name} is empty.")
            return LlavaNextHandler._to_int(value[0], attr_name)
        return int(value)

    def _load_components(self):
        super()._load_components()
        _, self.max_size = get_resize_info(self.model_path)
        if self.max_size <= 0:
            raise ValueError(f"Invalid max_size from get_resize_info for model path: {self.model_path}")
        self.patch_size = self._to_int(getattr(self.processor, "patch_size"), "patch_size")
        if self.patch_size <= 0:
            raise ValueError(f"Invalid processor.patch_size: {self.patch_size}")

    def _get_aggregation_grid(self):
        grid_height = self._to_int(
            getattr(self.processor, "patches_height_for_aggregation"),
            "patches_height_for_aggregation",
        )
        grid_width = self._to_int(
            getattr(self.processor, "patches_width_for_aggregation"),
            "patches_width_for_aggregation",
        )
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError(f"Invalid LLaVA-Next aggregation grid: {grid_height}x{grid_width}")
        return grid_height, grid_width

    def _attention_runtime_defaults(self):
        grid_height, grid_width = self._get_aggregation_grid()
        return {
            "patch_size": self.patch_size,
            "merge_size": 1,
            "layers_num": 32,
            "heads_num": 32,
            "vision_token_id": 128256,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "outlier_ratio": 50.0,
            "dominance_ratio": 5.0,
            "outlier_share_thr": 0.3,
        }

    def _resolve_attention_params(self, **kwargs):
        cfg = super()._resolve_attention_params(**kwargs)
        grid_height, grid_width = self._get_aggregation_grid()
        cfg["patch_size"] = self.patch_size
        cfg["merge_size"] = 1
        cfg["layers_num"] = 32
        cfg["heads_num"] = 32
        cfg["vision_token_id"] = 128256
        cfg["grid_height"] = grid_height
        cfg["grid_width"] = grid_width
        return cfg

    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> dict:
        text = QUESTION_WITH_TAG if use_structured_prompt else question
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        img = Image.open(img_path).convert("RGB")
        processed_image = resize_image([img], max_size=self.max_size, merged_patch_size=self.patch_size)

        inputs = self.processor(
            text=prompt,
            images=processed_image,
            return_tensors="pt",
        )
        if self.model is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        return {"inputs": inputs, "prompt_text": prompt, "processed_image": processed_image}
