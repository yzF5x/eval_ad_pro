from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Tuple

import torch

from utils.loader import load_model
from utils.visual_tools import (
    aggregate_cross_attentions,
    compute_spatial_consistency_fast,
    custom_weighted_sum,
    detect_attn_spike_by_share,
    detect_single_extreme_values_in_vlm_attn,
    elbow_chord,
    get_par_from_attention_fast,
    get_spatial_entropy_from_attention_fast,
    get_threshold_and_weight_from_sum,
    get_token_indices_by_pos_and_words,
    get_weight_with_indices,
    heatmap_visual,
    normalize_heatmap,
    optimized_save_per_layer_head_attention,
    visual_attn_token2image,
)


class BaseModelHandler(ABC):
    def __init__(self, model_path: str, device: str = "auto", **loader_kwargs):
        self.model_path = model_path
        self.device = device
        self.loader_kwargs = loader_kwargs
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_components()

    def _load_components(self):
        self.model, self.processor, self.tokenizer = load_model(
            self.model_path, device=self.device, **self.loader_kwargs
        )

    def _move_inputs_to_model(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = v.to(model_device)
            else:
                moved[k] = v
        return moved

    @abstractmethod
    def preprocess(self, img_path: str, question: str, use_structured_prompt: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def _attention_runtime_defaults(self) -> Dict[str, Any]:
        return {
            "patch_size": 14,
            "merge_size": 2,
            "layers_num": 28,
            "heads_num": 28,
            "vision_token_id": 151655,
            "outlier_ratio": 50.0,
            "dominance_ratio": 5.0,
            "outlier_share_thr": 0.3,
        }

    def _resolve_attention_params(self, **kwargs) -> Dict[str, Any]:
        cfg = dict(self._attention_runtime_defaults())
        cfg.update({k: v for k, v in kwargs.items() if v is not None})
        return cfg

    def _grid_shape_for_image(
        self,
        width: int,
        height: int,
        patch_size: int,
        merge_size: int,
        num_patches: int = None,
    ) -> Tuple[int, int]:
        if num_patches is not None:
            return self._infer_grid_from_num_patches(num_patches, width, height)
        return int(width / (patch_size * merge_size)), int(height / (patch_size * merge_size))

    def _vision_token_span(
        self,
        flat_ids: torch.Tensor,
        vision_token_id: int,
        fallback_num_patches: int,
    ) -> Tuple[int, int]:
        return self._infer_vision_span(flat_ids, vision_token_id, fallback_num_patches)

    @staticmethod
    def _infer_grid_from_num_patches(num_patches: int, width: int, height: int) -> Tuple[int, int]:
        if num_patches <= 0:
            raise ValueError("num_patches must be positive.")
        aspect = (width / max(height, 1)) if height > 0 else 1.0
        best_w, best_h = num_patches, 1
        best_score = float("inf")
        for h in range(1, int(num_patches**0.5) + 1):
            if num_patches % h != 0:
                continue
            w = num_patches // h
            score1 = abs((w / h) - aspect)
            score2 = abs((h / w) - aspect)
            if score1 < best_score:
                best_score = score1
                best_w, best_h = w, h
            if score2 < best_score:
                best_score = score2
                best_w, best_h = h, w
        return int(best_w), int(best_h)

    @staticmethod
    def _infer_vision_span(flat_ids: torch.Tensor, vision_token_id: int, fallback_num_patches: int) -> Tuple[int, int]:
        vision_positions = torch.where(flat_ids == vision_token_id)[0]
        if vision_positions.numel() == 0:
            raise ValueError(f"Vision token id {vision_token_id} not found in prompt sequence.")
        start = int(vision_positions[0].item())

        run_len = 0
        idx = start
        max_len = flat_ids.numel()
        while idx < max_len and int(flat_ids[idx].item()) == vision_token_id:
            run_len += 1
            idx += 1
        num_patches = run_len if run_len > 4 else int(fallback_num_patches)
        end = start + num_patches
        return start, end

    def generate(self, inputs: Dict[str, Any], max_new_tokens: int, return_attentions: bool = False) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model weights are not loaded; generation is unavailable in this handler instance.")
        return self.model.generate(
            **self._move_inputs_to_model(inputs),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_attentions,
            output_attentions=return_attentions,
        )

    def decode_output(self, sequences: torch.Tensor, input_len: int) -> str:
        trimmed = sequences[0][input_len:]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def extract_attention(
        self,
        generated: Dict[str, Any],
        input_len: int,
        processed_image: Any,
        prompt: str,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        cfg = self._resolve_attention_params(**kwargs)
        return optimized_save_per_layer_head_attention(
            tokenizer=self.tokenizer,
            output_ids=generated,
            input_token_len=input_len,
            processed_image=processed_image,
            processed_prompt=prompt,
            patch_size=int(cfg["patch_size"]),
            merge_size=int(cfg["merge_size"]),
            sequences=generated.get("sequences", None),
            vision_token_id=int(cfg["vision_token_id"]),
        )


    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/\\"))

