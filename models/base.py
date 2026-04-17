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

    def evaluate_saved_attention(
        self,
        compressed_attn: Any,
        sequences: torch.Tensor,
        input_token_len: int,
        output_token_len: int,
        processed_image: Any,
        processed_prompt: str,
        **kwargs,
    ) -> Tuple[Any, float, int, int]:
        cfg = self._resolve_attention_params(**kwargs)
        patch_size = int(cfg["patch_size"])
        merge_size = int(cfg["merge_size"])
        layers_num = int(cfg["layers_num"])
        heads_num = int(cfg["heads_num"])
        vision_token_id = int(cfg["vision_token_id"])
        return_aggregate = bool(kwargs.get("return_aggregate", kwargs.get("return_aggreagate", False)))
        pred_has_anomaly = kwargs.get("pred_has_anomaly", None)
        save_fig = bool(kwargs.get("save_fig", False))
        with_tag = bool(kwargs.get("with_tag", True))
        save_name = kwargs.get("save_name", "global_attn_heatmap.png")

        image = processed_image[-1]
        width, height = image.size
        fallback_grid_width, fallback_grid_height = self._grid_shape_for_image(
            width=width, height=height, patch_size=patch_size, merge_size=merge_size
        )
        fallback_num_patches = int(fallback_grid_width * fallback_grid_height)
        output_token_start = input_token_len
        output_token_end = output_token_start + output_token_len

        flat_ids = sequences[0, :output_token_start].view(-1)
        vision_token_start, vision_token_end = self._vision_token_span(
            flat_ids=flat_ids,
            vision_token_id=vision_token_id,
            fallback_num_patches=fallback_num_patches,
        )
        num_patches = int(vision_token_end - vision_token_start)
        grid_width, grid_height = self._grid_shape_for_image(
            width=width,
            height=height,
            patch_size=patch_size,
            merge_size=merge_size,
            num_patches=num_patches,
        )

        token_list = sequences[0, output_token_start:output_token_end]
        token_list_decoded = self.tokenizer.batch_decode(token_list, skip_special_tokens=True)

        input_text = self.tokenizer.decode(sequences[0, vision_token_end:output_token_start])
        keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, self.tokenizer)

        output_text = self.tokenizer.decode(token_list, skip_special_tokens=True)
        keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
            output_text,
            self.tokenizer,
            keep_pos={"NOUN"},
            explicit_remove_words={"defect", "defects", "anomaly", "anomalies", "image", "overview",
                                   "analyze", "conclusion", "answer", "think", "Yes", "No"},
        )

        token_list_decoded_extended = token_list_decoded * (layers_num * heads_num)
        b = torch.arange(layers_num * heads_num).unsqueeze(1) * output_token_len
        keep_indices_o = torch.tensor(keep_indices_o)
        keep_indices = b + keep_indices_o
        keep_indices = keep_indices.flatten().tolist()

        vlm_attn = compressed_attn.get("vlm_attn", None)
        if vlm_attn is None:
            vlm_attn = compressed_attn["filtered_vlm_attn"]

        prompt2text_attn_all = compressed_attn.get("prompt2text_attn", None)
        if prompt2text_attn_all is None:
            prompt2text_attn_all = compressed_attn["filtered_prompt2text_attn"]

        outlier_flag = compressed_attn.get("outlier_flag", None)
        bad_flag = compressed_attn.get("bad_flag", None)
        if outlier_flag is None:
            spike_patch_idx = compressed_attn.get("spike_patch_idx", -1)
            if spike_patch_idx == -1:
                spike_patch_idx = None
            _, outlier_idx = detect_single_extreme_values_in_vlm_attn(
                vlm_attn, ratio=50.0, dominance_ratio=5.0
            )
            outlier_flag = torch.zeros(vlm_attn.shape[0], device=vlm_attn.device, dtype=torch.bool)
            if outlier_idx is not None and outlier_idx.numel() > 0:
                outlier_flag[outlier_idx] = True
            if spike_patch_idx is None:
                bad_flag = torch.zeros_like(outlier_flag)
            else:
                _, bad_flag = detect_attn_spike_by_share(vlm_attn, spike_patch_idx, 0.3)

        outlier_tokens_num = int(outlier_flag.sum().item())
        all_tokens_num = int(outlier_flag.shape[0])

        device = vlm_attn.device
        ntok = vlm_attn.shape[0]
        keep_mask = (~outlier_flag) & (~bad_flag)
        if not keep_mask.any():
            keep_mask = torch.ones_like(keep_mask)
        valid_filtered_token = keep_mask

        if len(keep_indices_i) == 0 or prompt2text_attn_all.shape[1] == 0:
            fallback_map = normalize_heatmap(
                custom_weighted_sum(vlm_attn, torch.ones(ntok, dtype=torch.bool, device=device)),
                grid_height, height, width, grid_width=grid_width,
            )
            if save_fig:
                save_path = save_name.replace(".png", "_global_attention.png")
                heatmap_visual(fallback_map, image, title="original_global_attention", save_name=save_path)
            return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

        filtered_prompt2output_text = prompt2text_attn_all[:, keep_indices_i]
        try:
            selected_row_idx = valid_filtered_token.nonzero(as_tuple=True)[0]
            if selected_row_idx.numel() == 0:
                raise ValueError("No kept rows matched output token filter.")
            _ = prompt2text_attn_all[selected_row_idx][:, keep_indices_i]
        except Exception:
            fallback_map = normalize_heatmap(
                custom_weighted_sum(vlm_attn, torch.ones(ntok, dtype=torch.bool, device=device)),
                grid_height, height, width, grid_width=grid_width,
            )
            if save_fig:
                save_path = save_name.replace(".png", "_global_attention.png")
                heatmap_visual(fallback_map, image, title="original_global_attention", save_name=save_path)
            return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

        if with_tag:
            _, _, summed_all, _ = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
        else:
            _, _, summed_all, _ = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)

        summed = summed_all[valid_filtered_token]
        threshold = summed.median()
        valid_sum_index_all = summed_all >= threshold

        par_info_all = get_par_from_attention_fast(vlm_attn, 0.17, grid_height, grid_width)
        valid_par_index_all = par_info_all <= 0.5

        candidate_se_compute = valid_filtered_token & valid_par_index_all
        cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
        candidate_all = valid_filtered_token & valid_par_index_all & valid_sum_index_all
        cand_all_idx = candidate_all.nonzero(as_tuple=True)[0]
        if cand_all_idx.numel() <= 3:
            result = normalize_heatmap(
                custom_weighted_sum(vlm_attn, valid_filtered_token & valid_sum_index_all),
                grid_height, height, width, grid_width=grid_width,
            )
            return result, 1.0, outlier_tokens_num, all_tokens_num

        se_info_all = torch.full((ntok,), float("inf"), device=device, dtype=torch.float32)
        if cand_idx.numel() > 0:
            se_sub, _, _, _ = get_spatial_entropy_from_attention_fast(
                vlm_attn[cand_idx], grid_height=grid_height, grid_width=grid_width
            )
            se_info_all[cand_idx] = se_sub

        valid_se_isfinite = torch.isfinite(se_info_all)
        se_pool_mask = valid_filtered_token & valid_sum_index_all & valid_se_isfinite & valid_par_index_all
        se_pool = se_info_all[se_pool_mask]
        try:
            threshold_se = elbow_chord(se_pool.detach().cpu().numpy())
        except Exception:
            threshold_se = 10.0

        valid_se_index_all = se_info_all < threshold_se
        final_valid_index_reasoning = (
            valid_filtered_token & valid_sum_index_all & valid_se_isfinite &
            valid_se_index_all & valid_par_index_all
        )
        if final_valid_index_reasoning.sum().item() < 3:
            result = normalize_heatmap(
                custom_weighted_sum(vlm_attn, candidate_se_compute),
                grid_height, height, width, grid_width=grid_width,
            )
            return result, 1.0, outlier_tokens_num, all_tokens_num

        final_valid_image_reasoning = normalize_heatmap(
            custom_weighted_sum(vlm_attn, final_valid_index_reasoning.to(torch.int)),
            grid_height, height, width, grid_width=grid_width,
        )
        final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
        full_weights, _, final_token_weights, sorted_valid_indices = get_weight_with_indices(
            se_info_all, summed_all, final_valid_index_reasoning
        )

        topk = min(10, final_token_weights.numel())
        topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=False)
        topk_final_index = final_index[topk_index]
        final_keep_tokens = [token_list_decoded_extended[i % len(token_list_decoded)] for i in topk_final_index.tolist()]

        topk_final_valid_index_reasoning = torch.zeros_like(final_valid_index_reasoning)
        topk_final_valid_index_reasoning[topk_final_index] = True

        valid_sc_raw = full_weights > (1 / sorted_valid_indices.shape[0])
        valid_sc = torch.zeros(summed_all.shape[0], dtype=torch.bool)
        try:
            if pred_has_anomaly:
                if valid_sc_raw.sum() >= 3:
                    valid_sc[sorted_valid_indices[:3]] = True
                else:
                    valid_sc[sorted_valid_indices[:2]] = True
            else:
                valid_sc = final_valid_index_reasoning
        except Exception:
            valid_sc = final_valid_index_reasoning

        sc_new = compute_spatial_consistency_fast(vlm_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
        aggreated_final_image = normalize_heatmap(
            aggregate_cross_attentions(vlm_attn[topk_final_valid_index_reasoning][:], topk_final_token_weights),
            grid_height, height, width, grid_width=grid_width,
        )

        if return_aggregate:
            if save_fig:
                visual_attn_token2image(
                    final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:],
                    save_name.replace(".png", "_final_aggreated_attention_fast.png"),
                    grid_height, grid_width, height, width, image,
                    summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning],
                    threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning], topk_final_token_weights,
                )
                save_path = save_name.replace(".png", "_final_aggreated_image_fast.png")
                heatmap_visual(aggreated_final_image, image, title=f"SC: {sc_new:.2f}\n{output_text}", save_name=save_path)
            return aggreated_final_image, sc_new, outlier_tokens_num, all_tokens_num

        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:],
                save_name.replace(".png", "_final_filtered_attention_fast.png"),
                grid_height, grid_width, height, width, image,
                summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning],
                threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning],
            )
            save_path = save_name.replace(".png", "_final_valid_image_fast.png")
            heatmap_visual(final_valid_image_reasoning, image, title=f"SC: {sc_new:.2f}", save_name=save_path)
        return final_valid_image_reasoning, sc_new, outlier_tokens_num, all_tokens_num

    @property
    def model_name(self) -> str:
        return os.path.basename(self.model_path.rstrip("/\\"))

