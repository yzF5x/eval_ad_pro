from PIL import Image
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
import math
import spacy

from scipy.ndimage import label

import re
from collections import OrderedDict

_STRUCTURE_3x3 = np.ones((3, 3), dtype=np.int32)
_SPACY_PIPELINES: Dict[str, Any] = {}


def _is_internvl_model(model_type: Optional[str], vision_token_id: Optional[int] = None) -> bool:
    model_type_text = str(model_type or "").lower()
    if "internvl" in model_type_text:
        return True
    return vision_token_id in {151671, 151667}


def _resolve_grid_shape(
    width: int,
    height: int,
    patch_size: int,
    merge_size: int,
    model_type: Optional[str] = None,
    vision_token_id: Optional[int] = None,
    grid_height: Optional[int] = None,
    grid_width: Optional[int] = None,
) -> Tuple[int, int]:
    if (grid_height is None) ^ (grid_width is None):
        raise ValueError("grid_height and grid_width must be both provided or both omitted.")
    if grid_height is not None and grid_width is not None:
        gh = int(grid_height)
        gw = int(grid_width)
        if gh <= 0 or gw <= 0:
            raise ValueError(f"Invalid grid shape: {gh}x{gw}")
        return gw, gh
    if _is_internvl_model(model_type, vision_token_id):
        return 16, 16
    return int(width / (patch_size * merge_size)), int(height / (patch_size * merge_size))


def _get_spacy_pipeline(lang_model: str):
    nlp = _SPACY_PIPELINES.get(lang_model)
    if nlp is None:
        nlp = spacy.load(lang_model)
        _SPACY_PIPELINES[lang_model] = nlp
    return nlp


def detect_attn_spike_by_share(
    flatten_text2vision_attn: torch.Tensor,
    spike_patch_idx: int,
    share_thr: float = 0.30,
    eps: float = 1e-12
):
    if flatten_text2vision_attn.dim() == 3:
        x = flatten_text2vision_attn.view(flatten_text2vision_attn.shape[0], -1)
    else:
        x = flatten_text2vision_attn
    x = x.detach().float()

    spike_vals = x[:, spike_patch_idx]
    shares = spike_vals / (x.sum(dim=1) + eps)
    flag = (shares >= share_thr)
    indices = flag.nonzero(as_tuple=True)[0]
    return indices, flag


def detect_single_extreme_values_in_vlm_attn(
    flatten_text2vision_attn: torch.Tensor,
    ratio: float = 50.0,
    dominance_ratio: float = 5.0,
    eps: float = 1e-12,
    topk_spike_patches: Optional[int] = None,
    min_votes: int = 1,
    vote_ratio: float = 0.0,
):
    if flatten_text2vision_attn.dim() == 3:
        x = flatten_text2vision_attn.view(flatten_text2vision_attn.shape[0], -1)
    elif flatten_text2vision_attn.dim() == 2:
        x = flatten_text2vision_attn
    else:
        raise ValueError(f"Unsupported shape: {flatten_text2vision_attn.shape}")

    x = x.detach().float()
    Ntok, P = x.shape
    if P >= 2:
        max2_vals, _ = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
        max1 = max2_vals[:, 0]
        max2 = max2_vals[:, 1] + eps
    else:
        max1 = x[:, 0]
        max2 = torch.full_like(max1, eps)

    medians = x.median(dim=1).values + eps

    ratio1 = max1 / medians
    ratio2 = max1 / max2

    mask = (ratio1 > ratio) & (ratio2 > dominance_ratio)
    outlier_idx = mask.nonzero(as_tuple=True)[0]
    if outlier_idx.numel() == 0:
        if topk_spike_patches is not None:
            return torch.zeros(0, dtype=torch.long, device=x.device), outlier_idx
        return None, outlier_idx

    spike_pos = x[outlier_idx].argmax(dim=1)
    counts = torch.bincount(spike_pos, minlength=P)
    if topk_spike_patches is None:
        spike_patch_idx = int(counts.argmax().item())
        return spike_patch_idx, outlier_idx

    nonzero_patches = (counts > 0).nonzero(as_tuple=True)[0]
    if nonzero_patches.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=x.device), outlier_idx

    min_votes_needed = max(
        int(min_votes),
        int(math.ceil(float(outlier_idx.numel()) * float(vote_ratio))),
    )
    candidate = (counts >= min_votes_needed).nonzero(as_tuple=True)[0]
    if candidate.numel() == 0:
        candidate = nonzero_patches

    ranked = candidate[torch.argsort(counts[candidate], descending=True)]
    topk = min(int(topk_spike_patches), ranked.numel())
    return ranked[:topk], outlier_idx


def detect_entropy_focus_patches_in_vlm_attn(
    flatten_text2vision_attn: torch.Tensor,
    grid_height: int,
    grid_width: int,
    eps: float = 1e-12,
    topk_spike_patches: Optional[int] = None,
    min_votes: int = 1,
    vote_ratio: float = 0.0,
):
    if flatten_text2vision_attn.dim() == 3:
        x = flatten_text2vision_attn.view(flatten_text2vision_attn.shape[0], -1)
    elif flatten_text2vision_attn.dim() == 2:
        x = flatten_text2vision_attn
    else:
        raise ValueError(f"Unsupported shape: {flatten_text2vision_attn.shape}")

    x = x.detach().float()
    Ntok, P = x.shape
    if Ntok == 0 or P == 0:
        empty_idx = torch.zeros(0, dtype=torch.long, device=x.device)
        return empty_idx, empty_idx
    if P != int(grid_height) * int(grid_width):
        raise ValueError(
            f"Patch count {P} does not match grid shape {grid_height}x{grid_width}."
        )

    probs = x.clamp(min=0) / (x.clamp(min=0).sum(dim=1, keepdim=True) + eps)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1) / math.log(max(P, 2))
    finite_entropy = torch.isfinite(entropy)
    if not finite_entropy.any():
        empty_idx = torch.zeros(0, dtype=torch.long, device=x.device)
        return empty_idx, empty_idx

    valid_entropy = entropy[finite_entropy]
    if valid_entropy.numel() <= 2:
        threshold = torch.quantile(valid_entropy, 0.25)
    else:
        try:
            threshold = torch.tensor(
                elbow_chord(valid_entropy.detach().cpu().numpy()),
                device=x.device,
                dtype=entropy.dtype,
            )
        except Exception:
            threshold = torch.quantile(valid_entropy, 0.25)
    threshold = torch.minimum(threshold, torch.tensor(0.5, device=x.device, dtype=entropy.dtype))

    focus_mask = finite_entropy & (entropy <= threshold)
    focus_idx = focus_mask.nonzero(as_tuple=True)[0]
    if focus_idx.numel() == 0:
        empty_idx = torch.zeros(0, dtype=torch.long, device=x.device)
        return empty_idx, focus_idx

    spike_pos = x[focus_idx].argmax(dim=1)
    counts = torch.bincount(spike_pos, minlength=P)
    if topk_spike_patches is None:
        spike_patch_idx = int(counts.argmax().item())
        return spike_patch_idx, focus_idx

    nonzero_patches = (counts > 0).nonzero(as_tuple=True)[0]
    if nonzero_patches.numel() == 0:
        empty_idx = torch.zeros(0, dtype=torch.long, device=x.device)
        return empty_idx, focus_idx

    min_votes_needed = max(
        int(min_votes),
        int(math.ceil(float(focus_idx.numel()) * float(vote_ratio))),
    )
    candidate = (counts >= min_votes_needed).nonzero(as_tuple=True)[0]
    if candidate.numel() == 0:
        candidate = nonzero_patches

    ranked = candidate[torch.argsort(counts[candidate], descending=True)]
    topk = min(int(topk_spike_patches), ranked.numel())
    return ranked[:topk], focus_idx


def get_periphery_mask_fast(
    H: int,
    W: int,
    border_ratio: float,
    device
):
    mask = torch.zeros((H, W), device=device, dtype=torch.float32)

    bh = int(H * border_ratio)
    bw = int(W * border_ratio)

    if bh > 0:
        mask[:bh, :] = 1
        mask[-bh:, :] = 1
    if bw > 0:
        mask[:, :bw] = 1
        mask[:, -bw:] = 1

    return mask


def elbow_chord(values: List[float]) -> float:
    if len(values) <= 2:
        return min(values) if values else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return y[0]
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def combined_weights(values: torch.Tensor, exponent: float = 2.0) -> torch.Tensor:
    normalized_values = (values - values.min()) / (values.max() - values.min())
    weights = torch.exp(exponent * normalized_values)
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    return weights


def get_threshold_and_weight_from_sum(c: torch.Tensor, start: int = 8, end: int = 10) -> torch.Tensor:
    index = torch.zeros(c.shape[0], dtype=torch.float32)
    summed = c[:, start] + c[:, end]
    threshold = torch.quantile(summed, 0.6)

    valid_indices = summed > threshold
    summed_weights = combined_weights(summed)

    index[valid_indices] = summed_weights[valid_indices]
    return index, threshold, summed, summed_weights


def get_par_from_attention_fast(
    c: torch.Tensor,
    border_ratio: float = 0.2,
    grid_height: int = 15,
    grid_width: int = 15
):
    device = c.device
    N = c.shape[0]
    c_map = c.view(N, grid_height, grid_width)
    mask = get_periphery_mask_fast(grid_height, grid_width, border_ratio, device)
    border_sum = (c_map * mask).sum(dim=(1, 2))
    total_sum = c_map.sum(dim=(1, 2))
    par_info = border_sum / total_sum
    return par_info


def aggregate_cross_attentions(
    cross_attentions: torch.Tensor,
    token_weights: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    if cross_attentions.ndim < 1:
        raise ValueError("cross_attentions must have at least 1 dimension with shape (T, ...).")

    T = cross_attentions.shape[0]
    device = cross_attentions.device
    dtype = cross_attentions.dtype
    weights = token_weights.to(device=device).float()

    if weights.ndim != 1 or weights.shape[0] != T:
        try:
            weights = weights.view(T)
        except Exception:
            raise ValueError(f"token_weights must be shape ({T},) or broadcastable to it. Got {token_weights.shape}")

    expand_shape = [T] + [1] * (cross_attentions.ndim - 1)
    weights_view = weights.view(*expand_shape)
    weighted = cross_attentions.float() * weights_view
    agg = weighted.sum(dim=0)

    denom = weights.sum()
    if abs(float(denom)) < eps:
        denom = torch.tensor(eps, device=device, dtype=weights.dtype)
    agg = agg / denom

    if agg.dtype != dtype:
        agg = agg.to(dtype)
    return agg


# def minmax_norm_torch_scaled(
#     x: torch.Tensor,
#     low: float = 0.5,
#     high: float = 1.0,
#     invert: bool = False,
#     eps: float = 1e-8
# ) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         x = torch.tensor(x)

#     x = x.to(dtype=torch.float32)
#     xmin = x.min()
#     xmax = x.max()
#     span = (xmax - xmin).abs()

#     if span < eps:
#         mid = 0.5 * (low + high)
#         return torch.full_like(x, fill_value=mid)

#     norm01 = (x - xmin) / (span + eps)
#     if invert:
#         norm01 = 1.0 - norm01
#     scaled = low + (high - low) * norm01
#     return scaled


def minmax_norm_torch_scaled(
    x: torch.Tensor,
    low: float = 0.5,
    high: float = 1.0,
    invert: bool = False,
    eps: float = 1e-8
) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.to(dtype=torch.float32)
    
    # 1. 初始化结果张量，全为0
    # 这样 inf 值的位置如果不处理，自然就是 0
    result = torch.zeros_like(x)
    
    # 2. 找出有限值的掩码
    finite_mask = torch.isfinite(x)
    
    # 3. 如果存在有限值，仅对有限值进行归一化计算
    if finite_mask.any():
        # 提取有限值
        x_valid = x[finite_mask]
        
        # 基于有限值计算统计量
        xmin = x_valid.min()
        xmax = x_valid.max()
        span = (xmax - xmin).abs()

        if span < eps:
            # 如果有限值范围极小，映射到中间值
            norm_valid = torch.full_like(x_valid, fill_value=0.5)
        else:
            norm_valid = (x_valid - xmin) / (span + eps)

        if invert:
            norm_valid = 1.0 - norm_valid
            
        # 映射到目标区间 [low, high]
        scaled_valid = low + (high - low) * norm_valid
        
        # 4. 将计算好的有限值结果填回结果张量
        result[finite_mask] = scaled_valid
        
    # 如果全为 inf，finite_mask.any() 为 False，result 保持全 0
    return result


def get_weight_with_indices(
    se_list: torch.Tensor,
    sum_list: torch.Tensor,
    valid_index: Optional[torch.Tensor] = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    low: float = 0.5,
    high: float = 1.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = se_list.device
    M = se_list.numel()

    if valid_index is None:
        valid_mask = torch.ones(M, dtype=torch.bool, device=device)
    else:
        vi = torch.as_tensor(valid_index, device=device)
        if vi.dtype == torch.bool:
            if vi.numel() != M:
                raise ValueError("Boolean mask must have same length as se_list/sum_list")
            valid_mask = vi
        else:
            valid_mask = torch.zeros(M, dtype=torch.bool, device=device)
            valid_mask[vi.long()] = True

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    K = valid_indices.numel()

    if K == 0:
        full_weights = torch.zeros(M, device=device, dtype=torch.float32)
        return full_weights, valid_indices, torch.zeros(0, device=device), torch.zeros(0, dtype=torch.long, device=device)

    se_valid = se_list[valid_indices].to(device=device).float()
    sum_valid = sum_list[valid_indices].to(device=device).float()

    sum_norm = minmax_norm_torch_scaled(sum_valid, low=low, high=high, invert=False, eps=eps)
    se_norm = minmax_norm_torch_scaled(se_valid, low=low, high=high, invert=True, eps=eps)

    raw = alpha * sum_norm + beta * se_norm

    if K < 3:
        token_weights = torch.ones_like(raw, device=device, dtype=torch.float32)
    else:
        rmin = raw.min()
        rmax = raw.max()
        if (rmax - rmin).abs() < eps:
            token_weights = torch.ones_like(raw, device=device, dtype=torch.float32)
        else:
            token_weights = (raw - rmin) / (rmax - rmin + eps)

    s = token_weights.sum()
    if s.abs() < eps:
        token_weights = torch.ones_like(token_weights, device=device, dtype=torch.float32) / float(K)
    else:
        token_weights = token_weights / (s + eps)

    full_weights = torch.zeros(M, dtype=torch.float32, device=device)
    full_weights[valid_indices] = token_weights

    sorted_order = torch.argsort(token_weights, descending=True)
    sorted_valid_indices = valid_indices[sorted_order]

    return full_weights, valid_indices, token_weights, sorted_valid_indices


def compute_spatial_consistency_fast(
    cross_attentions: torch.Tensor,
    top_k_percent: int = 10
):
    device = cross_attentions.device
    T, num_patches = cross_attentions.shape
    if T == 0:
        return 0.0

    top_k = max(1, int(num_patches * top_k_percent / 100))
    _, topk_indices = torch.topk(cross_attentions, k=top_k, dim=1, largest=True, sorted=False)

    mask = torch.zeros((T, num_patches), dtype=torch.bool, device=device)
    mask.scatter_(1, topk_indices, True)

    intersection = mask.all(dim=0)
    union = mask.any(dim=0)

    inter_cnt = intersection.sum().item()
    union_cnt = union.sum().item()

    return float(inter_cnt / (union_cnt + 1e-8))


def get_spatial_entropy_from_attention_fast(
    c: torch.Tensor,
    grid_height: int = 15,
    grid_width: int = 15
):
    device = c.device
    N = c.shape[0]

    S = c.view(N, grid_height, grid_width)
    mean_val = S.mean(dim=(1, 2), keepdim=True)
    B = torch.relu(S - mean_val * 2)
    thr = torch.quantile(B.view(N, -1), 0.95, dim=1)
    total = B.sum(dim=(1, 2))

    B_cpu = B.detach().to(torch.float32).cpu().numpy()
    thr_cpu = thr.detach().cpu().numpy()
    total_cpu = total.detach().cpu().numpy()

    se_info = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    se_info_list = [None] * N

    for i in range(N):
        tot = float(total_cpu[i])
        if tot <= 0:
            se_info_list[i] = {"spatial_entropy": float("inf"), "labeled_array": None, "num_components": 0}
            continue

        Bi = B_cpu[i]
        binary = (Bi > thr_cpu[i])
        labeled, num = label(binary, structure=_STRUCTURE_3x3)

        lab_flat = labeled.ravel()
        w_flat = Bi.ravel()
        comp_mass = np.bincount(lab_flat, weights=w_flat)

        probs = comp_mass[1:] / tot
        probs = probs[probs > 0]
        se_raw = float(-(probs * np.log(probs)).sum()) if probs.size > 0 else 0.0

        if (not np.isfinite(se_raw)) or (se_raw <= 1e-7):
            se_val = float("inf")
        else:
            se_val = float(se_raw)

        se_info[i] = se_val
        se_info_list[i] = {"spatial_entropy": float(se_raw), "labeled_array": labeled, "num_components": int(num)}

    valid_se = se_info[torch.isfinite(se_info)]
    try:
        threshold = elbow_chord(valid_se.detach().cpu().numpy())
    except Exception:
        threshold = float("inf")

    valid_indices = se_info < threshold
    return se_info, se_info_list, threshold, valid_indices


def row_normalize(a: torch.Tensor, eps=1e-12) -> torch.Tensor:
    row_sums = a.sum(dim=1, keepdim=True)
    return a / (row_sums + eps)


def get_token_indices_by_pos_and_words(
    text,
    tokenizer,
    lang_model="en_core_web_sm",
    keep_pos=None,
    remove_pos=None,
    explicit_keep_words=None,
    explicit_remove_words=None,
    selection: str = "relevant",
    irrelevant_pos=None,
):
    nlp = _get_spacy_pipeline(lang_model)
    doc = nlp(text)
    if keep_pos is None:
        keep_pos = {"NOUN", "ADJ"}
    if remove_pos is None:
        remove_pos = {"PUNCT", "DET", "CCONJ", "ADV", "X", "SPACE", "ADP"}
    if explicit_keep_words is None:
        explicit_keep_words = set()
    if explicit_remove_words is None:
        explicit_remove_words = {"think", "answer", "addCriterion", "begin_of_box", "end_of_box"}
    if irrelevant_pos is None:
        irrelevant_pos = {"PUNCT", "DET", "CCONJ", "SCONJ", "ADP", "PART", "AUX", "PRON", "INTJ", "NUM"}

    tokens = tokenizer.tokenize(text)
    keep_indices = []
    keep_tokens = []

    if selection == "relevant":
        if keep_pos:
            selected_words = {token.text for token in doc if token.pos_ in keep_pos}
        elif remove_pos:
            selected_words = {token.text for token in doc if token.pos_ not in remove_pos}
        else:
            selected_words = {token.text for token in doc}

        for i, tok in enumerate(tokens):
            clean_tok = tok.lstrip("Ġ▁")
            if clean_tok in explicit_keep_words:
                keep_indices.append(i)
                keep_tokens.append(clean_tok)
            elif clean_tok in selected_words and clean_tok not in explicit_remove_words:
                keep_indices.append(i)
                keep_tokens.append(clean_tok)
        return keep_indices, keep_tokens

    if selection == "irrelevant":
        selected_words = set()
        for token in doc:
            t = token.text.strip()
            if not t:
                continue
            if token.pos_ in irrelevant_pos or token.is_stop or token.is_punct:
                selected_words.add(t)

        selected_words_lower = {w.lower() for w in selected_words}
        explicit_keep_words_lower = {w.lower() for w in explicit_keep_words}

        for i, tok in enumerate(tokens):
            clean_tok = tok.lstrip("Ġ▁")
            clean_lower = clean_tok.lower()
            if (
                clean_tok in selected_words
                or clean_lower in selected_words_lower
                or clean_lower in explicit_keep_words_lower
                or re.fullmatch(r"\W+", clean_tok) is not None
            ):
                keep_indices.append(i)
                keep_tokens.append(clean_tok)
        return keep_indices, keep_tokens

    raise ValueError(f"Unsupported selection mode: {selection}")


def _normalize_sink_head_filter_mode(mode: str) -> str:
    aliases = {
        "all_tokens": "all_tokens",
        "all": "all_tokens",
        "full": "all_tokens",
        "random": "random",
        "random_topk": "random",
        "random_tokens": "random",
        "anomaly_related_topk": "anomaly_related_topk",
        "related_topk": "anomaly_related_topk",
        "anomaly_related": "anomaly_related_topk",
        "related": "anomaly_related_topk",
        "anomaly_unrelated_topk": "anomaly_unrelated_topk",
        "unrelated_topk": "anomaly_unrelated_topk",
        "anomaly_unrelated": "anomaly_unrelated_topk",
        "unrelated": "anomaly_unrelated_topk",
        "pos_content": "pos_content",
        "content": "pos_content",
        "meaningful": "pos_content",
        "pos_meaningful": "pos_content",
        "pos_function": "pos_function",
        "function": "pos_function",
        "functional": "pos_function",
        "pos_irrelevant": "pos_function",
        "irrelevant": "pos_function",
    }
    normalized = aliases.get(str(mode or "").strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported sink_head_token_filter_mode: {mode}. "
            "Use one of {all_tokens, random, anomaly_related_topk, anomaly_unrelated_topk, pos_content, pos_function}."
        )
    return normalized


def _reshape_flatten_attention_by_token(flat_attn: torch.Tensor, output_token_len: int) -> torch.Tensor:
    if flat_attn.dim() != 2:
        raise ValueError(f"Expected 2D flattened attention, got shape: {tuple(flat_attn.shape)}")
    if output_token_len <= 0:
        raise ValueError(f"output_token_len must be positive, got: {output_token_len}")
    rows = int(flat_attn.shape[0])
    if rows % int(output_token_len) != 0:
        raise ValueError(f"Flattened rows {rows} not divisible by output_token_len {output_token_len}.")
    num_head_groups = rows // int(output_token_len)
    return flat_attn.reshape(num_head_groups, int(output_token_len), flat_attn.shape[1])


def _aggregate_token_attention_from_bad_flag(
    flatten_text2vision_attn: torch.Tensor,
    flatten_text2text_attn: torch.Tensor,
    bad_flag: torch.Tensor,
    output_token_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vision_3d = _reshape_flatten_attention_by_token(flatten_text2vision_attn, output_token_len)
    text_3d = _reshape_flatten_attention_by_token(flatten_text2text_attn, output_token_len)
    bad_2d = _reshape_flatten_attention_by_token(
        bad_flag.to(flatten_text2vision_attn.dtype).unsqueeze(1),
        output_token_len,
    ).squeeze(-1) > 0

    valid_2d = (~bad_2d).to(flatten_text2vision_attn.dtype)
    valid_counts = valid_2d.sum(dim=0)
    valid_any = valid_counts > 0

    vis_sum = (vision_3d * valid_2d.unsqueeze(-1)).sum(dim=0)
    txt_sum = (text_3d * valid_2d.unsqueeze(-1)).sum(dim=0)
    denom = valid_counts.clamp(min=1).unsqueeze(-1)
    token_text2vision_attn = vis_sum / denom
    token_text2text_attn = txt_sum / denom

    if (~valid_any).any():
        fallback_vis = vision_3d.mean(dim=0)
        fallback_txt = text_3d.mean(dim=0)
        token_text2vision_attn[~valid_any] = fallback_vis[~valid_any]
        token_text2text_attn[~valid_any] = fallback_txt[~valid_any]

    return token_text2vision_attn, token_text2text_attn, valid_any


def _token_indices_by_similarity_rank(
    flatten_text2text_attn: torch.Tensor,
    output_token_len: int,
    keep_indices_i: List[int],
    with_tag: bool,
    topk: int,
    most_related: bool = True,
) -> List[int]:
    if int(output_token_len) <= 0 or int(topk) <= 0:
        return []
    if flatten_text2text_attn.dim() != 2 or flatten_text2text_attn.shape[1] == 0:
        return []

    text3d = _reshape_flatten_attention_by_token(flatten_text2text_attn, int(output_token_len))
    token_text2text_attn = text3d.mean(dim=0)

    if len(keep_indices_i) > 0:
        valid_keep_indices_i = [i for i in keep_indices_i if i < int(token_text2text_attn.shape[1])]
    else:
        valid_keep_indices_i = list(range(int(token_text2text_attn.shape[1])))
    if len(valid_keep_indices_i) == 0:
        return []

    filtered_prompt2output_text = token_text2text_attn[:, valid_keep_indices_i]
    pref_start, pref_end = (0, 1) if with_tag else (5, 6)
    max_col = int(filtered_prompt2output_text.shape[1]) - 1
    start = min(pref_start, max_col)
    end = min(pref_end, max_col)
    _, _, summed_all, _ = get_threshold_and_weight_from_sum(
        filtered_prompt2output_text, start, end
    )

    k = min(max(int(topk), 1), int(output_token_len))
    ranked_indices = torch.topk(
        summed_all, k=k, largest=bool(most_related), sorted=True
    ).indices
    return ranked_indices.detach().cpu().tolist()


def _select_sink_detection_rows(
    sink_head_token_filter_mode: str,
    sink_head_token_topk: int,
    output_text: str,
    tokenizer,
    flatten_text2text_attn: torch.Tensor,
    keep_indices_i: List[int],
    with_tag: bool,
    output_token_len: int,
    total_rows: int,
    device: torch.device,
) -> torch.Tensor:
    mode = _normalize_sink_head_filter_mode(sink_head_token_filter_mode)

    if mode == "all_tokens":
        token_indices = list(range(int(output_token_len)))
    elif mode == "random":
        k = min(5, int(output_token_len))
        token_indices = torch.randperm(int(output_token_len), device=device)[:k].detach().cpu().tolist()
    elif mode == "anomaly_related_topk":
        token_indices = _token_indices_by_similarity_rank(
            flatten_text2text_attn=flatten_text2text_attn,
            output_token_len=output_token_len,
            keep_indices_i=keep_indices_i,
            with_tag=with_tag,
            topk=sink_head_token_topk,
            most_related=True,
        )
    elif mode == "anomaly_unrelated_topk":
        token_indices = _token_indices_by_similarity_rank(
            flatten_text2text_attn=flatten_text2text_attn,
            output_token_len=output_token_len,
            keep_indices_i=keep_indices_i,
            with_tag=with_tag,
            topk=sink_head_token_topk,
            most_related=False,
        )
    elif mode == "pos_content":
        token_indices, _ = get_token_indices_by_pos_and_words(
            output_text,
            tokenizer,
            keep_pos={"NOUN", "VERB", "ADJ"},
            explicit_remove_words={"think", "answer", "yes", "no"},
        )
    else:
        token_indices, _ = get_token_indices_by_pos_and_words(
            output_text,
            tokenizer,
            selection="irrelevant",
            explicit_keep_words={
                ".", ",", ";", ":", "!", "?", "and", "or", "but",
                "the", "a", "an", "to", "of", "in", "on", "for", "with",
            },
        )

    if len(token_indices) == 0:
        token_indices, _ = get_token_indices_by_pos_and_words(
            output_text,
            tokenizer,
            selection="irrelevant",
            explicit_keep_words={
                ".", ",", ";", ":", "!", "?", "and", "or", "but",
                "the", "a", "an", "to", "of", "in", "on", "for", "with",
            },
        )

    return expand_output_token_indices_to_rows(
        token_indices, output_token_len, total_rows, device=device
    )


def expand_output_token_indices_to_rows(
    token_indices: List[int],
    output_token_len: int,
    total_rows: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if len(token_indices) == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    out_len = int(output_token_len)
    if out_len <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    head_groups = int(total_rows) // out_len
    if head_groups <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    b = torch.arange(head_groups, device=device, dtype=torch.long).unsqueeze(1) * out_len
    token_indices_t = torch.tensor(token_indices, device=device, dtype=torch.long).view(1, -1)
    return (b + token_indices_t).reshape(-1)


def normalize_heatmap(flatten_text2vision_attn_weights, grid_height, height, width, grid_width=15):
    flatten_text2vision_attn_image = flatten_text2vision_attn_weights.reshape((grid_height, grid_width))
    flatten_text2vision_attn_image = flatten_text2vision_attn_image.to(torch.float32)
    flatten_text2vision_attn_image = F.interpolate(
        flatten_text2vision_attn_image.unsqueeze(0).unsqueeze(0), 
        size=(height, width), 
        mode='bicubic'
    ).squeeze()
    attn_over_image_np = flatten_text2vision_attn_image.numpy()
    return attn_over_image_np


def custom_weighted_sum(filtered_flatten_text2vision_attn: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    weights = index.view(-1, 1).float()
    num_to_count = weights.sum()
    weighted_attn = filtered_flatten_text2vision_attn * weights
    result = weighted_attn.sum(dim=0)
    if num_to_count > 0:
        result = result / num_to_count
    result.clamp_(min=0)
    return result


def heatmap_visual(attn_over_image_np, image, title='Attention heatmap overlay', save_name='image.png'):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image) 
    heatmap = ax.imshow(attn_over_image_np, cmap='jet', alpha=0.5)
    plt.title(title)  
    plt.axis('off') 
    plt.savefig(save_name)
    plt.close()
    return fig


def visual_attn_token2image(keep_tokens, filtered_flatten_text2vision_attn, save_name, grid_height, grid_width, height, width, image, summed=None, se_info=None, threshold=None, threshold_se=None, par_info=None, weight_info=None):
    additional_info = True
    if summed is None and se_info is None:
        additional_info = False
    num_tokens = len(keep_tokens)
    grid_cols = 5
    grid_rows = (num_tokens // grid_cols) + (1 if num_tokens % grid_cols != 0 else 0)
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    if additional_info:
        fig.suptitle(f"sum-threshold: {threshold*100:.2f}  se-threshold:{threshold_se:.2f}")
    axs = axs.flatten()
    token_attention_maps = []
    
    for idx, token in enumerate(keep_tokens):
        attn_weights_over_vis_tokens = filtered_flatten_text2vision_attn[idx]
        attn_over_image = normalize_heatmap(attn_weights_over_vis_tokens, grid_height, height, width, grid_width=grid_width)
        axs[idx].imshow(image)
        axs[idx].imshow(attn_over_image, cmap='jet', alpha=0.5)
        fontdict = {"fontsize": 8}
        fontdict["color"] = "red"
        fontdict["weight"] = "bold"
        if additional_info:
            if weight_info is None:
                title = f"{token} sum:{summed[idx]*100:.2f} se:{se_info[idx]:.2f}"
                if summed[idx] >= threshold and se_info[idx] <= threshold_se:
                    if par_info[idx] <= 0.5:
                        title += '|par'
                    axs[idx].set_title(title, fontdict=fontdict)
                else:
                    axs[idx].set_title(title, fontsize=8)
            else:
                mean = 1/weight_info.shape[0]
                title = f"{token} sum:{summed[idx]*100:.2f} se:{se_info[idx]:.2f} w: {weight_info[idx]:.2f}"
                if weight_info[idx] > mean:
                    axs[idx].set_title(title, fontdict=fontdict)
                else:
                    axs[idx].set_title(title, fontsize=8)
        else:
            title = token
            axs[idx].set_title(title, fontsize=8)
        axs[idx].axis('off')
        token_attention_maps.append(fig)
    if additional_info:
        plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(save_name, format='png', dpi=300)
    plt.close() 



def get_fair_normalized_per_layer_head_attention(
    output_ids,
    input_token_len,
    vision_token_start: int,
    vision_token_end: int,
    output_token_start: int,
    output_token_end: int,
    return_full_attn: bool = False,
):
    """
    Rebuild full attention with the original baseline convention:
    mask the BOS column, then row-normalize over the remaining history tokens.
    """
    prompt_attentions = output_ids["attentions"][0]
    num_layers = len(prompt_attentions)
    num_heads = prompt_attentions[0].size(1)
    prompt_len = prompt_attentions[0].size(-1)
    assert prompt_len == input_token_len, f"Expected prompt_len={input_token_len}, got {prompt_len}"

    output_token_len = len(output_ids["attentions"]) - 1
    total_len = input_token_len + output_token_len
    full_attn = torch.zeros(num_layers, num_heads, total_len, total_len, dtype=torch.float32)

    prompt_attn_all = []
    for layer in prompt_attentions:
        attn = layer.squeeze(0).detach().to(torch.float32).cpu()
        cur = attn.clone()
        if cur.size(-1) > 0 and cur.size(-2) > 1:
            cur[:, 1:, 0] = 0.0
            row_sums = cur[:, 1:].sum(dim=-1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            cur[:, 1:] = cur[:, 1:] / row_sums
        prompt_attn_all.append(cur)
    full_attn[:, :, :input_token_len, :input_token_len] = torch.stack(prompt_attn_all)

    for t in range(1, len(output_ids["attentions"])):
        token_idx = input_token_len + (t - 1)
        layer_tuple = output_ids["attentions"][t]
        for layer_idx, layer_attn in enumerate(layer_tuple):
            attn = layer_attn.squeeze(0).detach().to(torch.float32).cpu()
            if attn.dim() == 3:
                if attn.size(1) == 1:
                    historical_attn = attn[:, 0, :token_idx + 1].clone()
                else:
                    historical_attn = attn[:, -1, :token_idx + 1].clone()
            elif attn.dim() == 2:
                historical_attn = attn[:, :token_idx + 1].clone()
            else:
                raise ValueError(f"Unexpected attention shape: {attn.shape}")

            if token_idx + 1 > 1:
                historical_attn[:, 0] = 0.0
                sums = historical_attn.sum(dim=-1, keepdim=True)
                sums[sums == 0] = 1.0
                historical_attn = historical_attn / sums
            full_attn[layer_idx, :, token_idx, :token_idx + 1] = historical_attn

    vlm_attn_raw = full_attn[
        :,
        :,
        output_token_start:output_token_end,
        vision_token_start:vision_token_end,
    ].flatten(start_dim=0, end_dim=1)

    if return_full_attn:
        return vlm_attn_raw, full_attn
    return vlm_attn_raw


def optimized_save_per_layer_head_attention(
    output_ids,
    input_token_len,
    processed_image,
    patch_size=14,
    merge_size=2,
    sequences=None,
    vision_token_id=151655,
    model_type=None,
    grid_height=None,
    grid_width=None,
):
    """
    Save compressed attention using the original baseline normalization convention.
    """
    if sequences is None:
        sequences = output_ids.get("sequences", None)
    if sequences is None:
        raise ValueError("optimized_save_per_layer_head_attention requires sequences or output_ids['sequences'].")

    prompt_attentions = output_ids["attentions"][0]
    num_layers = len(prompt_attentions)
    num_heads = prompt_attentions[0].size(1)

    image = processed_image[-1]
    width, height = image.size
    grid_width, grid_height = _resolve_grid_shape(
        width,
        height,
        patch_size,
        merge_size,
        model_type=model_type,
        vision_token_id=vision_token_id,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    num_patches = int(grid_width * grid_height)

    output_token_len = len(output_ids["attentions"]) - 1
    output_token_start = int(input_token_len)
    output_token_end = int(output_token_start + output_token_len)

    flat_ids = sequences[0, :output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    if not mask.any():
        raise ValueError("Vision token id not found in prompt sequence.")
    vision_token_start = int(torch.where(mask)[0][0].item())
    vision_token_end = int(vision_token_start + num_patches)
    prompt_text_len = int(output_token_start - vision_token_end)
    if prompt_text_len < 0:
        raise ValueError("Computed prompt_text_len < 0. Check vision token span or patch settings.")

    vlm_attn, full_attn = get_fair_normalized_per_layer_head_attention(
        output_ids=output_ids,
        input_token_len=input_token_len,
        vision_token_start=vision_token_start,
        vision_token_end=vision_token_end,
        output_token_start=output_token_start,
        output_token_end=output_token_end,
        return_full_attn=True,
    )

    text2text_attn_raw = full_attn[
        :,
        :,
        output_token_start:output_token_end,
        vision_token_end:output_token_start,
    ]
    flatten_text2vision_attn = row_normalize(vlm_attn.flatten(start_dim=0, end_dim=1))
    flatten_text2text_attn = text2text_attn_raw.flatten(start_dim=0, end_dim=2)

    compressed_attn = {
        "flatten_text2vision_attn": flatten_text2vision_attn,
        "flatten_text2text_attn": flatten_text2text_attn,
    }

    meta = {
        "input_token_len": int(input_token_len),
        "output_token_len": int(output_token_len),
        "output_token_start": int(output_token_start),
        "output_token_end": int(output_token_end),
        "vision_token_start": int(vision_token_start),
        "vision_token_end": int(vision_token_end),
        "num_patches": int(num_patches),
        "prompt_text_len": int(prompt_text_len),
        "layers_num": int(num_layers),
        "heads_num": int(num_heads),
        "patch_size": int(patch_size),
        "merge_size": int(merge_size),
        "vision_token_id": int(vision_token_id),
        "grid_height": int(grid_height),
        "grid_width": int(grid_width),
    }

    return compressed_attn, meta



def evaluate_saved_attention_sink_first_token_mean(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    vision_token_id=151655,
    model_type=None,
    grid_height=None,
    grid_width=None,
    topk_spike_patches=3,
    sink_peak_min_votes=1,
    sink_peak_vote_ratio=0.0,
    sink_head_token_filter_mode="pos_function",
    sink_head_token_topk=5,
    outlier_ratio=50.0,
    dominance_ratio=5.0,
    share_thr=0.3,
):
    sink_head_token_topk = int(sink_head_token_topk)
    if sink_head_token_topk <= 0:
        raise ValueError(f"sink_head_token_topk must be > 0, got: {sink_head_token_topk}")

    image = processed_image[-1]
    width, height = image.size
    grid_width, grid_height = _resolve_grid_shape(
        width,
        height,
        patch_size,
        merge_size,
        model_type=model_type,
        vision_token_id=vision_token_id,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    num_patches = int(grid_width * grid_height)
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len
    to_change = '.' + save_name.split('.')[-1]

    print(f"text start: {output_token_start} ; text end : {output_token_end}")
    flat_ids = sequences[0, :output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    vision_token_start = torch.where(mask)[0][0].item()
    vision_token_end = int(vision_token_start + num_patches)
    print(f"vision start: {vision_token_start} ; vision end : {vision_token_end}")

    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)

    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, tokenizer)

    output_text_list = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text_list, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )
    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Nrow = flatten_text2vision_attn.shape[0]

    sink_filter_row_indices = _select_sink_detection_rows(
        sink_head_token_filter_mode=sink_head_token_filter_mode,
        sink_head_token_topk=sink_head_token_topk,
        output_text=output_text_list,
        tokenizer=tokenizer,
        flatten_text2text_attn=flatten_text2text_attn,
        keep_indices_i=keep_indices_i,
        with_tag=with_tag,
        output_token_len=output_token_len,
        total_rows=Nrow,
        device=device,
    )
    sink_filter_row_indices = sink_filter_row_indices[sink_filter_row_indices < Nrow]

    sink_spike_patch_indices = torch.zeros(0, dtype=torch.long, device=device)
    if sink_filter_row_indices.numel() > 0 and int(topk_spike_patches) > 0:
        sink_spike_patch_indices, _ = detect_single_extreme_values_in_vlm_attn(
            flatten_text2vision_attn[sink_filter_row_indices],
            ratio=outlier_ratio,
            dominance_ratio=dominance_ratio,
            topk_spike_patches=int(topk_spike_patches),
            min_votes=int(sink_peak_min_votes),
            vote_ratio=float(sink_peak_vote_ratio),
        )

    bad_flag = torch.zeros(Nrow, device=device, dtype=torch.bool)
    for patch_idx in sink_spike_patch_indices.tolist():
        _, patch_flag = detect_attn_spike_by_share(
            flatten_text2vision_attn, int(patch_idx), share_thr
        )
        bad_flag |= patch_flag

    outlier_tokens_num = int(bad_flag.sum().item())
    all_tokens_num = int(bad_flag.shape[0])

    token_text2vision_attn, token_text2text_attn, token_has_valid_head = _aggregate_token_attention_from_bad_flag(
        flatten_text2vision_attn=flatten_text2vision_attn,
        flatten_text2text_attn=flatten_text2text_attn,
        bad_flag=bad_flag,
        output_token_len=output_token_len,
    )
    token_text2text_attn = token_text2text_attn.to(token_text2vision_attn.dtype)

    token_valid_all = token_has_valid_head.to(device=device, dtype=torch.bool)
    if not token_valid_all.any():
        token_valid_all = torch.ones(output_token_len, dtype=torch.bool, device=device)
    attn_over_image_np1 = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, token_valid_all),
        grid_height, height, width, grid_width=grid_width
    )

    if len(keep_indices_i) == 0 or token_text2text_attn.shape[1] == 0 or len(keep_indices_o) == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = token_text2text_attn[:, keep_indices_i]
    try:
        row_idx, col_idx = torch.meshgrid(
            torch.tensor(keep_indices_o, device=device, dtype=torch.long),
            torch.tensor(keep_indices_i, device=device, dtype=torch.long),
            indexing='ij',
        )
        _ = token_text2text_attn[row_idx, col_idx]
    except Exception:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    valid_filtered_token = torch.zeros(output_token_len, dtype=torch.bool, device=device)
    valid_filtered_token[torch.tensor(keep_indices_o, device=device, dtype=torch.long)] = True
    valid_filtered_token = valid_filtered_token & token_has_valid_head
    if not valid_filtered_token.any():
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)

    summed = summed_all[valid_filtered_token]
    if summed.numel() == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(token_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    se_info_all, _, _, _ = get_spatial_entropy_from_attention_fast(
        token_text2vision_attn, grid_height=grid_height, grid_width=grid_width
    )
    candidate_se_compute = valid_filtered_token & valid_par_index_all

    valid_se_isfinite = torch.isfinite(se_info_all)
    se_info_valid_indices = valid_sum_index_all & valid_filtered_token & valid_par_index_all & valid_se_isfinite
    se_info = se_info_all[se_info_valid_indices]
    if se_info.numel() == 0:
        threshold_se = 10.0
    else:
        try:
            threshold_se = elbow_chord(se_info.detach().cpu().numpy())
        except Exception:
            threshold_se = 10.0
    valid_se_index_all = se_info_all < threshold_se

    final_valid_index_reasoning = valid_filtered_token.clone()
    conditions = [ 
        valid_sum_index_all, 
        valid_par_index_all, 
        valid_se_isfinite,
        valid_se_index_all
    ]
    for cond in conditions:
        candidate = final_valid_index_reasoning & cond
        if candidate.sum().item() >= 3:
            final_valid_index_reasoning = candidate
        else:
            break

    if final_valid_index_reasoning.sum().item() < 3:
        fallback_mask = candidate_se_compute
        if not fallback_mask.any():
            fallback_mask = valid_filtered_token
        result = normalize_heatmap(
            custom_weighted_sum(token_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    final_keep_tokens = [token_list_decoded[i] for i in final_index.tolist()]

    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        se_info_all, summed_all, final_valid_index_reasoning
    )
    final_valid_attn = token_text2vision_attn[final_valid_index_reasoning]
    final_valid_summed = summed_all[final_valid_index_reasoning]
    final_valid_se_info = se_info_all[final_valid_index_reasoning]
    final_valid_par_info = par_info_all[final_valid_index_reasoning]

    valid_sc = torch.zeros(output_token_len, dtype=torch.bool, device=device)
    try:
        valid_sc_raw = full_weights > (1 / sorted_valid_indices.shape[0])
        if pred_has_anomaly:
            if valid_sc_raw.sum() >= 3:
                valid_sc[sorted_valid_indices[:3]] = True
            else:
                valid_sc[sorted_valid_indices[:2]] = True
        else:
            valid_sc = final_valid_index_reasoning
    except Exception:
        valid_sc = final_valid_index_reasoning

    SC_new = compute_spatial_consistency_fast(token_text2vision_attn[valid_sc][:], top_k_percent=10)
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(final_valid_attn, final_token_weights),
        grid_height, height, width, grid_width=grid_width
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, final_valid_attn,
                save_name.replace(f'{to_change}', '_final_aggreated_attention_token_mean.png'),
                grid_height, grid_width, height, width, image,
                final_valid_summed, final_valid_se_info, threshold, threshold_se,
                final_valid_par_info, final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast_sink_first.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n{output_text_list}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, final_valid_attn,
                save_name.replace(f'{to_change}', '_final_filtered_attention_token_mean.png'),
                grid_height, grid_width, height, width, image,
                final_valid_summed, final_valid_se_info, threshold, threshold_se, final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast_sink_first.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)
    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num



def evaluate_saved_attention_sink_first_entropy_af_token_mean(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    vision_token_id=151655,
    model_type=None,
    grid_height=None,
    grid_width=None,
    topk_spike_patches=3,
    sink_peak_min_votes=1,
    sink_peak_vote_ratio=0.0,
    sink_head_token_filter_mode="pos_function",
    sink_head_token_topk=5,
    outlier_ratio=50.0,
    dominance_ratio=5.0,
    share_thr=0.3,
):
    sink_head_token_topk = int(sink_head_token_topk)
    if sink_head_token_topk <= 0:
        raise ValueError(f"sink_head_token_topk must be > 0, got: {sink_head_token_topk}")

    image = processed_image[-1]
    width, height = image.size
    grid_width, grid_height = _resolve_grid_shape(
        width,
        height,
        patch_size,
        merge_size,
        model_type=model_type,
        vision_token_id=vision_token_id,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    num_patches = int(grid_width * grid_height)
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len
    to_change = '.' + save_name.split('.')[-1]

    print(f"text start: {output_token_start} ; text end : {output_token_end}")
    flat_ids = sequences[0, :output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    vision_token_start = torch.where(mask)[0][0].item()
    vision_token_end = int(vision_token_start + num_patches)
    print(f"vision start: {vision_token_start} ; vision end : {vision_token_end}")

    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)

    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, tokenizer)

    output_text_list = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text_list, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )
    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Nrow = flatten_text2vision_attn.shape[0]

    sink_filter_row_indices = _select_sink_detection_rows(
        sink_head_token_filter_mode=sink_head_token_filter_mode,
        sink_head_token_topk=sink_head_token_topk,
        output_text=output_text_list,
        tokenizer=tokenizer,
        flatten_text2text_attn=flatten_text2text_attn,
        keep_indices_i=keep_indices_i,
        with_tag=with_tag,
        output_token_len=output_token_len,
        total_rows=Nrow,
        device=device,
    )
    sink_filter_row_indices = sink_filter_row_indices[sink_filter_row_indices < Nrow]

    sink_spike_patch_indices = torch.zeros(0, dtype=torch.long, device=device)
    if sink_filter_row_indices.numel() > 0 and int(topk_spike_patches) > 0:
        sink_spike_patch_indices, _ = detect_entropy_focus_patches_in_vlm_attn(
            flatten_text2vision_attn[sink_filter_row_indices],
            grid_height=grid_height,
            grid_width=grid_width,
            topk_spike_patches=int(topk_spike_patches),
            min_votes=int(sink_peak_min_votes),
            vote_ratio=float(sink_peak_vote_ratio),
        )

    bad_flag = torch.zeros(Nrow, device=device, dtype=torch.bool)
    for patch_idx in sink_spike_patch_indices.tolist():
        _, patch_flag = detect_attn_spike_by_share(
            flatten_text2vision_attn, int(patch_idx), share_thr
        )
        bad_flag |= patch_flag

    outlier_tokens_num = int(bad_flag.sum().item())
    all_tokens_num = int(bad_flag.shape[0])

    token_text2vision_attn, token_text2text_attn, token_has_valid_head = _aggregate_token_attention_from_bad_flag(
        flatten_text2vision_attn=flatten_text2vision_attn,
        flatten_text2text_attn=flatten_text2text_attn,
        bad_flag=bad_flag,
        output_token_len=output_token_len,
    )
    token_text2text_attn = token_text2text_attn.to(token_text2vision_attn.dtype)

    token_valid_all = token_has_valid_head.to(device=device, dtype=torch.bool)
    if not token_valid_all.any():
        token_valid_all = torch.ones(output_token_len, dtype=torch.bool, device=device)
    attn_over_image_np1 = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, token_valid_all),
        grid_height, height, width, grid_width=grid_width
    )

    if len(keep_indices_i) == 0 or token_text2text_attn.shape[1] == 0 or len(keep_indices_o) == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = token_text2text_attn[:, keep_indices_i]
    try:
        row_idx, col_idx = torch.meshgrid(
            torch.tensor(keep_indices_o, device=device, dtype=torch.long),
            torch.tensor(keep_indices_i, device=device, dtype=torch.long),
            indexing='ij',
        )
        _ = token_text2text_attn[row_idx, col_idx]
    except Exception:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    valid_filtered_token = torch.zeros(output_token_len, dtype=torch.bool, device=device)
    valid_filtered_token[torch.tensor(keep_indices_o, device=device, dtype=torch.long)] = True
    valid_filtered_token = valid_filtered_token & token_has_valid_head
    if not valid_filtered_token.any():
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)

    summed = summed_all[valid_filtered_token]
    if summed.numel() == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(token_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    se_info_all, _, _, _ = get_spatial_entropy_from_attention_fast(
        token_text2vision_attn, grid_height=grid_height, grid_width=grid_width
    )
    candidate_se_compute = valid_filtered_token & valid_par_index_all

    valid_se_isfinite = torch.isfinite(se_info_all)
    se_info_valid_indices = valid_sum_index_all & valid_filtered_token & valid_par_index_all & valid_se_isfinite
    se_info = se_info_all[se_info_valid_indices]
    if se_info.numel() == 0:
        threshold_se = 10.0
    else:
        try:
            threshold_se = elbow_chord(se_info.detach().cpu().numpy())
        except Exception:
            threshold_se = 10.0
    valid_se_index_all = se_info_all < threshold_se

    final_valid_index_reasoning = valid_filtered_token.clone()
    conditions = [
        valid_sum_index_all,
        valid_par_index_all,
        valid_se_isfinite,
        valid_se_index_all
    ]
    for cond in conditions:
        candidate = final_valid_index_reasoning & cond
        if candidate.sum().item() >= 3:
            final_valid_index_reasoning = candidate
        else:
            break

    if final_valid_index_reasoning.sum().item() < 3:
        fallback_mask = candidate_se_compute
        if not fallback_mask.any():
            fallback_mask = valid_filtered_token
        result = normalize_heatmap(
            custom_weighted_sum(token_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    final_keep_tokens = [token_list_decoded[i] for i in final_index.tolist()]

    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        se_info_all, summed_all, final_valid_index_reasoning
    )
    final_valid_attn = token_text2vision_attn[final_valid_index_reasoning]
    final_valid_summed = summed_all[final_valid_index_reasoning]
    final_valid_se_info = se_info_all[final_valid_index_reasoning]
    final_valid_par_info = par_info_all[final_valid_index_reasoning]

    valid_sc = torch.zeros(output_token_len, dtype=torch.bool, device=device)
    try:
        valid_sc_raw = full_weights > (1 / sorted_valid_indices.shape[0])
        if pred_has_anomaly:
            if valid_sc_raw.sum() >= 3:
                valid_sc[sorted_valid_indices[:3]] = True
            else:
                valid_sc[sorted_valid_indices[:2]] = True
        else:
            valid_sc = final_valid_index_reasoning
    except Exception:
        valid_sc = final_valid_index_reasoning

    SC_new = compute_spatial_consistency_fast(token_text2vision_attn[valid_sc][:], top_k_percent=10)
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(final_valid_attn, final_token_weights),
        grid_height, height, width, grid_width=grid_width
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, final_valid_attn,
                save_name.replace(f'{to_change}', '_final_aggreated_attention_entropy_af_token_mean.png'),
                grid_height, grid_width, height, width, image,
                final_valid_summed, final_valid_se_info, threshold, threshold_se,
                final_valid_par_info, final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast_sink_first.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n{output_text_list}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, final_valid_attn,
                save_name.replace(f'{to_change}', '_final_filtered_attention_entropy_af_token_mean.png'),
                grid_height, grid_width, height, width, image,
                final_valid_summed, final_valid_se_info, threshold, threshold_se, final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast_sink_first.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)
    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


def get_attention_from_saved_ablation(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    vision_token_id=151655,
    model_type=None,
    grid_height=None,
    grid_width=None,
):
    """
    Baseline 版本：直接基于保存的 compressed_attn 做 token-level 评估。
    为适配当前框架，输入与 evaluate_saved_attention_* 保持一致。
    """
    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None:
        flatten_text2vision_attn = compressed_attn.get("vlm_attn", None)
    if flatten_text2text_attn is None:
        flatten_text2text_attn = compressed_attn.get("prompt2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    image = processed_image[-1]
    width, height = image.size
    grid_width, grid_height = _resolve_grid_shape(
        width,
        height,
        patch_size,
        merge_size,
        model_type=model_type,
        vision_token_id=vision_token_id,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    to_change = '.' + save_name.split('.')[-1]
    device = flatten_text2vision_attn.device

    token_list = sequences[0, input_token_len:input_token_len + output_token_len]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)
    output_text = tokenizer.decode(token_list, skip_special_tokens=True)

    text3d = _reshape_flatten_attention_by_token(flatten_text2text_attn, int(output_token_len))
    vision3d = _reshape_flatten_attention_by_token(flatten_text2vision_attn, int(output_token_len))
    token_text2text_attn = text3d.mean(dim=0)
    token_text2vision_attn = vision3d.mean(dim=0)

    flat_ids = sequences[0, :input_token_len].view(-1)
    mask = (flat_ids == vision_token_id)
    if not mask.any():
        fallback = normalize_heatmap(
            custom_weighted_sum(token_text2vision_attn, torch.ones(int(output_token_len), dtype=torch.bool, device=device)),
            grid_height, height, width, grid_width=grid_width,
        )
        return fallback, 1.0, 0, int(output_token_len)

    num_patches = int(grid_width * grid_height)
    vision_token_start = int(torch.where(mask)[0][0].item())
    vision_token_end = int(vision_token_start + num_patches)
    input_text = tokenizer.decode(sequences[0, vision_token_end:input_token_len])
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, tokenizer)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )

    valid_all = torch.ones(int(output_token_len), dtype=torch.bool, device=device)
    attn_over_image_np1 = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, valid_all),
        grid_height, height, width, grid_width=grid_width
    )

    if len(keep_indices_i) == 0 or token_text2text_attn.shape[1] == 0 or len(keep_indices_o) == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, 0, int(output_token_len)

    filtered_prompt2output_text = token_text2text_attn[:, keep_indices_i]
    if with_tag:
        _, _, summed_all, _ = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
    else:
        _, _, summed_all, _ = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)

    valid_filtered_token = torch.zeros(int(output_token_len), dtype=torch.bool, device=device)
    valid_filtered_token[torch.tensor(keep_indices_o, device=device, dtype=torch.long)] = True
    summed = summed_all[valid_filtered_token]
    if summed.numel() == 0:
        return attn_over_image_np1, 1.0, 0, int(output_token_len)
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(token_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5
    se_info_all, _, _, _ = get_spatial_entropy_from_attention_fast(
        token_text2vision_attn, grid_height=grid_height, grid_width=grid_width
    )
    valid_se_isfinite = torch.isfinite(se_info_all)
    se_pool = se_info_all[valid_filtered_token & valid_sum_index_all & valid_par_index_all & valid_se_isfinite]
    if se_pool.numel() == 0:
        threshold_se = 10.0
    else:
        try:
            threshold_se = elbow_chord(se_pool.detach().cpu().numpy())
        except Exception:
            threshold_se = 10.0
    valid_se_index_all = se_info_all < threshold_se

    final_valid_index_reasoning = valid_filtered_token.clone()
    conditions = [
        valid_sum_index_all,
        valid_par_index_all,
        valid_se_isfinite,
        valid_se_index_all,
    ]
    for cond in conditions:
        candidate = final_valid_index_reasoning & cond
        if candidate.sum().item() >= 3:
            final_valid_index_reasoning = candidate
        else:
            break
    if final_valid_index_reasoning.sum().item() < 1:
        final_valid_index_reasoning = valid_filtered_token

    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width
    )

    full_weights, _, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        se_info_all, summed_all, final_valid_index_reasoning
    )
    valid_sc = final_valid_index_reasoning.clone()
    try:
        valid_sc_raw = full_weights > (1 / sorted_valid_indices.shape[0])
        if pred_has_anomaly:
            valid_sc = torch.zeros_like(final_valid_index_reasoning)
            k = 3 if valid_sc_raw.sum() >= 3 else 2
            valid_sc[sorted_valid_indices[:k]] = True
    except Exception:
        valid_sc = final_valid_index_reasoning

    sc = compute_spatial_consistency_fast(
        token_text2vision_attn[valid_sc], top_k_percent=10
    )
    aggregated_image = normalize_heatmap(
        aggregate_cross_attentions(token_text2vision_attn[final_valid_index_reasoning], final_token_weights),
        grid_height, height, width, grid_width=grid_width
    )

    if save_fig:
        final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
        final_keep_tokens = [token_list_decoded[i] for i in final_index.tolist()]
        if final_index.numel() > 0:
            visual_attn_token2image(
                final_keep_tokens,
                token_text2vision_attn[final_valid_index_reasoning],
                save_name.replace(f'{to_change}', '_baseline_final_attention.png'),
                grid_height, grid_width, height, width, image,
                summed_all[final_valid_index_reasoning],
                se_info_all[final_valid_index_reasoning],
                threshold, threshold_se,
                par_info_all[final_valid_index_reasoning],
                final_token_weights,
            )
        save_path = save_name.replace(
            f'{to_change}',
            '_baseline_aggregated.png' if return_aggregate else '_baseline_final.png'
        )
        heatmap_visual(
            aggregated_image if return_aggregate else final_valid_image_reasoning,
            image,
            title=f'SC: {sc:.2f}\n{output_text}',
            save_name=save_path,
        )

    result = aggregated_image if return_aggregate else final_valid_image_reasoning
    return result, sc, 0, int(output_token_len)
