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
    xmin = x.min()
    xmax = x.max()
    span = (xmax - xmin).abs()

    if span < eps:
        mid = 0.5 * (low + high)
        return torch.full_like(x, fill_value=mid)

    norm01 = (x - xmin) / (span + eps)
    if invert:
        norm01 = 1.0 - norm01
    scaled = low + (high - low) * norm01
    return scaled


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
    grid_height: int = 15,
    grid_width: int = 15,
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
    grid_width: int = 15,
    zero_eps: float = 1e-7
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

        if num <= 2:
            se_raw = float("inf")
        else:
            probs = comp_mass[1:] / tot
            probs = probs[probs > 0]
            se_raw = float(-(probs * np.log(probs)).sum()) if probs.size > 0 else 0.0

        if (not np.isfinite(se_raw)) or (se_raw <= zero_eps):
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


def expand_output_token_indices_to_rows(
    token_indices: List[int],
    layers_num: int,
    heads_num: int,
    output_token_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if len(token_indices) == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    b = torch.arange(layers_num * heads_num, device=device, dtype=torch.long).unsqueeze(1) * int(output_token_len)
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


def get_saved_per_layer_head_attention(
    tokenizer,
    output_ids,
    input_token_len,
    processed_image,
    processed_prompt,
    patch_size=14,
    merge_size=2,
    **kwargs
):
    prompt_attentions = output_ids["attentions"][0]
    num_layers = len(prompt_attentions)
    num_heads = prompt_attentions[0].size(1)
    prompt_len = prompt_attentions[0].size(-1) 
    print(f"layers : {num_layers} , heads : {num_heads} \n")
    assert prompt_len == input_token_len, f"Expected prompt_len={input_token_len}, got {prompt_len}"

    prompt_attn_all = []
    for layer in prompt_attentions:
        attn = layer.squeeze(0).cpu()
        cur = attn.clone()
        cur[:, 1:, 0] = 0.
        row_sums = cur[:, 1:].sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        cur[:, 1:] = cur[:, 1:] / row_sums
        prompt_attn_all.append(cur)

    prompt_attn_all = torch.stack(prompt_attn_all)

    output_token_len = len(output_ids["attentions"]) - 1
    total_len = input_token_len + output_token_len
    full_attn = torch.zeros(num_layers, num_heads, total_len, total_len)
    full_attn[:, :, :input_token_len, :input_token_len] = prompt_attn_all

    for t in range(1, len(output_ids["attentions"])):
        token_idx = input_token_len + (t - 1)
        layer_tuple = output_ids["attentions"][t]
        for layer_idx, layer_attn in enumerate(layer_tuple):
            attn = layer_attn.squeeze(0).cpu()
            historical_attn = attn[:, 0, :token_idx + 1]
            if token_idx + 1 > 1:
                historical_attn[:, 0] = 0.
                sums = historical_attn.sum(dim=-1, keepdim=True)
                sums[sums == 0] = 1.0
                historical_attn = historical_attn / sums
            full_attn[layer_idx, :, token_idx, :token_idx + 1] = historical_attn

    return full_attn, output_token_len


def get_attention_from_saved_per_layer_head_fast( 
                         tokenizer, 
                         llm_attn_matrix,
                         sequences,
                         input_token_len,
                         output_token_len,
                         processed_image,
                         processed_prompt,
                         return_aggregate=False,
                         patch_size=14,
                         merge_size=2,
                         save_name='global_attn_heatmap',
                         pred_has_anomaly=None,
                         save_fig=False,
                         with_tag=True,
                          layers_num=28,
                          heads_num=28,
                           vision_token_id=151655,
                           **kwargs):
    generated_ids_trimmed = kwargs.pop("generated_ids_trimmed", None)
    output_token_save_path = kwargs.pop("output_token_save_path", None)
    image = processed_image[-1]
    width, height = image.size 
    grid_width, grid_height = int(width / (patch_size*merge_size)), int(height / (patch_size*merge_size)) 
    num_patches = int(grid_width * grid_height)
    
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len 
    print(f"text start: {output_token_start} ; text end : {output_token_end}")
    flat_ids = sequences[0,:output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    vision_token_start = torch.where(mask)[0][0].item() 
    vision_token_end = int(vision_token_start + num_patches)
    print(f"vision start: {vision_token_start} ; vision end : {vision_token_end}")
    
    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)
    
    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, tokenizer)
    
    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(output_text, tokenizer, keep_pos={'NOUN'}, explicit_remove_words={'defect','defects','anomaly','anomalies','image','overview','analyze','conclusion','answer','think','Yes','No'})
    
    token_list_decoded = token_list_decoded * (layers_num * heads_num)
    b = torch.arange(layers_num * heads_num).unsqueeze(1) * output_token_len
    keep_indices_o = torch.tensor(keep_indices_o)
    keep_indices = b + keep_indices_o
    keep_indices = keep_indices.flatten()
    keep_indices = keep_indices.tolist()
    
    flatten_text2vision_attn = llm_attn_matrix[:, :, output_token_start:output_token_end, vision_token_start:vision_token_end]
    flatten_text2vision_attn = flatten_text2vision_attn.flatten(start_dim=0, end_dim=2)
    flatten_text2vision_attn = row_normalize(flatten_text2vision_attn)
    device = flatten_text2vision_attn.device
    Ntok = flatten_text2vision_attn.shape[0]
    
    valid_all = torch.ones(flatten_text2vision_attn.shape[0], dtype=torch.bool)
    attn_over_image_np1 = normalize_heatmap(custom_weighted_sum(flatten_text2vision_attn, valid_all), grid_height, height, width, grid_width=grid_width)
    
    flatten_text2text_attn = llm_attn_matrix[:, :, output_token_start:output_token_end, vision_token_end:output_token_start]
    flatten_text2text_attn = flatten_text2text_attn.flatten(start_dim=0, end_dim=2)
    filtered_prompt2output_text = flatten_text2text_attn[:, keep_indices_i]
    to_change = '.' + save_name.split('.')[-1]
    
    try:
        row_idx, col_idx = torch.meshgrid(torch.tensor(keep_indices), torch.tensor(keep_indices_i), indexing='ij')
        selected_prompt2text = flatten_text2text_attn[row_idx, col_idx]
    except:
        if save_fig:
            save_path = save_name.replace(f'{to_change}','_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, 0, 0

    valid_filtered_token = torch.zeros(Ntok, dtype=torch.bool, device=device)
    valid_filtered_token[keep_indices] = True
    
    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)
    
    summed = summed_all[keep_indices]
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(flatten_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    spike_patch_idx, outlier_idx = detect_single_extreme_values_in_vlm_attn(flatten_text2vision_attn, ratio=50.0, dominance_ratio=5.0)
    outlier_flag = torch.zeros(Ntok, device=flatten_text2vision_attn.device, dtype=torch.bool)
    outlier_flag[outlier_idx] = True
    outlier_tokens_num = sum(outlier_flag)
    all_tokens_num = outlier_flag.shape[0]
    bad_idx, bad_flag = detect_attn_spike_by_share(flatten_text2vision_attn, spike_patch_idx, 0.3)

    candidate_se_compute = valid_filtered_token & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
    candidate_all = valid_filtered_token & valid_par_index_all & valid_sum_index_all & (~outlier_flag) & (~bad_flag)
    cand_all_idx = candidate_all.nonzero(as_tuple=True)[0]
    
    if cand_all_idx.numel() <= 3:
        return normalize_heatmap(custom_weighted_sum(flatten_text2vision_attn, valid_filtered_token & valid_sum_index_all), grid_height, height, width, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
    se_info_all = torch.full((Ntok,), float("inf"), device=device, dtype=torch.float32)
    se_list_all = [
        {
            "spatial_entropy": float("inf"),
            "labeled_array": None,
            "num_components": 0,
            "valid": False,
            "skipped": True
        }
        for _ in range(Ntok)
    ]
    se_sub, se_list_sub, _, _ = get_spatial_entropy_from_attention_fast(
        flatten_text2vision_attn[cand_idx],
        grid_height=grid_height,
        grid_width=grid_width
    )
    se_info_all[cand_idx] = se_sub
    for local_i, global_i in enumerate(cand_idx.tolist()):
        se_list_all[global_i] = {
            **se_list_sub[local_i],
            "valid": True,
            "skipped": False
        }
    valid_se_isfinite = torch.isfinite(se_info_all)
    se_pool_mask = valid_filtered_token & valid_sum_index_all & valid_se_isfinite & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    se_pool = se_info_all[se_pool_mask]
    try:
        threshold_se = elbow_chord(se_pool.detach().cpu().numpy())
    except Exception:
        threshold_se = 10.0
    
    valid_se_index_all = se_info_all < threshold_se
    final_valid_index_reasoning = valid_filtered_token & valid_sum_index_all & valid_se_isfinite & valid_se_index_all & valid_par_index_all & (~outlier_flag) & (~bad_flag)

    if final_valid_index_reasoning.sum().item() < 3:
        return normalize_heatmap(custom_weighted_sum(flatten_text2vision_attn, candidate_se_compute), grid_height, height, width, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(flatten_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    
    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(se_info_all, summed_all, final_valid_index_reasoning)
    
    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=True)
    topk_final_index = final_index[topk_index]
    topk_final_valid_attn = flatten_text2vision_attn[topk_final_index]
    topk_final_valid_summed = summed_all[topk_final_index]
    topk_final_valid_se_info = se_info_all[topk_final_index]
    topk_final_valid_par_info = par_info_all[topk_final_index]
    
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]

    valid_sc_raw = full_weights > (1/sorted_valid_indices.shape[0])
    valid_sc = torch.zeros(summed_all.shape[0], dtype=torch.bool)
    try:
        if pred_has_anomaly:
            if valid_sc_raw.sum() >= 3:
                valid_sc[sorted_valid_indices[:3]] = True
            else:
                valid_sc[sorted_valid_indices[:2]] = True
        else:
            valid_sc = final_valid_index_reasoning
    except:
        valid_sc = final_valid_index_reasoning
    
    SC_new = compute_spatial_consistency_fast(flatten_text2vision_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
    aggreated_final_image = normalize_heatmap(aggregate_cross_attentions(topk_final_valid_attn, topk_final_token_weights), grid_height, height, width, grid_width=grid_width)

    
    if return_aggregate:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, topk_final_valid_attn, save_name.replace('.png','_final_aggreated_attention_fast.png'), grid_height, grid_width, height, width, image, topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se, topk_final_valid_par_info, topk_final_token_weights)
            save_path = save_name.replace(f'{to_change}', f'_final_aggreated_image_fast.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n {output_text}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, topk_final_valid_attn, save_name.replace('.png','_final_filtered_attention_fast.png'), grid_height, grid_width, height, width, image, topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se, topk_final_valid_par_info)
            save_path = save_name.replace(f'{to_change}', f'_final_valid_image_fast.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)

    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


'''尝试先找到异常的tokens，再评估异常tokens的SE和PAR，以及空间一致性'''
def get_attention_from_saved_new( 
                         tokenizer , 
                         llm_attn_matrix ,
                         sequences,
                         input_token_len,
                         output_token_len,
                         processed_image,
                         processed_prompt,
                         return_aggregate=False,
                         patch_size = 14,
                         merge_size = 2,
                         save_name = 'global_attn_heatmap',
                         pred_has_anomaly = None,
                         save_fig = False,
                         with_tag=True,
                         vision_token_id = 151655,
                         **kwargs):
    generated_ids_trimmed = kwargs.pop("generated_ids_trimmed",None)
    output_token_save_path = kwargs.pop("output_token_save_path",None)
    image = processed_image[-1]
    width , height = image.size 
    grid_width , grid_height = int(width / (patch_size*merge_size)) , int(height / (patch_size*merge_size)) 
    num_patches = int(grid_width * grid_height)
    
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len 
    print(f"text start: {output_token_start} ; text end : {output_token_end}")
    flat_ids = sequences[0,:output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    vision_token_start = torch.where(mask)[0][0].item() 
    vision_token_end = int(vision_token_start + num_patches)
    # # identify length or index of tokens
    print(f"vision start: {vision_token_start} ; vision end : {vision_token_end}")
    
    
    # Extract output token IDs and their corresponding decoded text
    output_token_inds = list(range(output_token_start, output_token_end))
    list_decode_all = tokenizer.batch_decode(sequences[0], skip_special_tokens=True)
    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)
    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    
    
    # input token筛选
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text,tokenizer)
    print("keep_tokens_i : ",keep_tokens_i)
    # output token筛选
    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices, keep_tokens = get_token_indices_by_pos_and_words(output_text, tokenizer,keep_pos={'NOUN'}, explicit_remove_words={'defect','defects','anomaly','anomalies','image','overview','analyze','conclusion','answer','think','Yes','No'})
    
    vlm_attn = llm_attn_matrix[output_token_start: output_token_end, vision_token_start:vision_token_end]
    vlm_attn = row_normalize(vlm_attn)
    
    valid_all = torch.ones(vlm_attn.shape[0], dtype=torch.bool)
    attn_over_image_np1 = normalize_heatmap(custom_weighted_sum(vlm_attn,valid_all),grid_height, height, width, gamma_factor=1,grid_width=grid_width)
    
    #依据输入文本和输出文本的重要性找到anomaly token
    prompt2text_attn_all = llm_attn_matrix[output_token_start: output_token_end, vision_token_end:output_token_start]
    filtered_prompt2output_text = prompt2text_attn_all[:,keep_indices_i]
    to_change = '.' + save_name.split('.')[-1]
    try:
        row_idx, col_idx = torch.meshgrid(torch.tensor(keep_indices), torch.tensor(keep_indices_i), indexing='ij')
        selected_prompt2text = prompt2text_attn_all[row_idx, col_idx]
    except:
        if save_fig:
            save_path = save_name.replace(f'{to_change}','_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention',save_name=save_path)
        return attn_over_image_np1, 1.0
    #依据输入文本和输出文本的重要性找到anomaly token
    # index, threshold, summed, summed_weights = get_threshold_and_weight_from_sum(selected_prompt2text,0,1)
    filtered_vlm_attn = vlm_attn[keep_indices][:]
    
    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text,0,1)
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text,5,6)
    
    summed = summed_all[keep_indices]
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold
    # elbow_chord(summed.cpu().numpy())
    par_info_all = get_par_from_attention(vlm_attn,0.17,grid_height,grid_width)
    valid_par_index_all = par_info_all <= 0.5
    valid_filtered_token = torch.zeros(summed_all.shape[0], dtype=torch.bool)
    valid_filtered_token[keep_indices] = True
    
    se_info_all,se_list_all,threshold_se_valid,valid_se_index_all = get_spatial_entropy_from_attention(vlm_attn,grid_height=grid_height, grid_width=grid_width)
    valid_se_isfinite = torch.isfinite(se_info_all)
    se_info_valid_indices = valid_sum_index_all & valid_filtered_token & valid_par_index_all&valid_se_isfinite
    
    se_info = se_info_all[se_info_valid_indices]
    try:
        threshold_se = elbow_chord(se_info.cpu().numpy())
    except:
        threshold_se = 10.0
    valid_se_index_all = se_info_all < threshold_se
    final_valid_index_reasoning = valid_filtered_token.clone()
    
    
    # final_valid_index_reasoning = (valid_filtered_token & valid_sum_index_all & par_info_all & valid_se_isfinite & valid_se_index_all  ).to(torch.int)
    # filtered_image1 = normalize_heatmap(custom_weighted_sum(vlm_attn,final_valid_index_reasoning),grid_height, height, width, gamma_factor=1)
    conditions = [
    valid_sum_index_all,
    valid_se_isfinite,
    valid_par_index_all,
    valid_se_index_all
]
    for i, cond in enumerate(conditions):
    # 尝试添加当前条件
        candidate = final_valid_index_reasoning & cond
    
    # 如果添加后仍有有效索引（即不全为 False）
        if candidate.sum().item() >= 3:
            final_valid_index_reasoning = candidate
        else:
            # 否则跳过该条件，保留之前的结果，并停止后续添加
            break
    # final_valid_index_reasoning = final_valid_index_reasoning.to(torch.int)
    final_valid_image_reasoning = normalize_heatmap(custom_weighted_sum(vlm_attn,final_valid_index_reasoning.to(torch.int)),grid_height, height, width, gamma_factor=1,grid_width=grid_width)
    # final_keep_tokens = token_list_decoded[final_valid_index_reasoning]
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    final_keep_tokens = [token_list_decoded[i] for i in final_index.tolist()]
    SC = compute_spatial_consistency(vlm_attn[final_valid_index_reasoning][:], grid_height, grid_width, top_k_percent=10)
    # final_token_weights = get_weight(se_info_all[final_valid_index_reasoning], summed_all[final_valid_index_reasoning])
    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(se_info_all, summed_all, final_valid_index_reasoning)
    valid_sc_raw = full_weights>(1/sorted_valid_indices.shape[0])
    valid_sc = torch.zeros(summed_all.shape[0], dtype=torch.bool)
    try:
        valid_sc_raw = full_weights>(1/sorted_valid_indices.shape[0])
    
        valid_sc = torch.zeros(summed_all.shape[0], dtype=torch.bool)
        
        content_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
        pred_answer = content_match.group(1).strip() if content_match else output_text.strip()
        pred_yes_no = re.search(r'(yes|no|Yes|No)', pred_answer)
        pred_yes_no = pred_yes_no.group(1) if pred_yes_no else ''
        pred_has_anomaly = True if "yes" in pred_yes_no or 'Yes' in pred_yes_no else False
        if pred_has_anomaly:
            if valid_sc_raw.sum()>=3:
                valid_sc[sorted_valid_indices[:3]]=True
            else:
                valid_sc[sorted_valid_indices[:2]]=True
        else:
            valid_sc = final_valid_index_reasoning
    except:
        valid_sc = final_valid_index_reasoning
    
    # if valid_sc_raw.sum()>=3:
    #     valid_sc[sorted_valid_indices[:3]]=True
    # else:
    #     valid_sc[sorted_valid_indices[:2]]=True
    SC_new = compute_spatial_consistency(vlm_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
    aggreated_final_image = normalize_heatmap(aggregate_cross_attentions(vlm_attn[final_valid_index_reasoning][:],final_token_weights),grid_height, height, width, gamma_factor=1,grid_width=grid_width)
    # filtered_image1 = normalize_heatmap(custom_weighted_sum(vlm_attn,valid_filtered_token),grid_height, height, width, gamma_factor=1,grid_width=grid_width)
    
    if return_aggregate:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, vlm_attn[final_valid_index_reasoning][:], save_name.replace('.png','_final_aggreated_attention.png'), grid_height, grid_width, height, width, image, summed_all[final_valid_index_reasoning], se_info_all[final_valid_index_reasoning], threshold, threshold_se, par_info_all[final_valid_index_reasoning],final_token_weights)
            save_path = save_name.replace(f'{to_change}',f'_final_aggreated_image.png')
            heatmap_visual(aggreated_final_image, image, title=f'final valid image condition-level {i}, SC: {SC_new:.2f}\n {output_text}',save_name=save_path)
        return aggreated_final_image,SC_new
    else:
        if save_fig:
            visual_attn_token2image(keep_tokens, filtered_vlm_attn, save_name.replace('.png','_filtered_tokens_attention.png'), grid_height,grid_width, height, width, image, summed, se_info, threshold, threshold_se, par_info_all[keep_indices])
            visual_attn_token2image(final_keep_tokens, vlm_attn[final_valid_index_reasoning][:], save_name.replace('.png','_final_filtered_attention.png'), grid_height, grid_width, height, width, image, summed_all[final_valid_index_reasoning], se_info_all[final_valid_index_reasoning], threshold, threshold_se, par_info_all[final_valid_index_reasoning])
            save_path = save_name.replace(f'{to_change}',f'_final_valid_image.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'final valid image condition-level {i}, SC: {SC:.2f}',save_name=save_path)

    return final_valid_image_reasoning,SC_new

def optimized_save_per_layer_head_attention(
    tokenizer,
    output_ids,
    input_token_len,
    processed_image,
    processed_prompt,
    patch_size=14,
    merge_size=2,
    **kwargs
):
    """
    保存 attention 矩阵（float32），仅保留后续评估需要的两块 attention。
    """
    sequences = kwargs.pop("sequences", None)
    vision_token_id = kwargs.pop("vision_token_id", 151655)
    model_type = kwargs.pop("model_type", None)
    grid_height = kwargs.pop("grid_height", None)
    grid_width = kwargs.pop("grid_width", None)

    if sequences is None:
        sequences = output_ids.get("sequences", None)
    if sequences is None:
        raise ValueError("optimized_save_per_layer_head_attention requires sequences or output_ids['sequences'].")

    prompt_attentions = output_ids["attentions"][0]
    num_layers = len(prompt_attentions)
    num_heads = prompt_attentions[0].size(1)
    prompt_len = prompt_attentions[0].size(-1)
    assert prompt_len == input_token_len, f"Expected prompt_len={input_token_len}, got {prompt_len}"

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
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len

    flat_ids = sequences[0, :output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    if not mask.any():
        raise ValueError("Vision token id not found in prompt sequence.")
    vision_token_start = torch.where(mask)[0][0].item()
    vision_token_end = int(vision_token_start + num_patches)
    prompt_text_len = int(output_token_start - vision_token_end)
    if prompt_text_len < 0:
        raise ValueError("Computed prompt_text_len < 0. Check vision token span or patch settings.")

    # 提取完整的 attention 矩阵
    attn_to_vision = torch.zeros(
        num_layers, num_heads, output_token_len, num_patches, dtype=torch.float32
    )
    attn_to_text = torch.zeros(
        num_layers, num_heads, output_token_len, prompt_text_len, dtype=torch.float32
    )

    for t in range(1, len(output_ids["attentions"])):
        token_idx = input_token_len + (t - 1)
        layer_tuple = output_ids["attentions"][t]
        for layer_idx, layer_attn in enumerate(layer_tuple):
            attn = layer_attn.squeeze(0)
            if attn.dim() == 3:
                if attn.size(1) == 1:
                    row = attn[:, 0, :token_idx + 1]
                else:
                    row = attn[:, -1, :token_idx + 1]
            elif attn.dim() == 2:
                row = attn[:, :token_idx + 1]
            else:
                raise ValueError(f"Unexpected attention shape: {attn.shape}")

            row = row.detach().to(torch.float32).cpu()
            attn_to_vision[layer_idx, :, t - 1, :] = row[:, vision_token_start:vision_token_end]
            if prompt_text_len > 0:
                attn_to_text[layer_idx, :, t - 1, :] = row[:, vision_token_end:output_token_start]

    # Flatten layer 和 head 维度
    flatten_text2vision_attn = attn_to_vision.flatten(start_dim=0, end_dim=2)
    flatten_text2vision_attn = row_normalize(flatten_text2vision_attn)
    flatten_text2text_attn = attn_to_text.flatten(start_dim=0, end_dim=2)

    # 仅保存后续会用到的两块 attention
    compressed_attn = {
        "flatten_text2vision_attn": flatten_text2vision_attn,
        "flatten_text2text_attn": flatten_text2text_attn,
    }

    meta = {
        "input_token_len": input_token_len,
        "output_token_len": output_token_len,
        "output_token_start": output_token_start,
        "output_token_end": output_token_end,
        "vision_token_start": vision_token_start,
        "vision_token_end": vision_token_end,
        "num_patches": num_patches,
        "prompt_text_len": prompt_text_len,
        "layers_num": num_layers,
        "heads_num": num_heads,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "vision_token_id": vision_token_id,
        "grid_height": grid_height,
        "grid_width": grid_width,
    }

    return compressed_attn, meta


def evaluate_saved_attention_fast(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    processed_prompt,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    layers_num=28,
    heads_num=28,
    vision_token_id=151655,
    **kwargs
):
    """
    从保存的 attention 中进行筛选和可视化。
    outlier 检测在评估阶段执行，保持与 get_attention_from_saved_per_layer_head_fast 一致。
    """
    outlier_ratio = kwargs.pop("outlier_ratio", 50.0)
    dominance_ratio = kwargs.pop("dominance_ratio", 5.0)
    outlier_share_thr = kwargs.pop("outlier_share_thr", 0.3)
    model_type = kwargs.pop("model_type", None)
    grid_height = kwargs.pop("grid_height", None)
    grid_width = kwargs.pop("grid_width", None)

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
    flat_ids = sequences[0,:output_token_start].view(-1)
    mask = (flat_ids == vision_token_id)
    vision_token_start = torch.where(mask)[0][0].item() 
    vision_token_end = int(vision_token_start + num_patches)
    print(f"vision start: {vision_token_start} ; vision end : {vision_token_end}")

    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)

    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    keep_indices_i, keep_tokens_i = get_token_indices_by_pos_and_words(input_text, tokenizer)

    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text, tokenizer, keep_pos={'NOUN'}, 
        explicit_remove_words={'defect','defects','anomaly','anomalies','image','overview',
                              'analyze','conclusion','answer','think','Yes','No'}
    )

    token_list_decoded = token_list_decoded * (layers_num * heads_num)
    b = torch.arange(layers_num * heads_num).unsqueeze(1) * output_token_len
    keep_indices_o = torch.tensor(keep_indices_o)
    keep_indices = b + keep_indices_o
    keep_indices = keep_indices.flatten()
    keep_indices = keep_indices.tolist()

    # 从保存的数据中读取 attention（向后兼容旧 key）
    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None:
        flatten_text2vision_attn = compressed_attn.get("vlm_attn", None)
    if flatten_text2text_attn is None:
        flatten_text2text_attn = compressed_attn.get("prompt2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Ntok = flatten_text2vision_attn.shape[0]

    keep_indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    keep_indices = keep_indices[keep_indices < Ntok]

    spike_patch_idx, outlier_idx = detect_single_extreme_values_in_vlm_attn(
        flatten_text2vision_attn, ratio=outlier_ratio, dominance_ratio=dominance_ratio
    )
    outlier_flag = torch.zeros(Ntok, device=flatten_text2vision_attn.device, dtype=torch.bool)
    if outlier_idx is not None and outlier_idx.numel() > 0:
        outlier_flag[outlier_idx] = True
    outlier_tokens_num = int(outlier_flag.sum().item())
    all_tokens_num = int(outlier_flag.shape[0])

    if spike_patch_idx is None:
        bad_flag = torch.zeros_like(outlier_flag)
    else:
        _, bad_flag = detect_attn_spike_by_share(flatten_text2vision_attn, spike_patch_idx, outlier_share_thr)

    if len(keep_indices_i) == 0 or flatten_text2text_attn.shape[1] == 0 or keep_indices.numel() == 0:
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, torch.ones(Ntok, dtype=torch.bool, device=device)),
            grid_height, height, width, grid_width=grid_width,
        )
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(fallback_map, image, title='original_global_attention', save_name=save_path)
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = flatten_text2text_attn[:, keep_indices_i]

    try:
        row_idx, col_idx = torch.meshgrid(
            keep_indices,
            torch.tensor(keep_indices_i, device=device, dtype=torch.long),
            indexing='ij'
        )
        _ = flatten_text2text_attn[row_idx, col_idx]
    except:
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, torch.ones(Ntok, dtype=torch.bool, device=device)),
            grid_height, height, width, grid_width=grid_width,
        )
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(fallback_map, image, title='original_global_attention', save_name=save_path)
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

    valid_filtered_token = torch.zeros(Ntok, dtype=torch.bool, device=device)
    valid_filtered_token[keep_indices] = True

    # 计算 token 权重
    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 0, 1
        )
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 5, 6
        )

    summed = summed_all[keep_indices]
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    # 计算空间一致性
    par_info_all = get_par_from_attention_fast(flatten_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    candidate_se_compute = valid_filtered_token & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
    candidate_all = valid_filtered_token & valid_par_index_all & valid_sum_index_all & (~outlier_flag) & (~bad_flag)
    cand_all_idx = candidate_all.nonzero(as_tuple=True)[0]

    if cand_all_idx.numel() <= 3:
        result = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, valid_filtered_token & valid_sum_index_all), 
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    # 计算空间熵
    se_info_all = torch.full((Ntok,), float("inf"), device=device, dtype=torch.float32)
    se_list_all = [
        {
            "spatial_entropy": float("inf"),
            "labeled_array": None,
            "num_components": 0,
            "valid": False,
            "skipped": True
        }
        for _ in range(Ntok)
    ]
    
    if cand_idx.numel() > 0:
        se_sub, se_list_sub, _, _ = get_spatial_entropy_from_attention_fast(
            flatten_text2vision_attn[cand_idx],
            grid_height=grid_height,
            grid_width=grid_width
        )
        se_info_all[cand_idx] = se_sub
        for local_i, global_i in enumerate(cand_idx.tolist()):
            se_list_all[global_i] = {
                **se_list_sub[local_i],
                "valid": True,
                "skipped": False
            }
    
    valid_se_isfinite = torch.isfinite(se_info_all)
    se_pool_mask = valid_filtered_token & valid_sum_index_all & valid_se_isfinite & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    se_pool = se_info_all[se_pool_mask]
    
    try:
        threshold_se = elbow_chord(se_pool.detach().cpu().numpy())
    except Exception:
        threshold_se = 10.0

    valid_se_index_all = se_info_all < threshold_se
    final_valid_index_reasoning = (
        valid_filtered_token & valid_sum_index_all & valid_se_isfinite & 
        valid_se_index_all & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    )

    if final_valid_index_reasoning.sum().item() < 3:
        result = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, candidate_se_compute), 
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    # 生成最终可视化
    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(flatten_text2vision_attn, final_valid_index_reasoning.to(torch.int)), 
        grid_height, height, width, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]

    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        se_info_all, summed_all, final_valid_index_reasoning
    )

    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=True)
    topk_final_index = final_index[topk_index]
    topk_final_valid_attn = flatten_text2vision_attn[topk_final_index]
    topk_final_valid_summed = summed_all[topk_final_index]
    topk_final_valid_se_info = se_info_all[topk_final_index]
    topk_final_valid_par_info = par_info_all[topk_final_index]
    
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]

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
    except:
        valid_sc = final_valid_index_reasoning

    SC_new = compute_spatial_consistency_fast(
        flatten_text2vision_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10
    )
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(
            topk_final_valid_attn, topk_final_token_weights
        ), 
        grid_height, height, width, grid_width=grid_width
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn, 
                save_name.replace(f'{to_change}', '_final_aggreated_attention_fast.png'), 
                grid_height, grid_width, height, width, image, 
                topk_final_valid_summed, 
                topk_final_valid_se_info, 
                threshold, threshold_se, 
                topk_final_valid_par_info, 
                topk_final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast.png')
            heatmap_visual(
                aggreated_final_image, image, 
                title=f'SC: {SC_new:.2f}\n{output_text}', 
                save_name=save_path
            )
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn, 
                save_name.replace(f'{to_change}', '_final_filtered_attention_fast.png'), 
                grid_height, grid_width, height, width, image, 
                topk_final_valid_summed, 
                topk_final_valid_se_info, 
                threshold, threshold_se, 
                topk_final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast.png')
            heatmap_visual(
                final_valid_image_reasoning, image, 
                title=f'SC: {SC_new:.2f}', 
                save_name=save_path
            )

    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


def evaluate_saved_attention_sink_first(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    processed_prompt,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    layers_num=28,
    heads_num=28,
    vision_token_id=151655,
    **kwargs
):
    """
    sink-first 版本：
    先从语义无关 token 对应行里提取峰值视觉 token（可配置多个），
    再在全量行中用这些视觉 token 识别被吸附行（bad_flag），
    最后复用 fast 版本的筛选、加权融合与可视化流程。
    """
    outlier_ratio = kwargs.pop("outlier_ratio", 50.0)
    dominance_ratio = kwargs.pop("dominance_ratio", 5.0)
    outlier_share_thr = kwargs.pop("outlier_share_thr", 0.3)
    model_type = kwargs.pop("model_type", None)
    grid_height = kwargs.pop("grid_height", None)
    grid_width = kwargs.pop("grid_width", None)
    topk_spike_patches = kwargs.pop("topk_spike_patches", kwargs.pop("sink_peak_topk", 3))
    sink_peak_min_votes = kwargs.pop("sink_peak_min_votes", 2)
    sink_peak_vote_ratio = kwargs.pop("sink_peak_vote_ratio", 0.15)

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

    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )
    irrelevant_indices_o, _ = get_token_indices_by_pos_and_words(
        output_text,
        tokenizer,
        selection="irrelevant",
        explicit_keep_words={
            ".", ",", ";", ":", "!", "?", "and", "or", "but",
            "the", "a", "an", "to", "of", "in", "on", "for", "with",
        },
    )

    token_list_decoded = token_list_decoded * (layers_num * heads_num)

    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None:
        flatten_text2vision_attn = compressed_attn.get("vlm_attn", None)
    if flatten_text2text_attn is None:
        flatten_text2text_attn = compressed_attn.get("prompt2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Ntok = flatten_text2vision_attn.shape[0]

    keep_indices = expand_output_token_indices_to_rows(
        keep_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    keep_indices = keep_indices[keep_indices < Ntok]

    irrelevant_row_indices = expand_output_token_indices_to_rows(
        irrelevant_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    irrelevant_row_indices = irrelevant_row_indices[irrelevant_row_indices < Ntok]

    sink_spike_patch_indices = torch.zeros(0, dtype=torch.long, device=device)
    if irrelevant_row_indices.numel() > 0 and int(topk_spike_patches) > 0:
        sink_spike_patch_indices, _ = detect_single_extreme_values_in_vlm_attn(
            flatten_text2vision_attn[irrelevant_row_indices],
            ratio=outlier_ratio,
            dominance_ratio=dominance_ratio,
            topk_spike_patches=int(topk_spike_patches),
            min_votes=int(sink_peak_min_votes),
            vote_ratio=float(sink_peak_vote_ratio),
        )

    bad_flag = torch.zeros(Ntok, device=device, dtype=torch.bool)
    for patch_idx in sink_spike_patch_indices.tolist():
        _, patch_flag = detect_attn_spike_by_share(
            flatten_text2vision_attn, int(patch_idx), outlier_share_thr
        )
        bad_flag |= patch_flag

    outlier_tokens_num = int(bad_flag.sum().item())
    all_tokens_num = int(bad_flag.shape[0])

    if len(keep_indices_i) == 0 or flatten_text2text_attn.shape[1] == 0 or keep_indices.numel() == 0:
        fallback_mask = (~bad_flag)
        if not fallback_mask.any():
            fallback_mask = torch.ones(Ntok, dtype=torch.bool, device=device)
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width,
        )
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(fallback_map, image, title='original_global_attention', save_name=save_path)
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = flatten_text2text_attn[:, keep_indices_i]

    try:
        _ = flatten_text2text_attn[keep_indices][:, keep_indices_i]
    except Exception:
        fallback_mask = (~bad_flag)
        if not fallback_mask.any():
            fallback_mask = torch.ones(Ntok, dtype=torch.bool, device=device)
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width,
        )
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(fallback_map, image, title='original_global_attention', save_name=save_path)
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

    valid_filtered_token = torch.zeros(Ntok, dtype=torch.bool, device=device)
    valid_filtered_token[keep_indices] = True
    valid_filtered_token = valid_filtered_token & (~bad_flag)
    if not valid_filtered_token.any():
        fallback_mask = (~bad_flag)
        if not fallback_mask.any():
            fallback_mask = torch.ones(Ntok, dtype=torch.bool, device=device)
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width,
        )
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num

    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 0, 1
        )
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 5, 6
        )

    summed = summed_all[valid_filtered_token]
    if summed.numel() == 0:
        fallback_mask = (~bad_flag)
        if not fallback_mask.any():
            fallback_mask = torch.ones(Ntok, dtype=torch.bool, device=device)
        fallback_map = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width,
        )
        return fallback_map, 1.0, outlier_tokens_num, all_tokens_num
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(flatten_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    candidate_se_compute = valid_filtered_token & valid_par_index_all
    cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
    candidate_all = valid_filtered_token & valid_par_index_all & valid_sum_index_all
    cand_all_idx = candidate_all.nonzero(as_tuple=True)[0]

    if cand_all_idx.numel() <= 3:
        result = normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, valid_filtered_token & valid_sum_index_all),
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    se_info_all = torch.full((Ntok,), float("inf"), device=device, dtype=torch.float32)
    se_list_all = [
        {
            "spatial_entropy": float("inf"),
            "labeled_array": None,
            "num_components": 0,
            "valid": False,
            "skipped": True
        }
        for _ in range(Ntok)
    ]

    if cand_idx.numel() > 0:
        se_sub, se_list_sub, _, _ = get_spatial_entropy_from_attention_fast(
            flatten_text2vision_attn[cand_idx],
            grid_height=grid_height,
            grid_width=grid_width
        )
        se_info_all[cand_idx] = se_sub
        for local_i, global_i in enumerate(cand_idx.tolist()):
            se_list_all[global_i] = {
                **se_list_sub[local_i],
                "valid": True,
                "skipped": False
            }

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
            custom_weighted_sum(flatten_text2vision_attn, candidate_se_compute),
            grid_height, height, width, grid_width=grid_width
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(flatten_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]

    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        se_info_all, summed_all, final_valid_index_reasoning
    )

    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=True)
    topk_final_index = final_index[topk_index]
    topk_final_valid_attn = flatten_text2vision_attn[topk_final_index]
    topk_final_valid_summed = summed_all[topk_final_index]
    topk_final_valid_se_info = se_info_all[topk_final_index]
    topk_final_valid_par_info = par_info_all[topk_final_index]
    
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]

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
    except:
        valid_sc = final_valid_index_reasoning

    SC_new = compute_spatial_consistency_fast(
        flatten_text2vision_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10
    )
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(
            topk_final_valid_attn, topk_final_token_weights
        ),
        grid_height, height, width, grid_width=grid_width
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_aggreated_attention_fast_sink_first.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed,
                topk_final_valid_se_info,
                threshold, threshold_se,
                topk_final_valid_par_info,
                topk_final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast_sink_first.png')
            heatmap_visual(
                aggreated_final_image, image,
                title=f'SC: {SC_new:.2f}\n{output_text}', save_name=save_path
            )
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_filtered_attention_fast_sink_first.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed,
                topk_final_valid_se_info,
                threshold, threshold_se,
                topk_final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast_sink_first.png')
            heatmap_visual(
                final_valid_image_reasoning, image,
                title=f'SC: {SC_new:.2f}', save_name=save_path
            )
    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


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
    bad_2d = _reshape_flatten_attention_by_token(bad_flag.to(flatten_text2vision_attn.dtype).unsqueeze(1), output_token_len).squeeze(-1) > 0

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


def evaluate_saved_attention_sink_first_token_mean(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    processed_prompt,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    layers_num=28,
    heads_num=28,
    vision_token_id=151655,
    **kwargs
):
    outlier_ratio = kwargs.pop("outlier_ratio", 50.0)
    dominance_ratio = kwargs.pop("dominance_ratio", 5.0)
    outlier_share_thr = kwargs.pop("outlier_share_thr", 0.3)
    model_type = kwargs.pop("model_type", None)
    grid_height = kwargs.pop("grid_height", None)
    grid_width = kwargs.pop("grid_width", None)
    topk_spike_patches = kwargs.pop("topk_spike_patches", kwargs.pop("sink_peak_topk", 3))
    sink_peak_min_votes = kwargs.pop("sink_peak_min_votes", 1)
    sink_peak_vote_ratio = kwargs.pop("sink_peak_vote_ratio", 0.0)

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

    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, keep_tokens_o = get_token_indices_by_pos_and_words(
        output_text, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )
    irrelevant_indices_o, _ = get_token_indices_by_pos_and_words(
        output_text,
        tokenizer,
        selection="irrelevant",
        explicit_keep_words={
            ".", ",", ";", ":", "!", "?", "and", "or", "but",
            "the", "a", "an", "to", "of", "in", "on", "for", "with",
        },
    )

    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Nrow = flatten_text2vision_attn.shape[0]

    keep_indices = expand_output_token_indices_to_rows(
        keep_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    keep_indices = keep_indices[keep_indices < Nrow]

    irrelevant_row_indices = expand_output_token_indices_to_rows(
        irrelevant_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    irrelevant_row_indices = irrelevant_row_indices[irrelevant_row_indices < Nrow]

    sink_spike_patch_indices = torch.zeros(0, dtype=torch.long, device=device)
    if irrelevant_row_indices.numel() > 0 and int(topk_spike_patches) > 0:
        sink_spike_patch_indices, _ = detect_single_extreme_values_in_vlm_attn(
            flatten_text2vision_attn[irrelevant_row_indices],
            ratio=outlier_ratio,
            dominance_ratio=dominance_ratio,
            topk_spike_patches=int(topk_spike_patches),
            min_votes=int(sink_peak_min_votes),
            vote_ratio=float(sink_peak_vote_ratio),
        )

    bad_flag = torch.zeros(Nrow, device=device, dtype=torch.bool)
    for patch_idx in sink_spike_patch_indices.tolist():
        _, patch_flag = detect_attn_spike_by_share(
            flatten_text2vision_attn, int(patch_idx), outlier_share_thr
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
    token_text2vision_attn = row_normalize(token_text2vision_attn)
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

    se_info_all = torch.full((output_token_len,), float("inf"), device=device, dtype=torch.float32)
    se_list_all = [
        {"spatial_entropy": float("inf"), "labeled_array": None, "num_components": 0, "valid": False, "skipped": True}
        for _ in range(output_token_len)
    ]
    candidate_se_compute = valid_filtered_token & valid_par_index_all
    cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
    if cand_idx.numel() > 0:
        se_sub, se_list_sub, _, _ = get_spatial_entropy_from_attention_fast(
            token_text2vision_attn[cand_idx], grid_height=grid_height, grid_width=grid_width
        )
        se_info_all[cand_idx] = se_sub
        for local_i, global_i in enumerate(cand_idx.tolist()):
            se_list_all[global_i] = {**se_list_sub[local_i], "valid": True, "skipped": False}

    valid_se_isfinite = torch.isfinite(se_info_all)
    se_info_valid_indices = valid_sum_index_all & valid_filtered_token & valid_par_index_all & valid_se_isfinite
    se_info = se_info_all[se_info_valid_indices]
    try:
        threshold_se = elbow_chord(se_info.detach().cpu().numpy())
    except Exception:
        threshold_se = 10.0
    valid_se_index_all = se_info_all < threshold_se

    final_valid_index_reasoning = valid_filtered_token.clone()
    conditions = [valid_sum_index_all, valid_se_isfinite, valid_par_index_all, valid_se_index_all]
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

    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=True)
    topk_final_index = final_index[topk_index]
    topk_final_valid_attn = token_text2vision_attn[topk_final_index]
    topk_final_valid_summed = summed_all[topk_final_index]
    topk_final_valid_se_info = se_info_all[topk_final_index]
    topk_final_valid_par_info = par_info_all[topk_final_index]
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]

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

    SC_new = compute_spatial_consistency_fast(token_text2vision_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(topk_final_valid_attn, topk_final_token_weights),
        grid_height, height, width, grid_width=grid_width
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_aggreated_attention_fast_sink_first_token_mean.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se,
                topk_final_valid_par_info, topk_final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast_sink_first.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n{output_text}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_filtered_attention_fast_sink_first_token_mean.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se, topk_final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast_sink_first.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)
    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


def _evaluate_saved_attention_sink_first_token_se_rank(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    processed_prompt,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    layers_num=28,
    heads_num=28,
    vision_token_id=151655,
    **kwargs
):
    outlier_ratio = kwargs.pop("outlier_ratio", 50.0)
    dominance_ratio = kwargs.pop("dominance_ratio", 5.0)
    outlier_share_thr = kwargs.pop("outlier_share_thr", 0.3)
    model_type = kwargs.pop("model_type", None)
    grid_height = kwargs.pop("grid_height", None)
    grid_width = kwargs.pop("grid_width", None)
    topk_spike_patches = kwargs.pop("topk_spike_patches", 1)
    sink_peak_min_votes = kwargs.pop("sink_peak_min_votes", 1)
    sink_peak_vote_ratio = kwargs.pop("sink_peak_vote_ratio", 0.0)

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
    keep_indices_i, _ = get_token_indices_by_pos_and_words(input_text, tokenizer)

    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    keep_indices_o, _ = get_token_indices_by_pos_and_words(
        output_text, tokenizer, keep_pos={'NOUN'},
        explicit_remove_words={'defect', 'defects', 'anomaly', 'anomalies', 'image', 'overview',
                               'analyze', 'conclusion', 'answer', 'think', 'Yes', 'No'}
    )
    irrelevant_indices_o, _ = get_token_indices_by_pos_and_words(
        output_text,
        tokenizer,
        selection="irrelevant",
        explicit_keep_words={
            ".", ",", ";", ":", "!", "?", "and", "or", "but",
            "the", "a", "an", "to", "of", "in", "on", "for", "with",
        },
    )

    flatten_text2vision_attn = compressed_attn.get("flatten_text2vision_attn", None)
    flatten_text2text_attn = compressed_attn.get("flatten_text2text_attn", None)
    if flatten_text2vision_attn is None:
        flatten_text2vision_attn = compressed_attn.get("vlm_attn", None)
    if flatten_text2text_attn is None:
        flatten_text2text_attn = compressed_attn.get("prompt2text_attn", None)
    if flatten_text2vision_attn is None or flatten_text2text_attn is None:
        raise ValueError("compressed_attn must contain both vision and text attention tensors.")

    device = flatten_text2vision_attn.device
    Nrow = flatten_text2vision_attn.shape[0]

    keep_indices = expand_output_token_indices_to_rows(
        keep_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    keep_indices = keep_indices[keep_indices < Nrow]

    irrelevant_row_indices = expand_output_token_indices_to_rows(
        irrelevant_indices_o, layers_num, heads_num, output_token_len, device=device
    )
    irrelevant_row_indices = irrelevant_row_indices[irrelevant_row_indices < Nrow]

    sink_spike_patch_indices = torch.zeros(0, dtype=torch.long, device=device)
    if irrelevant_row_indices.numel() > 0 and int(topk_spike_patches) > 0:
        sink_spike_patch_indices, _ = detect_single_extreme_values_in_vlm_attn(
            flatten_text2vision_attn[irrelevant_row_indices],
            ratio=outlier_ratio,
            dominance_ratio=dominance_ratio,
            topk_spike_patches=int(topk_spike_patches),
            min_votes=int(sink_peak_min_votes),
            vote_ratio=float(sink_peak_vote_ratio),
        )

    bad_flag = torch.zeros(Nrow, device=device, dtype=torch.bool)
    for patch_idx in sink_spike_patch_indices.tolist():
        _, patch_flag = detect_attn_spike_by_share(
            flatten_text2vision_attn, int(patch_idx), outlier_share_thr
        )
        bad_flag |= patch_flag

    outlier_tokens_num = int(bad_flag.sum().item())
    all_tokens_num = int(bad_flag.shape[0])

    def _row_fallback_map(mask: torch.Tensor):
        fallback_mask = mask
        if fallback_mask.numel() == 0 or not fallback_mask.any():
            fallback_mask = torch.ones(Nrow, dtype=torch.bool, device=device)
        return normalize_heatmap(
            custom_weighted_sum(flatten_text2vision_attn, fallback_mask),
            grid_height, height, width, grid_width=grid_width,
        )

    row_valid_base = (~bad_flag)
    attn_over_image_np1 = _row_fallback_map(row_valid_base)

    if len(keep_indices_i) == 0 or flatten_text2text_attn.shape[1] == 0 or keep_indices.numel() == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = flatten_text2text_attn[:, keep_indices_i]
    try:
        _ = flatten_text2text_attn[keep_indices][:, keep_indices_i]
    except Exception:
        if save_fig:
            save_path = save_name.replace(f'{to_change}', '_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    valid_filtered_row = torch.zeros(Nrow, dtype=torch.bool, device=device)
    valid_filtered_row[keep_indices] = True
    valid_filtered_row = valid_filtered_row & (~bad_flag)
    if not valid_filtered_row.any():
        result = _row_fallback_map(row_valid_base)
        return result, 1.0, outlier_tokens_num, all_tokens_num

    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 0, 1
        )
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text, 5, 6
        )

    summed = summed_all[valid_filtered_row]
    if summed.numel() == 0:
        result = _row_fallback_map(row_valid_base)
        return result, 1.0, outlier_tokens_num, all_tokens_num
    threshold = summed.median()
    valid_sum_index_all = summed_all >= threshold

    par_info_all = get_par_from_attention_fast(flatten_text2vision_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    candidate_rows = valid_filtered_row & valid_sum_index_all & valid_par_index_all
    cand_idx = candidate_rows.nonzero(as_tuple=True)[0]
    if cand_idx.numel() == 0:
        result = _row_fallback_map(valid_filtered_row & valid_par_index_all)
        return result, 1.0, outlier_tokens_num, all_tokens_num

    se_info_all = torch.full((Nrow,), float("inf"), device=device, dtype=torch.float32)
    se_sub, _, _, _ = get_spatial_entropy_from_attention_fast(
        flatten_text2vision_attn[cand_idx],
        grid_height=grid_height,
        grid_width=grid_width,
    )
    se_info_all[cand_idx] = se_sub

    valid_se_isfinite = torch.isfinite(se_info_all)
    se_pool = se_info_all[candidate_rows & valid_se_isfinite]
    try:
        threshold_se = elbow_chord(se_pool.detach().cpu().numpy())
    except Exception:
        threshold_se = 10.0
    auto_se_rows = candidate_rows & valid_se_isfinite & (se_info_all < threshold_se)

    row_token_idx = torch.arange(Nrow, device=device, dtype=torch.long) % int(output_token_len)
    selected_rows = torch.zeros(Nrow, dtype=torch.bool, device=device)
    for token_idx in range(int(output_token_len)):
        token_auto = (row_token_idx == token_idx) & auto_se_rows
        if token_auto.any():
            token_candidates = token_auto.nonzero(as_tuple=True)[0]
        else:
            token_candidates = ((row_token_idx == token_idx) & candidate_rows & valid_se_isfinite).nonzero(as_tuple=True)[0]
        if token_candidates.numel() == 0:
            continue
        best_local = torch.argmin(se_info_all[token_candidates])
        selected_rows[token_candidates[best_local]] = True

    if not selected_rows.any():
        result = _row_fallback_map(candidate_rows)
        return result, 1.0, outlier_tokens_num, all_tokens_num

    token_represent_mask = torch.zeros(output_token_len, dtype=torch.bool, device=device)
    token_text2vision_attn = torch.zeros(
        (output_token_len, flatten_text2vision_attn.shape[1]),
        dtype=flatten_text2vision_attn.dtype,
        device=device,
    )
    token_text2text_attn = torch.zeros(
        (output_token_len, flatten_text2text_attn.shape[1]),
        dtype=flatten_text2text_attn.dtype,
        device=device,
    )
    token_summed_all = torch.zeros(output_token_len, dtype=torch.float32, device=device)
    token_par_all = torch.full((output_token_len,), float("inf"), dtype=torch.float32, device=device)
    token_se_all = torch.full((output_token_len,), float("inf"), dtype=torch.float32, device=device)

    selected_rows_idx = selected_rows.nonzero(as_tuple=True)[0]
    for row_idx in selected_rows_idx.tolist():
        token_idx = int(row_idx % int(output_token_len))
        token_represent_mask[token_idx] = True
        token_text2vision_attn[token_idx] = flatten_text2vision_attn[row_idx]
        token_text2text_attn[token_idx] = flatten_text2text_attn[row_idx]
        token_summed_all[token_idx] = summed_all[row_idx]
        token_par_all[token_idx] = par_info_all[row_idx]
        token_se_all[token_idx] = se_info_all[row_idx]

    if not token_represent_mask.any():
        result = _row_fallback_map(candidate_rows)
        return result, 1.0, outlier_tokens_num, all_tokens_num

    final_valid_index_reasoning = token_represent_mask
    if final_valid_index_reasoning.sum().item() < 3:
        result = normalize_heatmap(
            custom_weighted_sum(token_text2vision_attn, final_valid_index_reasoning),
            grid_height, height, width, grid_width=grid_width,
        )
        return result, 1.0, outlier_tokens_num, all_tokens_num

    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(token_text2vision_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, grid_width=grid_width,
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]

    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(
        token_se_all, token_summed_all, final_valid_index_reasoning
    )

    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=True)
    topk_final_index = final_index[topk_index]
    topk_final_valid_attn = token_text2vision_attn[topk_final_index]
    topk_final_valid_summed = token_summed_all[topk_final_index]
    topk_final_valid_se_info = token_se_all[topk_final_index]
    topk_final_valid_par_info = token_par_all[topk_final_index]
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]

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

    SC_new = compute_spatial_consistency_fast(
        token_text2vision_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10
    )
    aggreated_final_image = normalize_heatmap(
        aggregate_cross_attentions(topk_final_valid_attn, topk_final_token_weights),
        grid_height, height, width, grid_width=grid_width,
    )

    if return_aggregate:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_aggreated_attention_fast_sink_first_se_rank.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se,
                topk_final_valid_par_info, topk_final_token_weights
            )
            save_path = save_name.replace(f'{to_change}', '_final_aggreated_image_fast_sink_first.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n{output_text}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(
                final_keep_tokens, topk_final_valid_attn,
                save_name.replace(f'{to_change}', '_final_filtered_attention_fast_sink_first_se_rank.png'),
                grid_height, grid_width, height, width, image,
                topk_final_valid_summed, topk_final_valid_se_info, threshold, threshold_se,
                topk_final_valid_par_info
            )
            save_path = save_name.replace(f'{to_change}', '_final_valid_image_fast_sink_first.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)
    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


def evaluate_saved_attention_sink_first_token_aggregate(
    tokenizer,
    compressed_attn,
    sequences,
    input_token_len,
    output_token_len,
    processed_image,
    processed_prompt,
    return_aggregate=False,
    patch_size=14,
    merge_size=2,
    save_name='global_attn_heatmap',
    pred_has_anomaly=None,
    save_fig=False,
    with_tag=True,
    layers_num=28,
    heads_num=28,
    vision_token_id=151655,
    token_aggregation_mode="token_mean",
    **kwargs
):
    mode = str(token_aggregation_mode or "token_mean").strip().lower().replace("-", "_")
    if mode in {"token_mean", "mean"}:
        return evaluate_saved_attention_sink_first_token_mean(
            tokenizer=tokenizer,
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_prompt,
            return_aggregate=return_aggregate,
            patch_size=patch_size,
            merge_size=merge_size,
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=save_fig,
            with_tag=with_tag,
            layers_num=layers_num,
            heads_num=heads_num,
            vision_token_id=vision_token_id,
            **kwargs,
        )
    if mode in {"se_rank", "token_se_rank", "se_min", "token_se_min"}:
        return _evaluate_saved_attention_sink_first_token_se_rank(
            tokenizer=tokenizer,
            compressed_attn=compressed_attn,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_prompt,
            return_aggregate=return_aggregate,
            patch_size=patch_size,
            merge_size=merge_size,
            save_name=save_name,
            pred_has_anomaly=pred_has_anomaly,
            save_fig=save_fig,
            with_tag=with_tag,
            layers_num=layers_num,
            heads_num=heads_num,
            vision_token_id=vision_token_id,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported token_aggregation_mode: {token_aggregation_mode}. "
        "Use 'token_mean' or 'se_rank'."
    )


def optimized_get_attention_from_saved_per_layer_head_fast(*args, **kwargs):
    return evaluate_saved_attention_fast(*args, **kwargs)


def optimized_get_attention_from_saved_per_layer_head_fast_sink_first(*args, **kwargs):
    return evaluate_saved_attention_sink_first_token_aggregate(*args, **kwargs)


def optimized_get_saved_per_layer_head_attention(*args, **kwargs):
    return optimized_save_per_layer_head_attention(*args, **kwargs)
