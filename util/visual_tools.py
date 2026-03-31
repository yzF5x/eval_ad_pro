from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import seaborn as sns
import spacy

from bisect import bisect_left, bisect_right
from scipy.ndimage import label

import re
from collections import OrderedDict
import time

_STRUCTURE_3x3 = np.ones((3, 3), dtype=np.int32)


def detect_attn_spike_by_share(
    vlm_attn: torch.Tensor,
    spike_patch_idx: int,
    share_thr: float = 0.30,
    eps: float = 1e-12
):
    if vlm_attn.dim() == 3:
        x = vlm_attn.view(vlm_attn.shape[0], -1)
    else:
        x = vlm_attn
    x = x.detach().float()

    spike_vals = x[:, spike_patch_idx]
    shares = spike_vals / (x.sum(dim=1) + eps)

    flag = (shares >= share_thr)
    indices = flag.nonzero(as_tuple=True)[0]
    return indices, flag


def detect_single_extreme_values_in_vlm_attn(
    vlm_attn: torch.Tensor,
    ratio: float = 50.0,
    dominance_ratio: float = 5.0,
    eps: float = 1e-12,
    return_indices: bool = True
):
    if vlm_attn.dim() == 3:
        x = vlm_attn.view(vlm_attn.shape[0], -1)
    elif vlm_attn.dim() == 2:
        x = vlm_attn
    else:
        raise ValueError(f"Unsupported shape: {vlm_attn.shape}")

    x = x.detach().float()
    Ntok, P = x.shape

    max2_vals, _ = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    max1 = max2_vals[:, 0]
    max2 = max2_vals[:, 1] + eps

    medians = x.median(dim=1).values + eps

    ratio1 = max1 / medians
    ratio2 = max1 / max2

    mask = (ratio1 > ratio) & (ratio2 > dominance_ratio)
    outlier_idx = mask.nonzero(as_tuple=True)[0]
    if outlier_idx.numel() == 0:
        return None, outlier_idx
    spike_pos = x[outlier_idx].argmax(dim=1)
    counts = torch.bincount(spike_pos, minlength=P)
    spike_patch_idx = int(counts.argmax().item())

    return spike_patch_idx, outlier_idx


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
    keep_pos={'NOUN','ADJ'},
    remove_pos={'PUNCT','DET','CCONJ','ADV','X','SPACE','ADP'},
    explicit_keep_words={},
    explicit_remove_words={'think','answer','addCriterion','begin_of_box','end_of_box'}
):
    nlp = spacy.load(lang_model)
    doc = nlp(text)
    spa_tokens = [(token.text, token.pos_) for token in doc]

    if keep_pos:
        selected_words = {token.text for token in doc if token.pos_ in keep_pos}
    elif remove_pos:
        selected_words = {token.text for token in doc if token.pos_ not in remove_pos}
    else:
        selected_words = {token.text for token in doc}

    tokens = tokenizer.tokenize(text)

    keep_indices = []
    keep_tokens = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.lstrip("Ġ▁")
        if clean_tok in explicit_keep_words:
            keep_indices.append(i)
            keep_tokens.append(clean_tok)
        else:
            if clean_tok in selected_words:
                if clean_tok in explicit_remove_words:
                    continue
                keep_indices.append(i)
                keep_tokens.append(clean_tok)

    return keep_indices, keep_tokens


def normalize_heatmap(vlm_attn_weights, grid_height, height, width, gamma_factor=1, grid_width=15):
    vlm_attn_image = vlm_attn_weights.reshape((grid_height, grid_width))
    vlm_attn_image = vlm_attn_image.to(torch.float32)
    vlm_attn_image = F.interpolate(
        vlm_attn_image.unsqueeze(0).unsqueeze(0), 
        size=(height, width), 
        mode='bicubic'
    ).squeeze()
    attn_over_image_np = np.power(vlm_attn_image.numpy(), 1 / gamma_factor)
    return attn_over_image_np


def custom_weighted_sum(filtered_vlm_attn: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    weights = index.view(-1, 1).float()
    num_to_count = weights.sum()
    weighted_attn = filtered_vlm_attn * weights
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


def visual_attn_token2image(keep_tokens, filtered_vlm_attn, save_name, grid_height, grid_width, height, width, image, summed=None, se_info=None, threshold=None, threshold_se=None, par_info=None, weight_info=None):
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
        attn_weights_over_vis_tokens = filtered_vlm_attn[idx]
        attn_over_image = normalize_heatmap(attn_weights_over_vis_tokens, grid_height, height, width, gamma_factor=1, grid_width=grid_width)
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
                         return_aggreagate=False,
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
    
    vlm_attn = llm_attn_matrix[:, :, output_token_start:output_token_end, vision_token_start:vision_token_end]
    vlm_attn = vlm_attn.flatten(start_dim=0, end_dim=2)
    vlm_attn = row_normalize(vlm_attn)
    device = vlm_attn.device
    Ntok = vlm_attn.shape[0]
    
    valid_all = torch.ones(vlm_attn.shape[0], dtype=torch.bool)
    attn_over_image_np1 = normalize_heatmap(custom_weighted_sum(vlm_attn, valid_all), grid_height, height, width, gamma_factor=1, grid_width=grid_width)
    
    prompt2text_attn_all = llm_attn_matrix[:, :, output_token_start:output_token_end, vision_token_end:output_token_start]
    prompt2text_attn_all = prompt2text_attn_all.flatten(start_dim=0, end_dim=2)
    filtered_prompt2output_text = prompt2text_attn_all[:, keep_indices_i]
    to_change = '.' + save_name.split('.')[-1]
    
    try:
        row_idx, col_idx = torch.meshgrid(torch.tensor(keep_indices), torch.tensor(keep_indices_i), indexing='ij')
        selected_prompt2text = prompt2text_attn_all[row_idx, col_idx]
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

    par_info_all = get_par_from_attention_fast(vlm_attn, 0.17, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5

    spike_patch_idx, outlier_idx = detect_single_extreme_values_in_vlm_attn(vlm_attn, ratio=50.0, dominance_ratio=5.0)
    outlier_flag = torch.zeros(Ntok, device=vlm_attn.device, dtype=torch.bool)
    outlier_flag[outlier_idx] = True
    outlier_tokens_num = sum(outlier_flag)
    all_tokens_num = outlier_flag.shape[0]
    bad_idx, bad_flag = detect_attn_spike_by_share(vlm_attn, spike_patch_idx, 0.3)

    candidate_se_compute = valid_filtered_token & valid_par_index_all & (~outlier_flag) & (~bad_flag)
    cand_idx = candidate_se_compute.nonzero(as_tuple=True)[0]
    candidate_all = valid_filtered_token & valid_par_index_all & valid_sum_index_all & (~outlier_flag) & (~bad_flag)
    cand_all_idx = candidate_all.nonzero(as_tuple=True)[0]
    
    if cand_all_idx.numel() <= 3:
        return normalize_heatmap(custom_weighted_sum(vlm_attn, valid_filtered_token & valid_sum_index_all), grid_height, height, width, gamma_factor=1, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
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
        vlm_attn[cand_idx],
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
        return normalize_heatmap(custom_weighted_sum(vlm_attn, candidate_se_compute), grid_height, height, width, gamma_factor=1, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(vlm_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, gamma_factor=1, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    
    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(se_info_all, summed_all, final_valid_index_reasoning)
    
    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=False)
    topk_final_index = final_index[topk_index]
    final_keep_tokens = [token_list_decoded[i] for i in topk_final_index.tolist()]
    topk_final_valid_index_reasoning = torch.zeros_like(final_valid_index_reasoning)  
    topk_final_valid_index_reasoning[topk_final_index] = True
    
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
    
    SC_new = compute_spatial_consistency_fast(vlm_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
    aggreated_final_image = normalize_heatmap(aggregate_cross_attentions(vlm_attn[topk_final_valid_index_reasoning][:], topk_final_token_weights), grid_height, height, width, gamma_factor=1, grid_width=grid_width)
    
    if return_aggreagate:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:], save_name.replace('.png','_final_aggreated_attention_fast.png'), grid_height, grid_width, height, width, image, summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning], threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning], topk_final_token_weights)
            save_path = save_name.replace(f'{to_change}', f'_final_aggreated_image_fast.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n {output_text}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:], save_name.replace('.png','_final_filtered_attention_fast.png'), grid_height, grid_width, height, width, image, summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning], threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning])
            save_path = save_name.replace(f'{to_change}', f'_final_valid_image_fast.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)

    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num


def optimized_get_saved_per_layer_head_attention(
    tokenizer,
    output_ids,
    input_token_len,
    processed_image,
    processed_prompt,
    patch_size=14,
    merge_size=2,
    **kwargs
):
    sequences = kwargs.pop("sequences", None)
    vision_token_id = kwargs.pop("vision_token_id", 151655)
    outlier_ratio = kwargs.pop("outlier_ratio", 50.0)
    dominance_ratio = kwargs.pop("dominance_ratio", 5.0)
    outlier_share_thr = kwargs.pop("outlier_share_thr", 0.3)

    if sequences is None:
        sequences = output_ids.get("sequences", None)
    if sequences is None:
        raise ValueError("optimized_get_saved_per_layer_head_attention requires sequences or output_ids['sequences'].")

    prompt_attentions = output_ids["attentions"][0]
    num_layers = len(prompt_attentions)
    num_heads = prompt_attentions[0].size(1)
    prompt_len = prompt_attentions[0].size(-1)
    assert prompt_len == input_token_len, f"Expected prompt_len={input_token_len}, got {prompt_len}"

    image = processed_image[-1]
    width, height = image.size
    grid_width = int(width / (patch_size * merge_size))
    grid_height = int(height / (patch_size * merge_size))
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

    vlm_attn = attn_to_vision.flatten(start_dim=0, end_dim=2)
    vlm_attn = row_normalize(vlm_attn)
    prompt2text_attn_all = attn_to_text.flatten(start_dim=0, end_dim=2)

    spike_patch_idx, outlier_idx = detect_single_extreme_values_in_vlm_attn(
        vlm_attn, ratio=outlier_ratio, dominance_ratio=dominance_ratio
    )
    outlier_flag = torch.zeros(vlm_attn.shape[0], device=vlm_attn.device, dtype=torch.bool)
    if outlier_idx is not None and outlier_idx.numel() > 0:
        outlier_flag[outlier_idx] = True
    outlier_tokens_num = int(outlier_flag.sum().item())
    all_tokens_num = int(outlier_flag.shape[0])

    if spike_patch_idx is None:
        bad_flag = torch.zeros_like(outlier_flag)
    else:
        _, bad_flag = detect_attn_spike_by_share(vlm_attn, spike_patch_idx, outlier_share_thr)

    keep_mask = (~outlier_flag) & (~bad_flag)
    if not keep_mask.any():
        keep_mask = torch.ones_like(keep_mask)
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]

    filtered_vlm_attn = vlm_attn[keep_mask]
    filtered_prompt2text = prompt2text_attn_all[keep_mask]

    compressed_attn = {
        "filtered_vlm_attn": filtered_vlm_attn,
        "filtered_prompt2text_attn": filtered_prompt2text,
        "kept_indices": kept_indices,
        "vlm_attn_normalized": True,
        "outlier_tokens_num": outlier_tokens_num,
        "all_tokens_num": all_tokens_num
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
        "vision_token_id": vision_token_id
    }

    return compressed_attn, meta


def optimized_get_attention_from_saved_per_layer_head_fast( 
                         tokenizer, 
                         compressed_attn,
                         sequences,
                         input_token_len,
                         output_token_len,
                         processed_image,
                         processed_prompt,
                         return_aggreagate=False,
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
    
    vlm_attn = compressed_attn["vlm_attn"]
    prompt2text_attn_all = compressed_attn["prompt2text_attn"]
    kept_indices = compressed_attn["kept_indices"]
    if not torch.is_tensor(kept_indices):
        kept_indices = torch.tensor(kept_indices)
    kept_indices = kept_indices.to(torch.long)
    if not compressed_attn.get("vlm_attn_normalized", False):
        vlm_attn = row_normalize(vlm_attn)
    device = vlm_attn.device
    Ntok = vlm_attn.shape[0]
    
    outlier_tokens_num = int(compressed_attn.get("outlier_tokens_num", 0))
    all_tokens_num = int(compressed_attn.get("all_tokens_num", Ntok))
    
    valid_all = torch.ones(vlm_attn.shape[0], dtype=torch.bool)
    attn_over_image_np1 = normalize_heatmap(custom_weighted_sum(vlm_attn, valid_all), grid_height, height, width, gamma_factor=1, grid_width=grid_width)
    to_change = '.' + save_name.split('.')[-1]
    
    if len(keep_indices_i) == 0 or prompt2text_attn_all.shape[1] == 0:
        if save_fig:
            save_path = save_name.replace(f'{to_change}','_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num

    filtered_prompt2output_text = prompt2text_attn_all[:, keep_indices_i]
    valid_filtered_token = torch.isin(kept_indices, keep_indices.to(kept_indices.device))
    
    try:
        selected_row_idx = valid_filtered_token.nonzero(as_tuple=True)[0]
        if selected_row_idx.numel() == 0:
            raise ValueError("No kept rows matched output token filter.")
        selected_prompt2text = filtered_prompt2output_text[selected_row_idx][:, keep_indices_i]
    except:
        if save_fig:
            save_path = save_name.replace(f'{to_change}','_global_attention.png')
            heatmap_visual(attn_over_image_np1, image, title='original_global_attention', save_name=save_path)
        return attn_over_image_np1, 1.0, outlier_tokens_num, all_tokens_num
    
    if with_tag:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 0, 1)
    else:
        index_all, threshold_all, summed_all, summed_weights = get_threshold_and_weight_from_sum(filtered_prompt2output_text, 5, 6)
    
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
        return normalize_heatmap(custom_weighted_sum(vlm_attn, valid_filtered_token & valid_sum_index_all), grid_height, height, width, gamma_factor=1, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
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
        vlm_attn[cand_idx],
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
    final_valid_index_reasoning = valid_filtered_token & valid_sum_index_all & valid_se_isfinite & valid_se_index_all & valid_par_index_all

    if final_valid_index_reasoning.sum().item() < 3:
        return normalize_heatmap(custom_weighted_sum(vlm_attn, candidate_se_compute), grid_height, height, width, gamma_factor=1, grid_width=grid_width), 1.0, outlier_tokens_num, all_tokens_num
    
    final_valid_image_reasoning = normalize_heatmap(
        custom_weighted_sum(vlm_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height, height, width, gamma_factor=1, grid_width=grid_width
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    
    full_weights, valid_indices, final_token_weights, sorted_valid_indices = get_weight_with_indices(se_info_all, summed_all, final_valid_index_reasoning)
    
    topk = min(10, final_token_weights.numel())
    topk_final_token_weights, topk_index = torch.topk(final_token_weights, topk, largest=True, sorted=False)
    topk_final_index = final_index[topk_index]
    orig_indices_for_topk = kept_indices[topk_final_index].tolist()
    final_keep_tokens = [token_list_decoded[i] for i in orig_indices_for_topk]
    topk_final_valid_index_reasoning = torch.zeros_like(final_valid_index_reasoning)  
    topk_final_valid_index_reasoning[topk_final_index] = True
    
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
    
    SC_new = compute_spatial_consistency_fast(vlm_attn[valid_sc][:], grid_height, grid_width, top_k_percent=10)
    aggreated_final_image = normalize_heatmap(aggregate_cross_attentions(vlm_attn[topk_final_valid_index_reasoning][:], topk_final_token_weights), grid_height, height, width, gamma_factor=1, grid_width=grid_width)
    
    if return_aggreagate:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:], save_name.replace('.png','_final_aggreated_attention_fast.png'), grid_height, grid_width, height, width, image, summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning], threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning], topk_final_token_weights)
            save_path = save_name.replace(f'{to_change}', f'_final_aggreated_image_fast.png')
            heatmap_visual(aggreated_final_image, image, title=f'SC: {SC_new:.2f}\n {output_text}', save_name=save_path)
        return aggreated_final_image, SC_new, outlier_tokens_num, all_tokens_num
    else:
        if save_fig:
            visual_attn_token2image(final_keep_tokens, vlm_attn[topk_final_valid_index_reasoning][:], save_name.replace('.png','_final_filtered_attention_fast.png'), grid_height, grid_width, height, width, image, summed_all[topk_final_valid_index_reasoning], se_info_all[topk_final_valid_index_reasoning], threshold, threshold_se, par_info_all[topk_final_valid_index_reasoning])
            save_path = save_name.replace(f'{to_change}', f'_final_valid_image_fast.png')
            heatmap_visual(final_valid_image_reasoning, image, title=f'SC: {SC_new:.2f}', save_name=save_path)

    return final_valid_image_reasoning, SC_new, outlier_tokens_num, all_tokens_num
