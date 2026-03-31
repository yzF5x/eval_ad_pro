from typing import List, Optional, Dict

def find_tag_span_simple(token_list_decoded: List[str], tag: str) -> Optional[Dict]:
    """
    找到 <tag>...</tag> 的大致 token 区间（半开区间 [start_token, end_token)）。
    返回 dict: {
      'tag': tag,
      'start_token': int,   # 包含，内容开始的 token 索引
      'end_token': int,     # 不包含，内容结束的 token 索引
      'content': str,       # 从 tokens 拼回的内容（去首尾空白）
      'method': 'direct' or 'fallback_token_scan'
    }
    若未找到返回 None.
    说明: 采用 ''.join(tokens) 进行匹配 —— 对被拆开的标签最稳健。
    """
    if not token_list_decoded:
        return None

    # 1) 无空格直接拼接（保证与 token_starts 的计数一致）
    joined = ''.join(token_list_decoded)

    open_tag = f'<{tag}>'
    close_tag = f'</{tag}>'

    o_idx = joined.find(open_tag)
    c_idx = joined.find(close_tag)

    # 计算每个 token 在 joined 中的起始字符位置（基于 ''.join）
    token_starts = []
    pos = 0
    for t in token_list_decoded:
        token_starts.append(pos)
        pos += len(t)

    def char_to_token_idx(char_pos: int) -> int:
        # 找第一个 token 使得 token_start + len(token) > char_pos
        # 返回 len(token_list_decoded) 表示在末尾
        for i, s in enumerate(token_starts):
            if s + len(token_list_decoded[i]) > char_pos:
                return i
        return len(token_list_decoded)

    # 2) 如果在 joined 中都找到且顺序正确 -> 映回 token 索引并返回
    if o_idx != -1 and c_idx != -1 and o_idx + len(open_tag) <= c_idx:
        content_start_char = o_idx + len(open_tag)
        content_end_char = c_idx
        start_token = char_to_token_idx(content_start_char)
        end_token = char_to_token_idx(content_end_char)
        content = ''.join(token_list_decoded[start_token:end_token]).strip()
        return {
            'tag': tag,
            'start_token': max(0, start_token),
            'end_token': max(start_token, end_token),
            'content': content,
            'method': 'direct'
        }

    # 3) 退路：token 层面简单扫描（找包含开/闭标签片段的 token）
    low = [t.lower() for t in token_list_decoded]
    open_tok_idx = None
    close_tok_idx = None
    for i, t in enumerate(low):
        if open_tag in t or f'<{tag}' in t or ('<' in t and tag in t):
            open_tok_idx = i
            break
    for j in range(len(low)-1, -1, -1):
        t = low[j]
        if close_tag in t or f'/{tag}' in t or ('/' in t and tag in t):
            close_tok_idx = j
            break

    if open_tok_idx is not None:
        # 内容通常在 open_tok_idx 后面开始
        start = open_tok_idx + 1
        # 若能找到闭合 token 且在开之后用它；否则取一个合理窗口
        if close_tok_idx is not None and close_tok_idx > open_tok_idx:
            end = close_tok_idx
        else:
            end = min(len(token_list_decoded), start + 120)
        content = ''.join(token_list_decoded[start:end]).strip()
        return {
            'tag': tag,
            'start_token': start,
            'end_token': end,
            'content': content,
            'method': 'fallback_token_scan'
        }

    # 4) 全失败 -> 返回 None
    return None
