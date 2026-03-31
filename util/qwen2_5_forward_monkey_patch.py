from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention,Qwen2_5_VLAttention , logger , Qwen2_5_VLRotaryEmbedding , apply_multimodal_rotary_pos_emb, repeat_kv , rotate_half , apply_rotary_pos_emb_vision
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig,Qwen2_5_VLConfig
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from typing import Tuple, Optional, Callable
def qwen2_5vl_vision_init(self, config: Qwen2_5_VLVisionConfig) -> None:
        super(Qwen2_5_VLVisionAttention , self).__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def qwen2_5vl_vision_eager_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    
    orig_dtype = query_states.dtype
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [
        torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
    ]

    attn_outputs = [
        attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]
        for q, k, v in zip(*splits)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = attn_output.to(orig_dtype)
    attn_output = self.proj(attn_output)
    return attn_output

def qwen2_5_vl_encoder_init(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
    super(Qwen2_5_VLAttention , self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
            "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
            "when creating this class."
        )

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.is_causal = True
    self.attention_dropout = config.attention_dropout
    self.rope_scaling = config.rope_scaling
    self.scaling = self.head_dim**-0.5

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )
    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

def qwen2_5_vl_encoder_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        orig_dtype = query_states.dtype
        query_states = query_states.to(torch.float32)
        key_states = key_states.to(torch.float32)
        value_states = value_states.to(torch.float32)
        
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states.to(orig_dtype), value_states.to(orig_dtype), self.layer_idx, cache_kwargs)
            key_states = key_states.to(torch.float32)
            value_states = value_states.to(torch.float32)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

def monkey_patch_qwen2_5vl_vision_eager_attn():
    Qwen2_5_VLVisionAttention.__init__ = qwen2_5vl_vision_init
    Qwen2_5_VLVisionAttention.forward = qwen2_5vl_vision_eager_forward

def monkey_patch_qwen2_5_vl_encoder_eager_attn():
    Qwen2_5_VLAttention.__init__ = qwen2_5_vl_encoder_init
    Qwen2_5_VLAttention.forward = qwen2_5_vl_encoder_forward