# PixachuModel – DINOv2 backbone + rotary embeddings based on 2-D L2 distances
#
# NB: the code purposefully re-uses most of the original DINOv2 building
#     blocks (MLP, LayerNorm, etc.) and only swaps the attention module.
#
#     Every patch has coordinates (row, col) in the patch grid.  The rotary
#     angle θ used for a given feature dimension is:
#         θ = r · 1 / 10000^(2i/d)
#     where r = sqrt(row² + col²).  CLS / special tokens simply receive r = 0
#     (→ no rotation).


import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Dinov2Model
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2DropPath,
    Dinov2MLP,
    Dinov2SelfOutput,
    Dinov2SwiGLUFFN,
)

from ._auto import AutoRegisterModelMixin
from ._config import PixachuConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    (…, 2i, 2i+1) -> (…, -x_{2i+1}, x_{2i})
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcasted rotary application.

    q, k       : (batch, seq_len, n_heads, head_dim)
    cos & sin  : (seq_len, 1, head_dim)
    """
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)

    return q_rot, k_rot


class PixachuSelfAttention(nn.Module):
    """
    A drop-in replacement for Dinov2SelfAttention that injects rotary position
    embeddings whose angle is a function of the L2 distance to the origin of
    the patch grid (rather than the 1-D token index).
    """

    def __init__(self, config: PixachuConfig):
        super().__init__()

        self.config = config
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads

        if self.head_dim * self.n_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.scale = self.head_dim**-0.5

        bias = getattr(config, "qkv_bias", True)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)

        # rotary cache
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._cached_meta: tuple[int, int, int] = (0, 0, 0)  # (seq_len, rows, cols)

    def _build_cos_sin(
        self,
        seq_len: int,
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-computes cos & sin tensors of shape (seq_len, 1, head_dim).

        Token 0 (CLS) is assigned r = 0.
        """
        # Check cache
        if (seq_len, rows, cols) == self._cached_meta and self.cos_cached.numel() != 0:
            return (
                self.cos_cached.to(device=device, dtype=dtype),
                self.sin_cached.to(device=device, dtype=dtype),
            )

        # Inverse frequency the standard way
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim)
        )  # (head_dim//2,)

        # Token-to-grid mapping
        ids = torch.arange(seq_len, device=device)
        cls_offset = 1 if seq_len != rows * cols else 0
        grid_ids = (ids - cls_offset).clamp(min=0)  # negative → CLS → r = 0

        row_ids = grid_ids // cols
        col_ids = grid_ids % cols
        r = (row_ids.float() ** 2 + col_ids.float() ** 2).sqrt()  # (seq_len,)

        # Outer product to obtain frequencies
        freqs = torch.outer(r, inv_freq)  # (seq_len, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

        cos = emb.cos()[:, None, :]  # (seq_len, 1, head_dim)
        sin = emb.sin()[:, None, :]

        # Cache
        self.cos_cached = cos.clone().detach()
        self.sin_cached = sin.clone().detach()
        self._cached_meta = (seq_len, rows, cols)

        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_dims: tuple[int, int],
        output_attentions: bool = False,
    ):
        """
        hidden_states : (batch, seq_len, hidden_dim)
        patch_dims    : (rows, cols) – number of patch rows / cols after patchify
        """
        bsz, seq_len, _ = hidden_states.size()
        rows, cols = patch_dims

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (batch, seq_len, n_heads, head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim)

        # Rotary
        cos, sin = self._build_cos_sin(seq_len, rows, cols, q.device, q.dtype)
        q, k = _apply_rotary(q, k, cos, sin)

        # Attention
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(attn_scores.dtype)
        attn_probs = self.attn_drop(attn_probs)

        context = torch.matmul(attn_probs, v)  # (batch, n_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.out_proj(context)

        if output_attentions:
            return output, attn_probs

        return output, None


class PixachuLayer(nn.Module):
    """
    Transformer block identical to the DINOv2 one but with PixachuSelfAttention.
    """

    def __init__(self, config: PixachuConfig):
        super().__init__()
        self.attention = PixachuSelfAttention(config)
        self.attention_output = Dinov2SelfOutput(config)

        if getattr(config, "use_swiglu_ffn", False):
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)

        self.layernorm_before = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        dpr = getattr(config, "drop_path_rate", 0.0)
        self.drop_path = Dinov2DropPath(dpr) if dpr > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_dims: tuple[int, int],
        output_attentions: bool = False,
    ):
        normed = self.layernorm_before(hidden_states)
        attn_out, attn_probs = self.attention(normed, patch_dims, output_attentions)
        attn_processed = self.attention_output(attn_out, hidden_states)
        hidden_states = hidden_states + self.drop_path(attn_processed)

        normed = self.layernorm_after(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = hidden_states + self.drop_path(mlp_out)

        if output_attentions:
            return hidden_states, attn_probs

        return hidden_states, None


class PixachuModel(AutoRegisterModelMixin, Dinov2Model):
    """
    A DINOv2 backbone where every transformer layer uses rotary 2-D
    self-attention (PixachuSelfAttention).  The rest of the architecture
    (patch embedding, MLP, layer_norm, …) remains untouched.
    """

    config_class = PixachuConfig

    def __init__(self, config: PixachuConfig):
        super().__init__(config)

        for i in range(len(self.encoder.layer)):
            self.encoder.layer[i] = PixachuLayer(config)

    def forward(
        self,
        pixel_values: torch.Tensor,  # (batch, C, H, W)
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape
        patch = self.config.patch_size

        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Image size ({height}×{width}) must be divisible by patch_size={patch}")

        rows, cols = height // patch, width // patch
        patch_dims = (rows, cols)

        # Patch embedding
        embedding_output = self.embeddings(pixel_values)  # (batch, seq_len, hidden)

        # Transformer encoder
        hidden_states = embedding_output
        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []

        for blk in self.encoder.layer:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_probs = blk(
                hidden_states,
                patch_dims=patch_dims,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions.append(attn_probs)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Final layer-norm (if present in DINOv2)
        if hasattr(self, "layernorm"):
            hidden_states = self.layernorm(hidden_states)
        elif hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        # Return
        if not return_dict:
            outputs = (hidden_states,)

            if output_hidden_states:
                outputs += (tuple(all_hidden_states),)

            if output_attentions:
                outputs += (tuple(all_attentions),)

            return outputs

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )
