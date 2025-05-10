# PixachuModel – DINOv2 backbone + rotary embeddings based on 2D coordinates
#
# NB: the code purposefully reuses most of the original DINOv2 building
#     blocks (MLP, LayerNorm, etc.) and only swaps the attention module.
#
#     Every patch has coordinates (row, col) in the patch grid. The rotary
#     angles θ and ψ used for a given feature dimension are:
#         θ = r_row · 1 / 10000^(2i/d)
#         ψ = r_col · 1 / 10000^(2i/d)
#     where r_row = sqrt(row²) and r_col = sqrt(col²). CLS / special tokens
#     simply receive r = 0 (→ no rotation).

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


def _apply_rotary_3d(
    x: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    cos_psi: torch.Tensor,
    sin_psi: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the 3-D rotation to (batch, seq_len, n_heads, head_dim) tensor.
    """
    b, s, h, d = x.shape
    t = d // 3  # number of triplets
    x = x.view(b, s, h, t, 3)  # -> (..., 3)

    # broadcast trig tensors to (1, seq_len, 1, t)
    ct = cos_theta[None, :, None, :]  # cos θ
    st = sin_theta[None, :, None, :]  # sin θ
    cp = cos_psi[None, :, None, :]  # cos ψ
    sp = sin_psi[None, :, None, :]  # sin ψ

    # components
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]

    # rotation (see matrix in doc-string)
    r0 = (cp * x0) + (sp * st * x1) + (sp * ct * x2)
    r1 = (ct * x1) - (st * x2)
    r2 = (-sp * x0) + (cp * st * x1) + (cp * ct * x2)

    x_rot = torch.stack((r0, r1, r2), dim=-1)  # (..., 3)
    return x_rot.view(b, s, h, d)


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

        if self.head_dim % 3 != 0:
            raise ValueError(
                f"For 3-D rotary embeddings the head dimension must be divisible by 3 (got head_dim = {self.head_dim})."
            )

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
        self.register_buffer("cos_theta_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_theta_cached", torch.empty(0), persistent=False)
        self.register_buffer("cos_psi_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_psi_cached", torch.empty(0), persistent=False)
        self._cached_meta: tuple[int, int, int] = (0, 0, 0)  # (seq_len, rows, cols)

    def _build_rotary_params(
        self,
        seq_len: int,
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pre-compute the four trigonometric tensors needed for the 3-D rotation.
        Shapes: (seq_len, n_triplets)
        """
        if (seq_len, rows, cols) == self._cached_meta and self.cos_theta_cached.numel() != 0:
            # return cached version on the proper device / dtype
            return (
                self.cos_theta_cached.to(device=device, dtype=dtype),
                self.sin_theta_cached.to(device=device, dtype=dtype),
                self.cos_psi_cached.to(device=device, dtype=dtype),
                self.sin_psi_cached.to(device=device, dtype=dtype),
            )

        n_triplets = self.head_dim // 3
        # Standard RoPE inverse frequency but for triplets (step = 3)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 3, device=device, dtype=dtype) / (self.head_dim * scale))
        )  # (n_triplets,)

        # Token-to-grid mapping
        ids = torch.arange(seq_len, device=device)
        cls_offset = 1 if seq_len != rows * cols else 0
        grid_ids = (ids - cls_offset).clamp(min=0)

        row_ids = grid_ids // cols  # θ  ← row
        col_ids = grid_ids % cols  # ψ  ← col

        theta = row_ids.float()[:, None] * inv_freq  # (seq_len, n_triplets)
        psi = col_ids.float()[:, None] * inv_freq

        cos_theta, sin_theta = theta.cos(), theta.sin()
        cos_psi, sin_psi = psi.cos(), psi.sin()

        # cache them
        self.cos_theta_cached = cos_theta.clone().detach()
        self.sin_theta_cached = sin_theta.clone().detach()
        self.cos_psi_cached = cos_psi.clone().detach()
        self.sin_psi_cached = sin_psi.clone().detach()
        self._cached_meta = (seq_len, rows, cols)

        return cos_theta, sin_theta, cos_psi, sin_psi

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

        # Perform smaller rotations for smaller patches due to increased sequence length
        scale = self.config.character_pixel_size // self.config.patch_size

        # Rotary parameters
        (cos_theta, sin_theta, cos_psi, sin_psi) = self._build_rotary_params(
            seq_len, rows, cols, q.device, q.dtype, scale=scale
        )

        # Apply 3-D rotation to q & k
        q = _apply_rotary_3d(q, cos_theta, sin_theta, cos_psi, sin_psi)
        k = _apply_rotary_3d(k, cos_theta, sin_theta, cos_psi, sin_psi)

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

    @staticmethod
    def _apply_rotary_3d(
        x: torch.Tensor,
        cos_theta: torch.Tensor,
        sin_theta: torch.Tensor,
        cos_psi: torch.Tensor,
        sin_psi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the 3-D rotation to (batch, seq_len, n_heads, head_dim) tensor.
        """
        b, s, h, d = x.shape
        t = d // 3  # number of triplets
        x = x.view(b, s, h, t, 3)  # -> (..., 3)

        # broadcast trig tensors to (1, seq_len, 1, t)
        ct = cos_theta[None, :, None, :]  # cos θ
        st = sin_theta[None, :, None, :]  # sin θ
        cp = cos_psi[None, :, None, :]  # cos ψ
        sp = sin_psi[None, :, None, :]  # sin ψ

        # components
        x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]

        # rotation (see matrix in doc-string)
        r0 = (cp * x0) + (sp * st * x1) + (sp * ct * x2)
        r1 = (ct * x1) - (st * x2)
        r2 = (-sp * x0) + (cp * st * x1) + (cp * ct * x2)

        x_rot = torch.stack((r0, r1, r2), dim=-1)  # (..., 3)
        return x_rot.view(b, s, h, d)
