"""
Dynamic module for part-wise temporal motion modeling.

For each body-part pattern discovered by HumanPrior, pools a per-frame
token from DINO features, concatenates the part's spatial center (cx, cy)
as input features, then applies temporal self-attention with 1D RoPE on
frame index only.

Spatial position is part of the token content (the model freely learns how
to use it), while temporal ordering is encoded via RoPE (providing sequence
direction and locality bias).

Accepts an optional quality score per frame to bias attention away from
low-quality frames.
"""

import einops
import torch
import torch.nn.functional as func
from torch import nn, Tensor

__all__ = ['DynamicModule']


class DynamicModule(nn.Module):
    """
    Part-wise temporal self-attention with 1D temporal RoPE.

    Spatial center coordinates are concatenated to the token and projected,
    not encoded via RoPE, so the model can attend freely across all spatial
    positions without distance bias.

    :param s_dim: DINO last-layer feature dimension (384 for ViT-S).
    :param d_motion: compressed token dimension for motion modeling.
    :param n_heads: number of attention heads.
    :param n_layers: number of transformer blocks.
    :param dropout: attention and FFN dropout rate.
    """

    def __init__(
            self,
            s_dim: int = 384,
            d_motion: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.0,
    ):
        super().__init__()

        # DINO features (s_dim) + spatial center (2) → d_motion
        self.proj = nn.Linear(s_dim + 2, d_motion)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_motion, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_motion)

    def forward(self, f_last: Tensor, attn: Tensor, quality: Tensor | None = None) -> Tensor:
        """
        :param f_last:  DINO last-layer features [b, t, c, h, w]
        :param attn:    pattern attention maps   [b, t, m, h, w]  (softmax over m)
        :param quality: per-frame quality scores [b, t] in (0, 1), or None
        :return: motion features [b, t, m, d_motion]
        """
        b, t, m, h, w = attn.shape

        # --- pool per-pattern tokens via attention-weighted average ---
        attn_norm = attn / attn.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        token = torch.einsum('btmhw, btchw -> btmc', attn_norm, f_last)  # [b, t, m, s_dim]

        # --- compute center coordinates from attention maps ---
        mu = _fit_center(attn, h, w)  # [b, t, m, 2]  normalized to [0, 1]

        # --- concat spatial position to token, then compress ---
        token = self.proj(torch.cat([token, mu], dim=-1))  # [b, t, m, d_motion]

        # --- per-pattern temporal self-attention ---
        # reshape: treat each pattern independently  [b*m, t, d]
        x = einops.rearrange(token, 'b t m d -> (b m) t d')

        # frame indices for 1D temporal RoPE
        t_idx = torch.arange(t, device=x.device, dtype=x.dtype)  # [t]

        # quality attention mask: [b*m, t] (repeat per pattern)
        q_mask = None
        if quality is not None:
            q_mask = einops.repeat(quality, 'b t -> (b m) t', m=m)

        for block in self.blocks:
            x = block(x, t_idx, q_mask)

        x = self.norm(x)
        return einops.rearrange(x, '(b m) t d -> b t m d', b=b, m=m)


# ---------------------------------------------------------------------------
# Internal components
# ---------------------------------------------------------------------------


def _fit_center(attn: Tensor, h: int, w: int) -> Tensor:
    """
    Compute attention-weighted centroid for each pattern, normalized to [0, 1].

    :param attn: [b, t, m, h, w]
    :return: [b, t, m, 2]  (cx, cy) in [0, 1]
    """
    device = attn.device
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij',
    )
    coords = torch.stack([gx, gy], dim=-1)  # [h, w, 2]

    alpha = attn / attn.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    mu = torch.einsum('btmhw, hwc -> btmc', alpha, coords)  # [b, t, m, 2]

    # normalize to [0, 1]
    mu[..., 0] = mu[..., 0] / max(w - 1, 1)
    mu[..., 1] = mu[..., 1] / max(h - 1, 1)
    return mu


class _TemporalRoPE(nn.Module):
    """
    1D rotary position encoding on frame index only.

    :param head_dim: dimension per attention head (must be even).
    :param theta: base frequency.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f'head_dim must be even, got {head_dim}'
        self.register_buffer('inv_freq', _inv_freq(head_dim, theta))

    def forward(self, x: Tensor, t_idx: Tensor) -> Tensor:
        """
        :param x:     [bs, n_heads, t, head_dim]
        :param t_idx: [t]  frame indices
        :return: rotated x, same shape
        """
        # angles: [t, head_dim/2]
        ang = torch.outer(t_idx, self.inv_freq)

        # broadcast: [1, 1, t, head_dim/2]
        cos = ang.cos().unsqueeze(0).unsqueeze(0)
        sin = ang.sin().unsqueeze(0).unsqueeze(0)

        x1 = x[..., 0::2]  # [bs, n_heads, t, head_dim/2]
        x2 = x[..., 1::2]

        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


def _inv_freq(dim: int, theta: float) -> Tensor:
    """Inverse frequency bands for RoPE. Returns [dim/2]."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


class _TemporalAttention(nn.Module):
    """Multi-head self-attention with 1D temporal RoPE and F.scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        self.rope = _TemporalRoPE(self.head_dim)

    def forward(self, x: Tensor, t_idx: Tensor, quality: Tensor | None = None) -> Tensor:
        """
        :param x:       [bs, t, d]
        :param t_idx:   [t]  frame indices
        :param quality: [bs, t] quality scores in (0, 1), or None
        :return: [bs, t, d]
        """
        bs, t, d = x.shape
        h = self.n_heads

        qkv = self.qkv(x).reshape(bs, t, 3, h, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [bs, t, h, head_dim]

        # transpose to [bs, h, t, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply 1D temporal RoPE to Q and K
        q = self.rope(q, t_idx)
        k = self.rope(k, t_idx)

        # quality-based attention bias: low quality → negative bias
        attn_mask = None
        if quality is not None:
            # [bs, t] → additive bias [bs, 1, 1, t] (broadcast over heads and query positions)
            attn_mask = quality.clamp(min=1e-8).log().unsqueeze(1).unsqueeze(2)

        # scaled dot-product attention (uses Flash Attention when available)
        out = func.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [bs, h, t, head_dim]

        out = out.transpose(1, 2).reshape(bs, t, d)  # [bs, t, d]
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attention → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _TemporalAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, t_idx: Tensor, quality: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), t_idx, quality)
        x = x + self.ffn(self.norm2(x))
        return x
