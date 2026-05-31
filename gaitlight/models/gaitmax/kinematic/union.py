"""
Union module for cross-pattern coordination modeling.

After per-pattern temporal modeling (DynamicModule) and temporal pooling,
each pattern holds a motion summary. This module lets patterns attend to
each other to capture inter-part coordination (e.g., arm-leg phase coupling).
"""

import torch.nn.functional as func
from torch import nn, Tensor

__all__ = ['UnionModule']


class UnionModule(nn.Module):
    """
    Cross-pattern self-attention over motion summaries.

    Input is [b, m, d] where each of the m tokens represents one body
    part's temporal motion summary.  Standard transformer blocks (no
    positional encoding — pattern identity is already in the token content
    from slot attention).

    :param d_model: token dimension.
    :param n_heads: number of attention heads.
    :param n_layers: number of transformer blocks.
    :param dropout: attention and FFN dropout rate.
    """

    def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 1,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: per-pattern motion summaries [b, m, d]
        :return: coordination-aware features [b, m, d]
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [b, m, d]
        :return: [b, m, d]
        """
        b, m, d = x.shape
        h = self.n_heads

        qkv = self.qkv(x).reshape(b, m, 3, h, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = func.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).reshape(b, m, d)
        return self.out_proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
