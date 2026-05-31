"""
Kinematic branch: motion-based gait recognition.

Pipeline: DINO features + pattern attention
→ DynamicModule  (per-part temporal self-attention with 1D RoPE)
→ Quality-weighted temporal pool
→ UnionModule     (cross-part coordination attention)
→ PartLinear + PartHead  (per-pattern embedding and classification)

This module owns no DINO backbone or HumanPrior; both are provided
externally.
"""

import einops
from torch import nn, Tensor

from gaitlight.models.gaitmax.head import PartHead, PartLinear
from gaitlight.models.gaitmax.kinematic.dynamic import DynamicModule
from gaitlight.models.gaitmax.kinematic.union import UnionModule

__all__ = ['KinematicBranch']


class KinematicBranch(nn.Module):
    """
    Motion-based gait recognition from body-part trajectories.

    :param s_dim: DINO last-layer feature dimension.
    :param d_motion: internal motion token dimension.
    :param n_heads: attention heads (shared by dynamic and union).
    :param n_dyn_layers: transformer layers in DynamicModule.
    :param n_union_layers: transformer layers in UnionModule.
    :param num_parts: number of body-part patterns (m from HumanPrior).
    :param embed_dim: final embedding dimension per part.
    :param num_classes: classification head output classes.
    :param dropout: dropout rate.
    """

    def __init__(
            self,
            s_dim: int = 384,
            d_motion: int = 64,
            n_heads: int = 4,
            n_dyn_layers: int = 2,
            n_union_layers: int = 1,
            num_parts: int = 11,
            embed_dim: int = 256,
            num_classes: int = 100,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.dynamic = DynamicModule(
            s_dim=s_dim, d_motion=d_motion,
            n_heads=n_heads, n_layers=n_dyn_layers, dropout=dropout,
        )
        self.union = UnionModule(
            d_model=d_motion, n_heads=n_heads,
            n_layers=n_union_layers, dropout=dropout,
        )

        # head: per-pattern embedding
        self.fc = PartLinear(num_parts, d_motion, embed_dim)
        self.head = PartHead(num_parts, num_classes, embed_dim)

    def forward(
            self, f_last: Tensor, attn: Tensor, quality: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        :param f_last:  DINO last-layer features [b, t, c, h, w]
        :param attn:    pattern attention maps   [b, t, m, h, w]
        :param quality: per-frame quality scores [b, t] in (0, 1), or None
        :return:
            - embed: gait embedding [b, embed_dim, m]
            - logits: classification logits [b, num_classes, m]
        """
        # per-part temporal modeling (quality biases attention)
        x = self.dynamic(f_last, attn, quality)  # [b, t, m, d_motion]

        # quality-weighted temporal pooling
        x = _quality_pool(x, quality)  # [b, m, d_motion]

        # cross-part coordination
        x = self.union(x)  # [b, m, d_motion]

        # per-pattern head:  [b, m, d] → [b, d, m] for PartLinear/PartHead
        x = einops.rearrange(x, 'b m d -> b d m')

        embed = self.fc(x)  # [b, embed_dim, m]
        cls = self.head(embed)  # {'embed': [b, embed_dim, m], 'logits': [b, k, m]}

        return {
            'embed': embed,
            'logits': cls['logits'],
        }


def _quality_pool(x: Tensor, quality: Tensor | None) -> Tensor:
    """
    Quality-weighted temporal pooling.

    When quality is provided, computes weighted mean over time.
    Otherwise falls back to max pooling.

    :param x: [b, t, m, d]
    :param quality: [b, t] in (0, 1), or None
    :return: [b, m, d]
    """
    if quality is None:
        return x.max(dim=1).values

    # quality-gated max: scale features by quality, then max.
    # preserves peak-capturing property while suppressing low-quality frames.
    w = quality[:, :, None, None]  # [b, t, 1, 1]
    return (x * w).max(dim=1).values  # [b, m, d]
