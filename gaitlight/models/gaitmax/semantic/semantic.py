"""
Semantic branch assembled from BigGait (Ye et al.).

Full pipeline: DINO features + mask → Denoising / Appearance encoding
→ Black DA → FusionBackbone → Temporal Pooling → Horizontal Pooling
→ PartLinear → PartHead → (embed, logits).

This module owns no DINO backbone or mask estimator; both are provided
externally (from DINOWrapper and HumanPrior respectively).
"""

import random
from typing import Any

from torch import nn, Tensor

from gaitlight.models.gaitmax.head import PartHead, PartLinear
from gaitlight.models.gaitmax.semantic.appearance import AppearanceModule
from gaitlight.models.gaitmax.semantic.denoising import DenoisingModule
from gaitlight.models.gaitmax.semantic.fusion import FusionBackbone


class SemanticBranch(nn.Module):
    """
    BigGait-style semantic gait recognition branch.

    :param s_dim: per-layer DINO feature dimension (384 for ViT-S).
    :param k: number of DINO layers concatenated.
    :param t_dim: distilled channel dimension for denoising / appearance.
    :param channels: FusionBackbone channel widths per ResNet stage.
    :param layers: number of BasicBlocks per stage.
    :param strides: spatial stride per stage.
    :param num_parts: horizontal pooling strip count.
    :param embed_dim: final embedding dimension per part.
    :param num_classes: classification head output classes.
    :param da_ratio: Black DA — fraction of batch samples to mask (training only).
    """

    def __init__(
            self,
            s_dim: int = 384,
            k: int = 4,
            t_dim: int = 16,
            channels: tuple[int, ...] = (64, 128, 256, 512),
            layers: tuple[int, ...] = (1, 1, 1, 1),
            strides: tuple[int, ...] = (1, 2, 2, 1),
            num_parts: int = 16,
            embed_dim: int = 256,
            num_classes: int = 100,
            da_ratio: float = 0.2,
    ):
        super().__init__()

        # encoding
        self.denoising = DenoisingModule(s_dim=s_dim, k=k, t_dim=t_dim)
        self.appearance = AppearanceModule(s_dim=s_dim, k=k, t_dim=t_dim)

        # fusion + backbone
        self.backbone = FusionBackbone(
            in_dim=t_dim, channels=channels, layers=layers, strides=strides,
        )

        # pooling
        self.num_parts = num_parts

        # head
        self.fc = PartLinear(num_parts, channels[-1], embed_dim)
        self.head = PartHead(num_parts, num_classes, embed_dim)

        # data augmentation
        self.da_ratio = da_ratio

    def forward(
            self, f_v: Tensor, mask: Tensor, quality: Tensor | None = None,
    ) -> dict[str, dict[str, Any] | Any]:
        """
        :param f_v:     DINO features [k, b, t, c, h, w]
        :param mask:    foreground mask [b, t, h, w] bool
        :param quality: per-frame quality scores [b, t] in (0, 1), or None
        :return:
            - embed: gait embedding [b, d, p]
            - logits: classification logits [b, num_class, p]
            - loss: dict of auxiliary losses {loss_con, loss_div}
        """
        # --- encode ---
        den_out = self.denoising(f_v, mask)
        app_out = self.appearance(f_v, mask)

        f_den = den_out['feat']  # [b, t, t_dim, h, w]
        f_app = app_out['feat']  # [b, t, t_dim, h, w]

        # --- Black DA: randomly zero one branch for ~20% of samples ---
        if self.training:
            f_den, f_app = self._black_da(f_den, f_app)

        # --- fusion backbone ---
        fused = self.backbone(f_den, f_app)  # [b, t, C, h', w']

        # --- quality-weighted temporal pooling ---
        pooled = self._temporal_pool(fused, quality)  # [b, C, h', w']

        # --- horizontal pooling (split h' into num_parts strips, mean+max) ---
        pooled = self._horizontal_pool(pooled)  # [b, C, p]

        # --- head ---
        embed = self.fc(pooled)  # [b, d, p]
        cls = self.head(embed)  # {'embed': [b, d, p], 'logits': [b, k, p]}

        return {
            'embed': embed,
            'logits': cls['logits'],
            'loss': {
                'loss_con': den_out['loss_con'],
                'loss_div': den_out['loss_div'],
            },
        }

    def _black_da(self, f_den: Tensor, f_app: Tensor) -> tuple[Tensor, Tensor]:
        """
        Randomly zero out one branch for a subset of batch samples,
        forcing the network to not rely on a single modality.
        """
        b = f_den.shape[0]
        n_mask = max(1, round(b * self.da_ratio))
        indices = random.sample(range(b), n_mask)

        for i in indices:
            if random.random() < 0.5:
                f_den[i] = 0
            else:
                f_app[i] = 0

        return f_den, f_app

    @staticmethod
    def _temporal_pool(x: Tensor, quality: Tensor | None) -> Tensor:
        """
        Quality-weighted temporal pooling for spatial feature maps.

        When quality is provided, computes weighted mean over time.
        Otherwise falls back to max pooling.

        :param x: [b, t, C, h, w]
        :param quality: [b, t] in (0, 1), or None
        :return: [b, C, h, w]
        """
        if quality is None:
            return x.max(dim=1).values

        # quality-gated max: scale features by quality, then max.
        # preserves peak-capturing property while suppressing low-quality frames.
        w = quality[:, :, None, None, None]  # [b, t, 1, 1, 1]
        return (x * w).max(dim=1).values

    def _horizontal_pool(self, x: Tensor) -> Tensor:
        """
        Split feature map into horizontal strips and pool (mean + max).

        :param x: [b, c, h, w]
        :return: [b, c, p]
        """
        b, c, h, w = x.shape
        x = x.reshape(b, c, self.num_parts, -1)  # [b, c, p, h'*w'/p]
        return x.mean(dim=-1) + x.max(dim=-1).values  # [b, c, p]
