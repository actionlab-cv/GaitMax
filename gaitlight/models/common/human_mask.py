import logging

import einops
import torch
from kornia import morphology
from torch import nn

logger = logging.getLogger(__name__)


class HumanMask(nn.Module):
    """
    Human mask estimation module.

    :param s_dim: DINO feature dimension.
    :param k: number of DINO intermediate layers fused.
    """

    def __init__(self, s_dim: int = 384, k: int = 1):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm2d(s_dim * k),
            nn.Conv2d(s_dim * k, s_dim, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(s_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(s_dim, 2, bias=False)

    @staticmethod
    @torch.no_grad()
    def _enhance_edge(x: torch.Tensor) -> torch.Tensor:
        """
        Morphological edge enhancement: interior is set to 1, edge region retains soft probability.

        :param x: foreground probability map [bt, 1, h, w]
        :return: enhanced mask [bt, 1, h, w]
        """
        e = torch.round(x)

        m_dil = morphology.dilation(e, torch.ones(3, 3, device=x.device))
        m_ero = morphology.erosion(m_dil, torch.ones(5, 5, device=x.device))

        edge = (m_dil > 0.5) ^ (m_ero > 0.5)
        return edge * x + (m_ero > 0.5).float()

    def forward(self, f_v: torch.Tensor) -> dict:
        """
        :param f_v: DINO features [k, b, t, c, h, w]
        :return:
            - mask: enhanced foreground mask [b, t, 1, h, w]
            - seg: raw segmentation output [b, t, 2, h, w]
        """
        k, b, t, c, h, w = f_v.shape

        # [k, b, t, c, h, w] -> [bt, ck, h, w]
        f_s = einops.rearrange(f_v, 'k b t c h w -> (b t) (c k) h w')

        # [bt, ck, h, w] -> [bt, s_dim, h, w] -> [bt, 2, h, w]
        f_s = self.enc(f_s)
        seg = self.head(f_s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).softmax(dim=1)

        # enhanced foreground mask
        # [bt, 1, h, w] -> [b, t, 1, h, w]
        mask = self._enhance_edge(seg.detach()[:, 1:2])
        mask = einops.rearrange(mask, '(b t) 1 h w -> b t 1 h w', t=t)

        return {
            'mask': mask,
            'seg': einops.rearrange(seg, '(b t) c h w -> b t c h w', t=t),
        }
