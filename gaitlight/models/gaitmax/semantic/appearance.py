"""
Appearance module from BigGait (Ye et al.).

Distills multi-layer DINO features into per-token continuous appearance
descriptors via a linear projection + sigmoid, preserving texture/color
information complementary to the denoising (part) branch.

Reference config (BigGait):
    source_dim: 1536 (4 * 384), target_dim: 16, linear + sigmoid
"""

import einops
import torch
from torch import nn, Tensor


class AppearanceModule(nn.Module):
    """
    DINO multi-layer features -> concat -> Linear -> BN -> sigmoid -> appearance map.

    :param s_dim: per-layer DINO feature dimension (384 for ViT-S).
    :param k: number of DINO layers to concat (source_dim = s_dim * k).
    :param t_dim: target dimension (appearance channels).
    """

    def __init__(self, s_dim: int = 384, k: int = 4, t_dim: int = 16):
        super().__init__()

        src = s_dim * k

        self.bn_s = nn.BatchNorm1d(src, affine=False)
        self.linear = nn.Linear(src, t_dim)
        self.bn_t = nn.BatchNorm1d(t_dim, affine=False)

        self.t_dim = t_dim

    def forward(self, f_v: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """
        :param f_v:  DINO features [k, b, t, c, h, w]
        :param mask: foreground mask [b, t, h, w] bool
        :return:
            - feat: appearance map [b, t, t_dim, h, w] (sigmoid, each channel independent)
        """
        k, b, t, c, h, w = f_v.shape

        # concat k layers: [bthw, k*c]
        flat = einops.rearrange(f_v, 'k b t c h w -> (b t h w) (k c)')
        m = mask.reshape(-1)  # [bthw]

        # encode foreground tokens only
        fg = flat[m]  # [n_fg, k*c]
        fg = torch.sigmoid(self.bn_t(self.linear(self.bn_s(fg))))  # [n_fg, t_dim]

        # scatter back (match dtype for bf16-mixed training)
        out = torch.zeros(b * t * h * w, self.t_dim, dtype=fg.dtype, device=fg.device)
        out[m] = fg

        # [b, t, t_dim, h, w]
        feat = einops.rearrange(out, '(b t h w) c -> b t c h w', b=b, t=t, h=h, w=w)

        return {'feat': feat}
