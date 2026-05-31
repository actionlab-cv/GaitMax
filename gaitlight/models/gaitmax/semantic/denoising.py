"""
Denoising module from BigGait (Ye et al.).

Distills multi-layer DINO features into per-token soft part assignments
via softmax, producing a spatial body-part segmentation map.

Reference config (BigGait):
    source_dim: 1536 (4 * 384), target_dim: 16, softmax + GELU MLP
"""

import einops
import kornia.filters
import torch
import torch.nn.functional as func
from torch import nn, Tensor


class DenoisingModule(nn.Module):
    """
    DINO multi-layer features -> concat -> MLP -> softmax -> part map.

    :param s_dim: per-layer DINO feature dimension (384 for ViT-S).
    :param k: number of DINO layers to concat (source_dim = s_dim * k).
    :param t_dim: target dimension (number of body-part channels).
    """

    def __init__(self, s_dim: int = 384, k: int = 4, t_dim: int = 16):
        super().__init__()

        src = s_dim * k

        self.bn = nn.BatchNorm1d(src, affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(src, src // 2),
            nn.BatchNorm1d(src // 2, affine=False),
            nn.GELU(),
            nn.Linear(src // 2, t_dim),
        )

        self.t_dim = t_dim

    def forward(self, f_v: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """
        :param f_v:  DINO features [k, b, t, c, h, w]
        :param mask: foreground mask [b, t, h, w] bool
        :return:
            - feat: part assignment map [b, t, t_dim, h, w] (softmax over t_dim)
            - loss_con: connectivity loss (scalar)
            - loss_div: diversity loss (scalar)
        """
        k, b, t, c, h, w = f_v.shape

        # concat k layers: [b, t, h, w, k*c]
        x = einops.rearrange(f_v, 'k b t c h w -> b t h w (k c)')

        # flatten to token list, apply mask
        flat = einops.rearrange(x, 'b t h w c -> (b t h w) c')
        m = mask.reshape(-1)  # [bthw]

        # encode foreground tokens only
        fg = flat[m]  # [n_fg, k*c]
        fg = func.softmax(self.mlp(self.bn(fg)), dim=-1)  # [n_fg, t_dim]

        # scatter back (match dtype for bf16-mixed training)
        out = torch.zeros(b * t * h * w, self.t_dim, dtype=fg.dtype, device=fg.device)
        out[m] = fg

        # [b, t, t_dim, h, w]
        feat = einops.rearrange(out, '(b t h w) c -> b t c h w', b=b, t=t, h=h, w=w)

        # --- losses ---
        # connectivity: Sobel gradient on part maps (exclude last channel as BigGait does)
        feat_4d = einops.rearrange(feat[:, :, :-1], 'b t c h w -> (b t) c h w')
        loss_con = _connectivity_loss(feat_4d)

        # diversity: encourage uniform area distribution across parts
        feat_3d = einops.rearrange(feat, 'b t c h w -> (b t) (h w) c')
        loss_div = _diversity_loss(feat_3d, self.t_dim)

        return {'feat': feat, 'loss_con': loss_con, 'loss_div': loss_div}


def _connectivity_loss(x: Tensor) -> Tensor:
    """
    Penalise spatial discontinuity via Sobel gradients.

    :param x: [bt, c, h, w]
    """
    grad = kornia.filters.spatial_gradient(x, mode='sobel')  # [bt, c, 2, h, w]
    return torch.abs(grad).sum(dim=2).mean()


def _diversity_loss(x: Tensor, n_parts: int) -> Tensor:
    """
    Encourage each part to cover roughly equal area (max-entropy target).

    :param x: [bt, hw, c]  (softmax values)
    :param n_parts: number of parts
    """
    # per-part area proportion
    area = x.sum(dim=1)  # [bt, c]
    total = area.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p = area / total  # [bt, c]

    # current entropy
    ent = -(p * torch.log2(p + 1e-6)).sum(dim=-1)  # [bt]

    # maximum entropy (uniform)
    u = torch.full((n_parts,), 1.0 / n_parts, device=x.device, dtype=x.dtype)
    max_ent = -(u * torch.log2(u)).sum()

    return (max_ent - ent).mean()
