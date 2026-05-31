import logging
import math

import einops
import torch
import torch.nn.functional as func
from torch import nn

logger = logging.getLogger(__name__)


class SpatialPrior(nn.Module):
    def __init__(self, s_dim: int = 384, h_dim: int = 64, size: tuple[int, int] = (64, 32)):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(h_dim, 2 * h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * h_dim),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * h_dim, 4 * h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * h_dim),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * h_dim, 4 * h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * h_dim),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Conv2d(h_dim, s_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(2 * h_dim, s_dim, kernel_size=1, bias=True)
        self.fc3 = nn.Conv2d(4 * h_dim, s_dim, kernel_size=1, bias=True)
        self.fc4 = nn.Conv2d(4 * h_dim, s_dim, kernel_size=1, bias=True)

        self.map_size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c1 = func.adaptive_avg_pool2d(self.fc1(c1), self.map_size)
        c2 = func.interpolate(self.fc2(c2), size=self.map_size, mode='bilinear', align_corners=False)
        c3 = func.interpolate(self.fc3(c3), size=self.map_size, mode='bilinear', align_corners=False)
        c4 = func.interpolate(self.fc4(c4), size=self.map_size, mode='bilinear', align_corners=False)

        return torch.stack([c1, c2, c3, c4], dim=0)  # [4, bt, s_dim, h, w]


class DINOWrapper(nn.Module):
    def __init__(self, version: str, weights: str, chunk: int, select: list[int], map_size: tuple[int, int], spatial: bool = False):
        super().__init__()

        # backbone
        dino_r = 'facebookresearch/dinov3' if 'dinov3' in version else 'facebookresearch/dinov2'
        self.dino = self._load_dino(dino_r, version, weights if 'dinov2' not in version else None)
        self.dino.requires_grad_(False)
        self.dino_chunk = chunk
        self.dino_select = select

        # dimension
        self.embed_dim = int(self.dino.embed_dim)
        self.patch_size = int(self.dino.patch_size)
        self.size = tuple(map_size)

        # spatial prior
        self.spatial = spatial
        self.spa_prior = SpatialPrior(s_dim=self.embed_dim, h_dim=64, size=self.size) if spatial else None

        # init
        if self.spa_prior is not None:
            self.spa_prior.apply(self._init_weights)

    @staticmethod
    def _load_dino(repo: str, model: str, weights: str) -> nn.Module:
        kwargs = {'verbose': False}
        if 'dinov2' not in repo:
            kwargs |= {'weights': weights}
        return torch.hub.load(repo, model, **kwargs)

    @torch.no_grad()
    def _dino_forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Sample intermediate features from DINO. Input: [bt, c, h, w]"""
        self.dino.eval()

        chunks = torch.split(frame, self.dino_chunk, dim=0)
        l_f = [torch.stack(self.dino.get_intermediate_layers(chunk, n=self.dino_select, reshape=True), dim=0)
               for chunk in chunks]

        f = torch.cat(l_f, dim=1)  # [k, bt, c, h, w]

        # resize: flatten to 4D, interpolate, restore
        k, bt, c, h, w = f.shape
        f = func.interpolate(f.flatten(0, 1), size=self.size, mode='bilinear', align_corners=False)
        return f.unflatten(0, (k, bt))  # [k, bt, c, h, w]

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [b, t, c, h, w] -> [bt, c, h, w]
        b, t, c, h, w = x.shape
        x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        # [k, bt, c, h, w]
        f = self._dino_forward(x_flat)

        if self.spatial:
            # [4, bt, s_dim, h, w]
            sp = self.spa_prior(x_flat)
            k = f.shape[0]
            assert k % sp.shape[0] == 0, f'select={k} must be divisible by spatial prior levels={sp.shape[0]}'
            f = f + torch.repeat_interleave(sp, k // sp.shape[0], dim=0)

        return einops.rearrange(f, 'k (b t) c h w -> k b t c h w', t=t)
