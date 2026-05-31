"""
Fusion backbone from BigGait (Ye et al.).

Two-branch pre-processing (denoising / appearance), attention-based fusion,
and shared post-processing ResNet backbone.  All convolutions are frame-wise
2D operations applied independently to each temporal frame.

Reference config (BigGait):
    Pre:   Conv(16→64) + 1 BasicBlock(64)
    Fusion: 1×1(128→4) → 3×3(4) → 1×1(4→128), softmax gate over 2 branches
    Post:  BasicBlock stages 64→128→256→512, strides [2, 2, 1]
"""

import einops
import torch
import torch.nn.functional as func
from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock


class FusionBackbone(nn.Module):
    """
    Pre-ResNet ×2 → AttentionFusion → Post-ResNet.

    Operates frame-wise: all convolutions are 2D, applied to each temporal
    frame independently via reshape.

    :param in_dim: input channel dimension from denoising / appearance branches.
    :param channels: channel widths for each ResNet stage.
    :param layers: number of BasicBlocks per stage [pre, post1, post2, post3].
    :param strides: spatial stride per stage [pre, post1, post2, post3].
    """

    def __init__(
            self,
            in_dim: int = 16,
            channels: tuple[int, ...] = (64, 128, 256, 512),
            layers: tuple[int, ...] = (1, 1, 1, 1),
            strides: tuple[int, ...] = (1, 2, 2, 1),
    ):
        super().__init__()

        # --- per-branch shallow encoder (Pre) ---
        self.pre_den = _PreStage(in_dim, channels[0], layers[0], strides[0])
        self.pre_app = _PreStage(in_dim, channels[0], layers[0], strides[0])

        # --- attention fusion ---
        self.fusion = _AttentionFusion(channels[0], squeeze_ratio=16, n_branch=2)

        # --- shared deep backbone (Post) ---
        self.post = _PostStage(channels, layers[1:], strides[1:])

    def forward(self, f_den: Tensor, f_app: Tensor) -> Tensor:
        """
        :param f_den: denoising features [b, t, c_in, h, w]
        :param f_app: appearance features [b, t, c_in, h, w]
        :return: fused features [b, c_out, h', w']  (temporally pooled)
        """
        # frame-wise pre-processing
        f_den = self._framewise(self.pre_den, f_den)  # [b, t, C0, h, w]
        f_app = self._framewise(self.pre_app, f_app)  # [b, t, C0, h, w]

        # attention fusion: [b, t, C0, h, w]
        fused = self._framewise_fusion(f_den, f_app)

        # frame-wise post-processing: [b, t, C3, h', w']
        fused = self._framewise(self.post, fused)

        return fused

    @staticmethod
    def _framewise(module: nn.Module, x: Tensor) -> Tensor:
        """Apply a 2D module to each frame independently."""
        b, t, c, h, w = x.shape
        out = module(x.reshape(b * t, c, h, w))
        return out.reshape(b, t, *out.shape[1:])

    def _framewise_fusion(self, f_den: Tensor, f_app: Tensor) -> Tensor:
        """Apply attention fusion frame-wise."""
        b, t, c, h, w = f_den.shape
        f_den = f_den.reshape(b * t, c, h, w)
        f_app = f_app.reshape(b * t, c, h, w)
        out = self.fusion(f_den, f_app)
        return out.reshape(b, t, c, h, w)


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _PreStage(nn.Module):
    """Conv2d stem + one BasicBlock stage (Pre_ResNet9 equivalent)."""

    def __init__(self, in_dim: int, out_dim: int, n_blocks: int, stride: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = self._make_blocks(out_dim, out_dim, n_blocks, stride)

    @staticmethod
    def _make_blocks(in_dim: int, out_dim: int, n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_dim != out_dim * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_dim * BasicBlock.expansion),
            )
        layers = [BasicBlock(in_dim, out_dim, stride, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_dim * BasicBlock.expansion, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))


class _AttentionFusion(nn.Module):
    """
    Channel-wise attention gate that softly selects between n branches.

    Concat branch features → bottleneck conv → softmax scores → weighted sum.
    """

    def __init__(self, in_channels: int, squeeze_ratio: int = 16, n_branch: int = 2):
        super().__init__()
        hidden = in_channels // squeeze_ratio
        self.n_branch = n_branch

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * n_branch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels * n_branch, 1, bias=False),
        )

    def forward(self, *branches: Tensor) -> Tensor:
        """
        :param branches: n tensors of shape [bt, c, h, w]
        :return: fused [bt, c, h, w]
        """
        cat = torch.cat(branches, dim=1)  # [bt, n*c, h, w]
        score = self.gate(cat)  # [bt, n*c, h, w]
        score = einops.rearrange(score, 'bt (n c) h w -> bt c n h w', n=self.n_branch)
        score = func.softmax(score, dim=2)  # compete over branches

        out = torch.zeros_like(branches[0])
        for i, br in enumerate(branches):
            out = out + br * score[:, :, i]
        return out


class _PostStage(nn.Module):
    """Stacked BasicBlock stages (Post_ResNet9 equivalent)."""

    def __init__(
            self,
            channels: tuple[int, ...],
            layers: tuple[int, ...],
            strides: tuple[int, ...],
    ):
        super().__init__()
        stages = []
        in_dim = channels[0]
        for ch, n_blk, s in zip(channels[1:], layers, strides):
            stages.append(self._make_stage(in_dim, ch, n_blk, s))
            in_dim = ch * BasicBlock.expansion
        self.stages = nn.Sequential(*stages)

    @staticmethod
    def _make_stage(in_dim: int, out_dim: int, n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_dim != out_dim * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_dim * BasicBlock.expansion),
            )
        layers = [BasicBlock(in_dim, out_dim, stride, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_dim * BasicBlock.expansion, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stages(x)
