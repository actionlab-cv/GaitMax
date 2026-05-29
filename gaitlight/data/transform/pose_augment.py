"""
Training-time augmentations for skeleton (pose) data.

Reproduces the augmentation pipeline from OpenGait's GaitGraph2:
NormalizeEmpty → FlipSequence → InversePosesPre → JointNoise → PointNoise → RandomMove
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

from gaitlight.types import InputBatch

# COCO-17 left-right swap indices
_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class PoseAugment(nn.Module):
    """
    Composite pose augmentation applied to ``batch.seq.pose`` of shape ``[b, t, V, 3]``.

    Applies (in order):
    1. **NormalizeEmpty** — fill missing joints (x=0) with frame centroid, conf=0.
    2. **FlipSequence** — temporally reverse the sequence (p=flip_prob).
    3. **InversePosesPre** — left-right mirror by swapping joint indices (p=mirror_prob, per-frame).
    4. **JointNoise** — per-joint spatial offset shared across time (std=joint_std).
    5. **PointNoise** — independent noise on every keypoint (std=point_std).
    6. **RandomMove** — global translation of the entire skeleton (random_r=[rx, ry]).
    """

    def __init__(
            self,
            flip_prob: float = 0.5,
            mirror_prob: float = 0.1,
            joint_std: float = 0.25,
            point_std: float = 0.05,
            random_r: list[float] | None = None,
    ):
        super().__init__()
        self.flip_prob = flip_prob
        self.mirror_prob = mirror_prob
        self.joint_std = joint_std
        self.point_std = point_std
        self.random_r = random_r or [4.0, 1.0]
        self.register_buffer('_flip_idx', torch.tensor(_FLIP_IDX, dtype=torch.long))

    @torch.no_grad()
    def forward(self, batch: InputBatch) -> InputBatch:
        pose = batch.seq.pose  # [b, t, V, 3]
        if pose is None:
            return batch

        pose = pose.clone().float()

        for i in range(pose.shape[0]):
            pose[i] = self._augment_single(pose[i])

        new_seq = dataclasses.replace(batch.seq, pose=pose)
        return dataclasses.replace(batch, seq=new_seq)

    def _augment_single(self, x: Tensor) -> Tensor:
        """Augment a single sequence [t, V, 3]."""
        # 1. NormalizeEmpty: fill joints where x-coord == 0
        x = _normalize_empty(x)

        # 2. FlipSequence: temporal reversal
        if torch.rand(1).item() < self.flip_prob:
            x = x.flip(0)

        # 3. InversePosesPre: left-right mirror (per-frame)
        mask = torch.rand(x.shape[0], device=x.device) < self.mirror_prob
        if mask.any():
            x[mask] = x[mask][:, self._flip_idx]

        # 4. JointNoise: same offset per joint across all frames
        noise = torch.zeros(x.shape[1], 3, device=x.device, dtype=x.dtype)
        noise[:, :2].normal_(0, self.joint_std)
        x = x + noise.unsqueeze(0)  # broadcast over t

        # 5. PointNoise: independent per-keypoint
        x = x + torch.randn_like(x) * self.point_std

        # 6. RandomMove: global translation
        shift = torch.zeros(3, device=x.device, dtype=x.dtype)
        shift[0].uniform_(-self.random_r[0], self.random_r[0])
        shift[1].uniform_(-self.random_r[1], self.random_r[1])
        x = x + shift

        return x


def _normalize_empty(x: Tensor) -> Tensor:
    """Fill missing joints (x-coord == 0) with frame centroid, set conf=0. [t, V, 3]"""
    missing = x[:, :, 0] == 0  # [t, V]
    if not missing.any():
        return x
    centroid = x.mean(dim=1, keepdim=True)  # [t, 1, 3]
    centroid_expand = centroid.expand_as(x)  # [t, V, 3]
    x = x.clone()
    x[missing] = centroid_expand[missing]
    x[missing, 2] = 0  # confidence = 0 for filled joints
    return x
