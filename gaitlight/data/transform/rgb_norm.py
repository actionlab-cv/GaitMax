import dataclasses

import torch
from torch import nn

from gaitlight.types import InputBatch


class RGBNorm(nn.Module):
    def __init__(self, mean: tuple[float] = None, std: tuple[float] = None):
        super().__init__()

        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        self.register_buffer('mean', torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, batch: InputBatch) -> InputBatch:
        frame = batch.seq.frame  # [b, t, 3, h, w]
        if frame is None:
            return batch

        f = frame.float() / 255 if frame.dtype == torch.uint8 else frame.float()
        f = (f - self.mean) / self.std

        return dataclasses.replace(batch, seq=dataclasses.replace(batch.seq, frame=f))

    @staticmethod
    def inverse(frame: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
        """Reverse ImageNet normalization: [b, t, 3, h, w] normalized -> [0, 1] float."""
        m = torch.tensor(mean, device=frame.device).view(1, 1, 3, 1, 1)
        s = torch.tensor(std, device=frame.device).view(1, 1, 3, 1, 1)
        return (frame * s + m).clamp(0, 1)
