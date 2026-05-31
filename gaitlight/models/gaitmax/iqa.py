"""
Lightweight frame quality estimator.

Produces per-frame quality scores from HumanPrior signals (mask coverage
and slot confidence).  Uses softmax over the temporal axis so scores are
relative (sum to 1), avoiding the drift problem of independent sigmoid
scoring.

Scores are used as soft weights in quality-gated temporal pooling and as
attention bias in kinematic temporal attention.
"""

import torch
import torch.nn.functional as func
from torch import nn, Tensor

__all__ = ['FrameQuality']


class FrameQuality(nn.Module):
    """
    Learned per-frame quality scoring from HumanPrior outputs.

    Scores are relative across frames (softmax over time), not absolute.
    A learnable temperature controls distribution sharpness.

    :param n_slots: number of pattern slots (m from HumanPrior).
    :param init_tau: initial temperature for softmax (higher = more uniform).
    """

    def __init__(self, n_slots: int = 11, init_tau: float = 1.0):
        super().__init__()

        # input: per-slot conf (n_slots) + mean_conf (1) + mask_coverage (1)
        in_dim = n_slots + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim * 2, 1),
        )

        # learnable temperature: controls softmax sharpness
        self.log_tau = nn.Parameter(torch.tensor(init_tau).log())

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def forward(self, conf: Tensor, mask: Tensor) -> Tensor:
        """
        :param conf: per-slot confidence [b, t, m]
        :param mask: foreground mask [b, t, h, w] bool
        :return: quality scores [b, t], softmax over t (sum=1 per sample)
        """
        b, t, m = conf.shape

        # features: [b, t, m+2]
        mask_cov = mask.float().mean(dim=(-2, -1))  # [b, t]
        mean_conf = conf.mean(dim=-1)                # [b, t]
        feat = torch.cat([conf, mean_conf.unsqueeze(-1), mask_cov.unsqueeze(-1)], dim=-1)

        # raw logits → temperature-scaled softmax over time
        logits = self.mlp(feat).squeeze(-1)          # [b, t]
        quality = func.softmax(logits / self.tau, dim=-1)  # [b, t], sum=1

        return quality
