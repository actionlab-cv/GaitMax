"""
Branch-level confidence router for adaptive distance fusion.

Maintains running statistics (EMA) of per-branch embeddings and learns
to evaluate per-sample branch confidence from deviation features that
preserve part-wise structure.

Confidence is used in pairwise-weighted triplet distance:
    D(i,j) = (w_sem * d_sem + w_kin * d_kin) / (w_sem + w_kin)
    where w[i,j] = conf[i] * conf[j]

When one side has low confidence → that branch's distance is suppressed
→ does not interfere with the other branch's matching.
"""

import random

import torch
import torch.nn.functional as func
from torch import nn, Tensor

__all__ = ['Router']


class Router(nn.Module):
    """
    Per-sample, per-branch confidence from embedding structure analysis.

    Computes deviation features that capture both overall quality (vs running
    mean) and part-wise structure (diversity, norm distribution), then maps
    to a scalar confidence via a small learned MLP.

    :param sem_dim: semantic embedding dimension.
    :param kin_dim: kinematic embedding dimension.
    :param momentum: EMA momentum for running statistics.
    :param dropout_prob: probability of zeroing one branch's conf during training.
    """

    def __init__(
        self,
        sem_dim: int = 256,
        kin_dim: int = 256,
        momentum: float = 0.01,
        dropout_prob: float = 0.15,
    ):
        super().__init__()

        # running statistics of centroid (EMA, not trained)
        self.register_buffer('sem_running_mean', torch.zeros(sem_dim))
        self.register_buffer('sem_running_var', torch.ones(sem_dim))
        self.register_buffer('kin_running_mean', torch.zeros(kin_dim))
        self.register_buffer('kin_running_var', torch.ones(kin_dim))

        self.momentum = momentum
        self.dropout_prob = dropout_prob

        # deviation features (7 per branch) → confidence
        # [mean_z, std_z, cosine_to_mean, centroid_norm,
        #  part_diversity, part_norm_mean, part_norm_std]
        n_feat = 7
        self.sem_gate = nn.Sequential(
            nn.Linear(n_feat, n_feat * 4), nn.ReLU(inplace=True), nn.Linear(n_feat * 4, 1),
        )
        self.kin_gate = nn.Sequential(
            nn.Linear(n_feat, n_feat * 4), nn.ReLU(inplace=True), nn.Linear(n_feat * 4, 1),
        )

    def forward(self, embed_sem: Tensor, embed_kin: Tensor) -> dict[str, Tensor]:
        """
        :param embed_sem: semantic embedding  [b, d_sem, p_sem]
        :param embed_kin: kinematic embedding  [b, d_kin, p_kin]
        :return:
            - conf_sem: per-sample semantic confidence [b] in (0, 1)
            - conf_kin: per-sample kinematic confidence [b] in (0, 1)
        """
        sem = embed_sem.detach()  # [b, d, p]
        kin = embed_kin.detach()  # [b, d, p]

        if self.training:
            self._update_ema(sem, kin)

        conf_sem = self._compute_conf(sem, self.sem_running_mean, self.sem_running_var, self.sem_gate)
        conf_kin = self._compute_conf(kin, self.kin_running_mean, self.kin_running_var, self.kin_gate)

        # branch dropout during training
        if self.training and random.random() < self.dropout_prob:
            if random.random() < 0.5:
                conf_sem = torch.zeros_like(conf_sem)
            else:
                conf_kin = torch.zeros_like(conf_kin)

        return {'conf_sem': conf_sem, 'conf_kin': conf_kin}

    @staticmethod
    def _compute_conf(embed: Tensor, mean: Tensor, var: Tensor, gate: nn.Module) -> Tensor:
        """
        Compute per-sample confidence from part-aware deviation features.

        :param embed: [b, d, p]
        :param mean: running mean of centroid [d]
        :param var: running variance of centroid [d]
        :param gate: MLP → scalar confidence
        :return: [b] confidence in (0, 1)
        """
        b, d, p = embed.shape

        # --- centroid features (overall quality vs history) ---
        # cast to float32 to match running buffers
        embed = embed.float()
        centroid = embed.mean(dim=-1)                            # [b, d]
        diff = centroid - mean.unsqueeze(0)
        std = (var.unsqueeze(0) + 1e-8).sqrt()
        z = diff / std                                            # [b, d]

        f_mean_z = z.mean(dim=-1)                                 # [b] mean z-score
        f_std_z = z.std(dim=-1)                                   # [b] z-score spread
        f_cosine = func.cosine_similarity(centroid, mean.unsqueeze(0), dim=-1)  # [b]
        f_centroid_norm = centroid.norm(dim=-1)                   # [b]

        # --- part structure features (diversity / collapse detection) ---
        part_norms = embed.norm(dim=1)                            # [b, p]
        f_part_norm_mean = part_norms.mean(dim=-1)                # [b]
        f_part_norm_std = part_norms.std(dim=-1)                  # [b]

        # inter-part diversity: variance of embeddings across parts
        # high = diverse parts, low = collapsed (all parts identical)
        f_diversity = embed.var(dim=-1).mean(dim=-1)              # [b]

        # --- concat → MLP → sigmoid ---
        feat = torch.stack([
            f_mean_z, f_std_z, f_cosine, f_centroid_norm,
            f_diversity, f_part_norm_mean, f_part_norm_std,
        ], dim=-1)  # [b, 7]

        return gate(feat).squeeze(-1).sigmoid()                   # [b]

    @torch.no_grad()
    def _update_ema(self, sem: Tensor, kin: Tensor) -> None:
        """Update running mean and variance of centroids with EMA."""
        m = self.momentum
        for embed, rmean, rvar in [
            (sem, self.sem_running_mean, self.sem_running_var),
            (kin, self.kin_running_mean, self.kin_running_var),
        ]:
            centroid = embed.mean(dim=-1).float()  # [b, d], cast to float32 for EMA buffers
            rmean.lerp_(centroid.mean(dim=0), m)
            rvar.lerp_(centroid.var(dim=0, correction=0), m)
