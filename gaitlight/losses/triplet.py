import logging
from typing import Any

import einops
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from gaitlight.losses._common import EPS

logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    def __init__(self, config: DictConfig | ListConfig = None, margin: float = 0.2):
        super().__init__()

        self.dynamic = config is not None
        if self.dynamic:
            self.config = config
            self.global_step = 0

        self.s_margin = margin

    @property
    def margin(self, eps: float = EPS) -> float:
        if not self.dynamic:
            return self.s_margin

        # dynamic margin
        warm = self.config.warm
        total = self.config.total
        st, ed = OmegaConf.to_object(self.config.margin)
        t = max(0, self.global_step - warm)
        t = min(t / max(total - warm, eps), 1.0)
        return st + t * (ed - st)

    def step(self):
        if self.dynamic:
            self.global_step += 1

    def forward(self, u_data: Any) -> dict[str, Any]:
        """
        Triplet loss for embedding.

        Supports two modes:
        1. Standard: u_data = {'embed': [n, c, p], 'label': [n]}
           Per-part L2 distance, averaged across parts.
        2. Weighted fusion: u_data = {'embed_sem': [n, c, p1], 'embed_kin': [n, c, p2],
                                       'label': [n], 'conf_sem': [n], 'conf_kin': [n]}
           Pairwise-weighted distance from two branches.

        When 'conf_sem' is absent, falls back to standard mode.

        :return:
            - visual: {'dist', 'hard', 'triplet', 'active', 'ap', 'an'}
            - loss: {'triplet'}
        """
        if 'conf_sem' in u_data:
            return self._forward_weighted(u_data)
        return self._forward_standard(u_data)

    # ---- standard mode (unchanged) ----

    def _forward_standard(self, u_data: Any) -> dict[str, Any]:
        """Original per-part triplet loss."""
        embed = u_data['embed']
        embed = embed.unsqueeze(-1) if embed.dim() == 2 else embed
        n, c, p = embed.shape
        label = u_data['label']

        loss, visual = {}, {}

        # distance [p, n, n]
        embed = einops.rearrange(embed, 'n c p -> p n c')
        dist = self._distance_v2(embed)

        visual |= {'dist': dist.mean().detach()}

        # triplet
        _trip = self._to_triplet_v1(label, dist)
        ap, an = _trip['anc_pos'], _trip['anc_neg']
        visual |= {'ap': ap.mean().detach(), 'an': an.mean().detach()}

        diff = (ap - an).view(p, -1)
        l_tri = torch.relu(diff + self.margin)

        hard = l_tri.max(dim=-1)[0]
        visual |= {'hard': hard.mean().detach()}

        _loss = self._avg_non_zero(l_tri)
        l_avg, l_num = _loss['avg'], _loss['num']
        visual |= {'triplet': l_avg.detach(), 'active': l_num.detach()}
        loss |= {'triplet': l_avg}

        return {'loss': loss, 'visual': visual}

    # ---- weighted fusion mode ----

    def _forward_weighted(self, u_data: Any) -> dict[str, Any]:
        """
        Pairwise-weighted triplet loss from two branches.

        Distance = (w_sem * d_sem + w_kin * d_kin) / (w_sem + w_kin + eps)
        where w[i,j] = conf[i] * conf[j]  (both sides must be confident).
        """
        embed_sem = u_data['embed_sem']  # [n, c, p_sem]
        embed_kin = u_data['embed_kin']  # [n, c, p_kin]
        label = u_data['label']          # [n]
        conf_sem = u_data['conf_sem']    # [n]
        conf_kin = u_data['conf_kin']    # [n]

        embed_sem = embed_sem.unsqueeze(-1) if embed_sem.dim() == 2 else embed_sem
        embed_kin = embed_kin.unsqueeze(-1) if embed_kin.dim() == 2 else embed_kin

        loss, visual = {}, {}

        # per-branch distance: [p, n, n] → mean across parts → [n, n]
        d_sem = self._branch_dist(embed_sem)  # [n, n]
        d_kin = self._branch_dist(embed_kin)  # [n, n]

        # pairwise weights: [n, n]
        w_sem = conf_sem.unsqueeze(1) * conf_sem.unsqueeze(0)  # [n, n]
        w_kin = conf_kin.unsqueeze(1) * conf_kin.unsqueeze(0)  # [n, n]

        # weighted fusion distance: [n, n]
        dist = (w_sem * d_sem + w_kin * d_kin) / (w_sem + w_kin + 1e-8)

        visual |= {'dist': dist.mean().detach()}

        # triplet on fused distance: treat as 1 "part"
        dist = dist.unsqueeze(0)  # [1, n, n]
        _trip = self._to_triplet_v1(label, dist)
        ap, an = _trip['anc_pos'], _trip['anc_neg']
        visual |= {'ap': ap.mean().detach(), 'an': an.mean().detach()}

        diff = (ap - an).view(1, -1)
        l_tri = torch.relu(diff + self.margin)

        hard = l_tri.max(dim=-1)[0]
        visual |= {'hard': hard.mean().detach()}

        _loss = self._avg_non_zero(l_tri)
        l_avg, l_num = _loss['avg'], _loss['num']
        visual |= {'triplet': l_avg.detach(), 'active': l_num.detach()}
        loss |= {'triplet': l_avg}

        return {'loss': loss, 'visual': visual}

    @staticmethod
    def _branch_dist(embed: torch.Tensor) -> torch.Tensor:
        """
        Compute mean per-part L2 distance for a single branch.

        :param embed: [n, c, p]
        :return: [n, n]
        """
        x = einops.rearrange(embed, 'n c p -> p n c')
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        dist = x2 + x2.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
        dist = torch.sqrt(torch.relu(dist) + EPS)  # [p, n, n]
        return dist.mean(dim=0)  # [n, n]

    @staticmethod
    def _distance_v1(x: torch.Tensor) -> torch.Tensor:
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)
        y2 = torch.sum(x ** 2, -1).unsqueeze(1)
        inner = x.matmul(x.transpose(1, 2))
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(torch.relu(dist) + EPS)  # [p, n_x, n_y]
        return dist

    @staticmethod
    def _distance_v2(x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance of x [p, n, dim]

        :param x: input tensor [p, n, dim]
        :return: pairwise distance matrix [p, n, n]
        """

        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)  # [p, n, 1]
        dist = x2 + x2.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
        return torch.sqrt(torch.relu(dist) + EPS)

    @staticmethod
    def _to_triplet_v1(label: torch.Tensor, dist: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convert distance matrix to triplet distances.

        :param label: labels [n]
        :param dist: pairwise distance matrix [p, n, n]
        :return:
            anc_pos: anchor positive distance [p, n, a, 1]
            anc_neg: anchor negative distance [p, n, 1, a]
        """
        p, n = dist.shape[:2]

        matches = label.unsqueeze(1) == label.unsqueeze(0)  # [n, n]
        ap = dist[:, matches].view(p, n, -1, 1)
        an = dist[:, ~matches].view(p, n, 1, -1)

        return {'anc_pos': ap, 'anc_neg': an}

    @staticmethod
    def _to_triplet_v2(label: torch.Tensor, dist: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convert distance matrix to triplet distances.

        :param label: labels [n]
        :param dist: pairwise distance matrix [p, n, n]
        :return:
            anc_pos: anchor positive distance [p, n, a, 1]
            anc_neg: anchor negative distance [p, n, 1, a]
        """
        p, n = dist.shape[:2]

        matches = label.unsqueeze(0).unsqueeze(2) == label.unsqueeze(0).unsqueeze(1)
        ap = dist * matches.float()
        an = dist + (1e6 * matches.float())

        ap = ap.view(p, n, -1, 1)
        an = an.view(p, n, 1, -1)

        return {'anc_pos': ap, 'anc_neg': an}

    @staticmethod
    def _avg_non_zero(x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute average of non-zero elements.

        :param x: tensor [p, n, ...]
        :return:
            avg: average loss per block
            num: number of non-zero elements
        """

        eps = EPS
        l_sum = x.sum(dim=-1)
        l_num = (x != 0).sum(dim=-1).float()
        l_avg = l_sum / (l_num + eps)
        l_avg[l_num == 0] = 0

        return {'avg': l_avg.mean(), 'num': l_num.mean()}


if __name__ == '__main__':
    input = {
        'embed': torch.rand(64, 384, 16).cuda().requires_grad_(),
        'label': torch.tensor([0, 1, 2, 3] * 16).cuda().requires_grad_(),
    }

    net = TripletLoss().cuda()
    out = net(input)

    print(out['loss']['triplet'])
