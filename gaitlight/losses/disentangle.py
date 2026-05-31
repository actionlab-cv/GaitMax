import logging
from typing import Any

import einops
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

from gaitlight.losses._common import EPS

logger = logging.getLogger(__name__)


class DisentangleLoss(nn.Module):
    def __init__(self, config: DictConfig | ListConfig = None):
        super().__init__()

        self.eps = EPS

    def forward(self, u_data: Any) -> dict[str, Any]:
        """
        Conditional Decorrelation Loss: minimize the squared Pearson correlation
        between pairwise gait-embedding distances and pairwise caption similarities
        (over same-identity pairs, per part x attribute).

        :param u_data: {'embed': [n, c, p], 'label': [n], 'cpt': [n, l, d]}
        :return:
            - visual: {}
            - loss: {'cd'}
        """

        # embed [n, c, p]
        assert 'embed' in u_data, 'embed not found in batch'
        embed = u_data['embed']

        # expand part (if needed)
        embed = embed.unsqueeze(-1) if embed.dim() == 2 else embed
        n, c, p = embed.shape

        # labels [n]
        assert 'label' in u_data, 'label not found in batch'
        label: torch.Tensor = u_data['label']

        # caption
        assert 'cpt' in u_data, 'caption not found in batch'
        cpt = u_data['cpt']

        # meta holder
        loss, visual = {}, {}

        # find same individual pair
        mat = label.unsqueeze(1) == label.unsqueeze(0)  # [n, n]
        idx = torch.arange(n, device=embed.device)
        mat[idx, idx] = False
        i_idx, j_idx = mat.nonzero(as_tuple=True)

        # property similarity
        # [l, n]
        att = einops.rearrange(cpt, 'n l d -> l n d')
        f_a, f_b = att[:, i_idx, :], att[:, j_idx, :]
        sim = torch.cosine_similarity(f_a, f_b, dim=2, eps=self.eps)
        _mu = sim.mean(dim=1, keepdim=True)
        _std = sim.std(dim=1, unbiased=False, keepdim=True) + self.eps
        sim = (sim - _mu) / _std

        # embedding distance
        # [p, n]
        emb = einops.rearrange(embed, 'n c p -> p n c')
        f_a, f_b = emb[:, i_idx, :], emb[:, j_idx, :]
        # eps-stabilized L2 (torch.norm has nan gradient at exactly-zero distance)
        dis = torch.sqrt(((f_a - f_b) ** 2).sum(dim=2) + self.eps)
        _mu = dis.mean(dim=1, keepdim=True)
        _std = dis.std(dim=1, unbiased=False, keepdim=True) + self.eps
        dis = (dis - _mu) / _std

        # correlation
        # [p, l]
        dis = dis.unsqueeze(1)  # [p, 1, n]
        sim = sim.unsqueeze(0)  # [1, l, n]
        corr = (dis * sim).mean(dim=2)

        # loss
        l_avg = corr.pow(2).mean()
        loss |= {'cd': l_avg}

        return {'loss': loss, 'visual': visual}
