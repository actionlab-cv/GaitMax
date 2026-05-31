"""
Part-wise head modules for gait recognition.

Shared by both semantic and kinematic branches.
"""

import einops
import torch
from torch import nn

__all__ = ['PartLinear', 'PartHead']


class PartLinear(nn.Module):
    """
    Per-part fully connected projection.

    :param num_parts: number of body parts.
    :param s_dim: input dimension.
    :param t_dim: output dimension.
    """

    def __init__(self, num_parts: int, s_dim: int, t_dim: int):
        super().__init__()
        self.fc = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_parts, s_dim, t_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [b, s_dim, part]
        :return: [b, t_dim, part]
        """
        return torch.einsum('bsp, pst -> btp', x, self.fc).contiguous()


class PartHead(nn.Module):
    """
    BNNeck classification head: BN → L2 normalize → per-part cosine classifier.

    :param num_parts: number of body parts.
    :param num_classes: number of identity classes.
    :param s_dim: embedding dimension.
    """

    def __init__(self, num_parts: int, num_classes: int, s_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_parts * s_dim)
        self.fc = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_parts, s_dim, num_classes)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        :param x: embeddings [b, d, p]
        :return:
            - embed: normalized embeddings [b, d, p]
            - logits: classification logits [b, k, p]
        """
        b, d, p = x.shape

        # global BN across all parts
        x = einops.rearrange(x, 'b d p -> b (d p)')
        x = self.bn(x)
        x = einops.rearrange(x, 'b (d p) -> p b d', p=p)

        # cosine similarity: normalize both embeddings and weights
        x = nn.functional.normalize(x, dim=-1)
        fc = nn.functional.normalize(self.fc, dim=1)
        logits = torch.einsum('pbd, pdk -> pbk', x, fc)

        x = einops.rearrange(x, 'p b d -> b d p')
        logits = einops.rearrange(logits, 'p b k -> b k p')

        return {'embed': x, 'logits': logits}
