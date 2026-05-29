import torch
from torch import nn


def euc_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between x and y.

    :param x: [n, c, p]
    :param y: [m, c, p]

    :return: distance matrix [n, m]
    """

    # expand part (if needed)
    # [n, c] -> [n, c, 1]
    x = x.unsqueeze(-1) if x.dim() == 2 else x
    y = y.unsqueeze(-1) if y.dim() == 2 else y

    # [n, c, p] -> [p, n, c]
    x_bins = x.permute(2, 0, 1).to(torch.float32)
    y_bins = y.permute(2, 0, 1).to(torch.float32)

    # distance
    dists = torch.cdist(x_bins, y_bins, p=2).to(x.dtype)
    dist_avg = dists.mean(dim=0)
    dist_avg[torch.isnan(dist_avg)] = float('inf')

    return dist_avg


def cos_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between x and y.

    :param x: [n, c, p]
    :param y: [m, c, p]

    :return: distance matrix [n, m]
    """

    # norm
    x_norm = nn.functional.normalize(x, p=2, dim=1)
    y_norm = nn.functional.normalize(y, p=2, dim=1)

    # [n, c, p] -> [p, n, c]
    x_bins = x_norm.permute(2, 0, 1)
    y_bins = y_norm.permute(2, 0, 1)

    # distance
    sim = torch.einsum("bik,bjk->bij", x_bins, y_bins)
    sim_avg = sim.mean(dim=0)
    sim_avg[torch.isnan(sim_avg)] = float('inf')

    return 1 - sim_avg
