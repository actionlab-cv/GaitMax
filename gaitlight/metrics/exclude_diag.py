import torch


def exclude_diag(acc: torch.Tensor, reduce: bool = True):
    """
    Exclude diagonal elements from accuracy matrix.

    :param acc: accuracy matrix
    :param reduce: reduce mean or not

    :return: accuracy matrix without diagonal elements
    """

    div = acc.shape[1] - 1
    mask = 1 - torch.eye(acc.size(0), acc.size(1), device=acc.device, dtype=acc.dtype)
    rst = (acc * mask).sum(dim=1) / div

    return rst.mean() if reduce else rst
