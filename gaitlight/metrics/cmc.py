import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


@torch.inference_mode()
def cmc(
        distance: torch.Tensor,
        query_labels: torch.Tensor, gallery_labels: torch.Tensor,
        query_cams: Optional[torch.Tensor] = None, gallery_cams: Optional[torch.Tensor] = None,
        max_rank: int = 50,
) -> Dict[str, float | torch.Tensor]:
    """
    Calculate Cumulative Matching Characteristics (Person Re-ID).

    :param distance: distance matrix from query to gallery [num_query, num_gallery]
    :param query_labels: [num_query]
    :param gallery_labels: [num_gallery]
    :param query_cams: [num_query]
    :param gallery_cams: [num_gallery]
    :param max_rank: max rank to calculate CMC

    :return:
        cmc: Cumulative Matching Characteristics
        mAP: mean Average Precision
        mINP: mean Inverse Negative Penalty
    """

    # sanity check
    num_q, num_g = distance.shape
    if num_g < max_rank:
        max_rank = num_g // 2
        logger.warning(f'max_rank is changed to {max_rank}')

    # sort and find correct matches
    indices = distance.argsort(dim=1)
    matches = gallery_labels[indices] == query_labels.unsqueeze(1)

    # metrics
    cmc_l, ap_l, inp_l = [], [], []
    num_valid = 0

    # for each query
    for q_idx in range(num_q):
        # find label and camera
        q_label = query_labels[q_idx]

        # filter out the same camera
        mask = torch.zeros_like(gallery_labels, dtype=torch.bool)
        if query_cams is not None and gallery_cams is not None:
            order = indices[q_idx]
            q_cam = query_cams[q_idx]
            mask = (gallery_labels[order] == q_label) & (gallery_cams[order] == q_cam)
        ori_cmc = matches[q_idx][~mask].float()

        # calculate cmc
        if ori_cmc.sum() == 0:
            "query find no match in gallery"
            continue

        # inp
        _cmc = torch.cumsum(ori_cmc, dim=0)
        _match = (ori_cmc == 1).nonzero(as_tuple=False).squeeze()
        max_match = _match if _match.dim() == 0 else _match[-1]
        inp = _cmc[max_match] / (max_match + 1)
        inp_l.append(inp)

        # cmc
        _cmc = torch.clamp(_cmc, max=1)
        cmc_l.append(_cmc[:max_rank])
        num_valid += 1

        # ap
        num_rel = ori_cmc.sum()
        hit = torch.cumsum(ori_cmc, dim=0)
        num_hits = torch.arange(1, len(hit) + 1, device=hit.device)
        precision = hit / num_hits * ori_cmc
        ap = precision.sum() / num_rel
        ap_l.append(ap)

    if num_valid == 0:
        logger.warning('all query failed to hit valid gallery')
        return {'cmc': torch.zeros(max_rank), 'mAP': 0, 'mINP': 0}

    # calculate
    m_cmc = torch.stack(cmc_l).mean(dim=0)
    m_ap = torch.tensor(ap_l).mean()
    m_inp = torch.tensor(inp_l).mean()
    return {'cmc': m_cmc, 'mAP': m_ap, 'mINP': m_inp}


if __name__ == '__main__':
    dist = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.5]], device='cuda')
    l_qu = torch.tensor([1, 2], device='cuda')
    l_gr = torch.tensor([2, 1, 1], device='cuda')
    l_q_cam = torch.tensor([0, 1], device='cuda')
    l_g_cam = torch.tensor([1, 0, 1], device='cuda')

    mtc = cmc(dist, l_qu, l_gr, l_q_cam, l_g_cam)
    print(f'cmc: {mtc["cmc"]}')
    print(f'mAP: {mtc["mAP"]}')
    print(f'mINP: {mtc["mINP"]}')
