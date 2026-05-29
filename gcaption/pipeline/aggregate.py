"""Sequence-level aggregation of per-frame captions (GaitMax Supp. B).

Per attribute: categorical -> majority vote, then the embedding of the mode-holding frame closest to the centroid.
Free-text -> the frame closest to the centroid (the paper's voting / self-correction against transient occlusion/blur).
Yields one [a, d] embedding per sequence for CDLoss `cpt` (a=7 attributes, d=768).
"""

from collections import Counter

import torch

from gcaption.pipeline.embed import attr_value
from gcaption.schema import ATTR_ORDER, CATEGORICAL, FrameCaption


def aggregate(frames: list[FrameCaption], embeds: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Aggregate per-frame attribute embeddings to one per-sequence label + vector.

    Shapes
    ------
    frames : list of t FrameCaption
    embeds : [t, a, d] float32, L2-normalized
    returns: (label dict, seq_emb [a, d] float32)
    """
    a, d = embeds.shape[1], embeds.shape[2]
    label: dict = {}
    seq_emb = torch.empty(a, d)
    for j, attr in enumerate(ATTR_ORDER):
        # shape: [t, a, d] -> [t, d]
        col = embeds[:, j, :]
        # shape: [t, d] -> [1, d] (keepdim for the broadcast below)
        centroid = col.mean(dim=0, keepdim=True)
        # shape: [t, d] vs [1, d] -> [t]
        cos = torch.cosine_similarity(col, centroid, dim=-1)
        values = [attr_value(fc, attr) for fc in frames]
        if attr in CATEGORICAL:
            mode_v = Counter(values).most_common(1)[0][0]
            cand = [i for i, v in enumerate(values) if v == mode_v]
            i_star = max(cand, key=lambda i: cos[i].item())
            label[attr] = mode_v
        else:
            i_star = int(cos.argmax().item())
            label[attr] = values[i_star]
        seq_emb[j] = col[i_star]
    return label, seq_emb
