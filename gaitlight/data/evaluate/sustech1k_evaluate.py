"""
SUSTech1K dataset evaluation.

Protocol loaded from data/partition/protocol/sustech1k.json.
Gallery: 00-nm.  Probe conditions: Normal, Bag, Clothing, Carrying,
Umbrella, Uniform, Occlusion, Night.
Uses substring matching for sequence types (e.g., 'bg' matches any caption containing 'bg').
Cross-view matrix with identical-view exclusion.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy
import torch
from rich.console import Console
from rich.table import Table

from gaitlight.metrics.distance import euc_dist
from gaitlight.metrics.exclude_diag import exclude_diag

logger = logging.getLogger(__name__)

_PROTOCOL_PATH = Path(__file__).parents[1] / 'partition' / 'protocol' / 'sustech1k.json'


def _contains_match(seq_types: numpy.ndarray, patterns: list[str]) -> numpy.ndarray:
    """Substring match: True if seq_type contains any of the patterns."""
    mask = numpy.zeros(len(seq_types), dtype=bool)
    for pat in patterns:
        mask |= numpy.char.find(seq_types, pat) >= 0
    return mask


@torch.inference_mode()
def sustech1k_evaluate(u_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate SUSTech1K dataset (single-view gallery, cross-view).

    :param u_data: dict with 'embed' [n, c, p], 'label' [n], 'meta' [list of SequenceMeta]
    :return: dict with 'visual' and 'score'
    """
    console = Console()

    protocol = json.loads(_PROTOCOL_PATH.read_text())
    gallery_seqs = protocol['gallery']
    probe_groups = protocol['probe']
    group_names = protocol['names']

    embed: torch.Tensor = u_data['embed']
    label: torch.Tensor = u_data['label']
    seq_type = numpy.array([m.caption for m in u_data['meta']])
    view = numpy.array([m.view for m in u_data['meta']])
    view_list = sorted(set(view))
    num_rank = 5

    visual = {}

    # cross-view accuracy: [conditions, probe_view, gallery_view, rank]
    cross_acc = torch.full((len(group_names), len(view_list), len(view_list), num_rank), -1.0)

    for p_idx, name in enumerate(group_names):
        probe_seqs = probe_groups[name]

        for pv_idx, pv in enumerate(view_list):
            for gv_idx, gv in enumerate(view_list):
                # gallery: substring match
                g_mask = _contains_match(seq_type, gallery_seqs) & (view == gv)
                g_embed, g_label = embed[g_mask], label[g_mask]

                # probe: substring match
                p_mask = _contains_match(seq_type, probe_seqs) & (view == pv)
                p_embed, p_label = embed[p_mask], label[p_mask]

                if len(p_embed) == 0 or len(g_embed) == 0:
                    continue

                dist = euc_dist(p_embed, g_embed)
                indices = dist.argsort(dim=1)
                matches = p_label.reshape(-1, 1) == g_label[indices[:, :num_rank]]
                acc = torch.sum(torch.cumsum(matches, dim=1) > 0, dim=0).float() / dist.shape[0]
                cross_acc[p_idx, pv_idx, gv_idx] = acc

    # table: exclude identical-view
    table = Table(title="[bold]SUSTech1K Rank-1 (Exclude Identical-View)[/]")
    table.add_column("Condition", justify="center")
    table.add_column("Rank-1", justify="center")
    table.add_column("Rank-5", justify="center")

    for i, name in enumerate(group_names):
        r1 = exclude_diag(cross_acc[i, :, :, 0])
        r5 = exclude_diag(cross_acc[i, :, :, 4])
        table.add_row(name, f'{r1 * 100:.2f}', f'{r5 * 100:.2f}')

    console.print('\n')
    console.print(table)

    # wandb: only per-condition R1
    visual['gait'] = {
        name.lower(): exclude_diag(cross_acc[i, :, :, 0]).item()
        for i, name in enumerate(group_names)
    }

    # detail: R1 and R5 per condition
    detail = {}
    for i, name in enumerate(group_names):
        key = name.lower()
        detail[f'{key}_r1'] = exclude_diag(cross_acc[i, :, :, 0]).item()
        detail[f'{key}_r5'] = exclude_diag(cross_acc[i, :, :, 4]).item()

    return {'visual': visual, 'detail': detail, 'score': visual['gait']['normal']}
