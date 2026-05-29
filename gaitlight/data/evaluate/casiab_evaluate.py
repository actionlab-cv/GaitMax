"""
CASIA-B dataset evaluation.

Protocol loaded from data/partition/protocol/casiab.json.
Gallery: nm-01 ~ nm-04.  Probe conditions: NM, BG, CL.
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

_PROTOCOL_PATH = Path(__file__).parents[1] / 'partition' / 'protocol' / 'casiab.json'


@torch.inference_mode()
def casiab_evaluate(u_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate CASIA-B dataset (single-view gallery, cross-view).

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
                # gallery
                g_mask = numpy.isin(seq_type, gallery_seqs) & (view == gv)
                g_embed, g_label = embed[g_mask], label[g_mask]

                # probe
                p_mask = numpy.isin(seq_type, probe_seqs) & (view == pv)
                p_embed, p_label = embed[p_mask], label[p_mask]

                if len(p_embed) == 0 or len(g_embed) == 0:
                    continue

                dist = euc_dist(p_embed, g_embed)
                indices = dist.argsort(dim=1)
                matches = p_label.reshape(-1, 1) == g_label[indices[:, :num_rank]]
                acc = torch.sum(torch.cumsum(matches, dim=1) > 0, dim=0).float() / dist.shape[0]
                cross_acc[p_idx, pv_idx, gv_idx] = acc

    # table: exclude identical-view
    table = Table(title="[bold]CASIA-B Summary (Exclude Identical-View)[/]")
    table.add_column("Condition", justify="center")
    table.add_column("Rank-1", justify="center")
    table.add_column("Rank-5", justify="center")

    r1_vals, r5_vals = [], []
    for i, name in enumerate(group_names):
        r1 = exclude_diag(cross_acc[i, :, :, 0])
        r5 = exclude_diag(cross_acc[i, :, :, 4])
        r1_vals.append(r1)
        r5_vals.append(r5)
        table.add_row(name, f'{r1 * 100:.2f}', f'{r5 * 100:.2f}')
    mean_r1 = sum(r1_vals) / len(r1_vals)
    mean_r5 = sum(r5_vals) / len(r5_vals)
    table.add_section()
    table.add_row("[bold]Mean[/]", f'[bold]{mean_r1 * 100:.2f}[/]', f'[bold]{mean_r5 * 100:.2f}[/]')

    console.print('\n')
    console.print(table)

    # per-angle table: Rank-1 and Rank-5
    for rank, rank_idx in [('Rank-1', 0), ('Rank-5', 4)]:
        angle_table = Table(title=f"[bold]CASIA-B {rank} Per Angle (Exclude Identical-View)[/]")
        angle_table.add_column("Condition", justify="center")
        for v in view_list:
            angle_table.add_column(str(v), justify="center")
        angle_table.add_column("Mean", justify="center", style="bold")
        for i, name in enumerate(group_names):
            per_angle = exclude_diag(cross_acc[i, :, :, rank_idx], reduce=False)
            mean_val = per_angle.mean()
            angle_table.add_row(name, *[f'{x * 100:.2f}' for x in per_angle], f'{mean_val * 100:.2f}')
        console.print('\n')
        console.print(angle_table)

    # wandb: only per-condition R1
    visual['gait'] = {
        name.lower(): exclude_diag(cross_acc[i, :, :, 0]).item()
        for i, name in enumerate(group_names)
    }

    # detail: full cross_acc for file output
    detail = {}
    for rank_tag, rank_idx in [('r1', 0), ('r5', 4)]:
        for i, name in enumerate(group_names):
            key = name.lower()
            detail[f'{key}_{rank_tag}'] = exclude_diag(cross_acc[i, :, :, rank_idx]).item()
            per_angle = exclude_diag(cross_acc[i, :, :, rank_idx], reduce=False)
            for v_idx, v in enumerate(view_list):
                detail[f'{key}_{rank_tag}_{v}'] = per_angle[v_idx].item()

    return {'visual': visual, 'detail': detail, 'score': visual['gait']['nm']}
