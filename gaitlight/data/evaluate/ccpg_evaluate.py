"""
CCPG dataset evaluation.

Protocol loaded from data/partition/protocol/ccpg.json.
4 probe-gallery groups: CL, UP, DN, BG.
Metrics: Rank-1, mAP, mINP (exclude identical-view).
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy
import torch
from rich.console import Console
from rich.table import Table

from gaitlight.metrics.cmc import cmc
from gaitlight.metrics.distance import euc_dist
from gaitlight.metrics.exclude_diag import exclude_diag

logger = logging.getLogger(__name__)

_PROTOCOL_PATH = Path(__file__).parents[1] / 'partition' / 'protocol' / 'ccpg.json'


@torch.inference_mode()
def ccpg_evaluate(u_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate CCPG dataset.

    :param u_data: dict with 'embed' [n, c, p], 'label' [n], 'meta' [list of SequenceMeta]
    :return: dict with 'visual' and 'score'
    """
    console = Console()
    visual = {}

    # load protocol
    protocol = json.loads(_PROTOCOL_PATH.read_text())
    probe_groups = protocol['probe']
    gallery_groups = protocol['gallery']
    group_names = protocol['names']

    # data
    embed: torch.Tensor = u_data['embed']
    label: torch.Tensor = u_data['label']
    caption = [m.caption for m in u_data['meta']]
    view = torch.tensor([int(m.view.split('_')[0]) for m in u_data['meta']], device=embed.device)

    # --- person re-id (exclude identical-view) ---
    cmc_l, ap_l, inp_l = [], [], []

    d_table = Table(title="[bold]Probe - Gallery Division[/]")
    d_table.add_column("Group", justify="center")
    d_table.add_column("Probe", justify="center")
    d_table.add_column("# Probe", justify="center")
    d_table.add_column("Gallery", justify="center")
    d_table.add_column("# Gallery", justify="center")

    for idx, name in enumerate(group_names):
        probe_cap = probe_groups[idx]
        gallery_cap = gallery_groups[idx]

        g_mask = numpy.isin(caption, gallery_cap)
        g_embed, g_label, g_view = embed[g_mask], label[g_mask], view[g_mask]

        p_mask = numpy.isin(caption, probe_cap)
        p_embed, p_label, p_view = embed[p_mask], label[p_mask], view[p_mask]

        d_table.add_row(name, str(probe_cap), str(len(p_embed)), str(gallery_cap), str(len(g_embed)))

        dist = euc_dist(p_embed, g_embed)
        rst = cmc(dist, p_label, g_label, p_view, g_view)
        cmc_l.append(rst['cmc'][0])
        ap_l.append(rst['mAP'])
        inp_l.append(rst['mINP'])

    console.print('\n')
    console.print(d_table)

    # re-id table
    p_table = Table(title="[bold]Person Re-ID (Exclude Identical-View)[/]")
    p_table.add_column("Metric", justify="center")
    for name in group_names:
        p_table.add_column(name, justify="center")
    p_table.add_row("Rank-1", *[f"{v * 100:.2f}" for v in cmc_l])
    p_table.add_row("mAP", *[f"{v * 100:.2f}" for v in ap_l])
    p_table.add_row("mINP", *[f"{v * 100:.2f}" for v in inp_l])
    console.print('\n')
    console.print(p_table)

    visual['reid'] = {n.lower(): v.item() if isinstance(v, torch.Tensor) else v for n, v in zip(group_names, cmc_l)}

    # --- gait recognition (cross-view) ---
    max_rank = 5
    view_np = view.cpu().numpy()
    _view = sorted(set(view_np))
    cross_acc = torch.full((len(group_names), len(_view), len(_view), max_rank), -1.0)

    for p_idx, name in enumerate(group_names):
        probe_cap = probe_groups[p_idx]
        gallery_cap = gallery_groups[p_idx]

        for pv_idx, pv in enumerate(_view):
            for gv_idx, gv in enumerate(_view):
                g_mask = numpy.isin(caption, gallery_cap) & numpy.isin(view_np, [gv])
                g_embed, g_label = embed[g_mask], label[g_mask]

                p_mask = numpy.isin(caption, probe_cap) & numpy.isin(view_np, [pv])
                p_embed, p_label = embed[p_mask], label[p_mask]

                if len(p_embed) == 0 or len(g_embed) == 0:
                    continue

                dist = euc_dist(p_embed, g_embed)
                indices = dist.argsort(dim=1)
                matches = p_label.reshape(-1, 1) == g_label[indices[:, :max_rank]]
                acc = torch.sum(torch.cumsum(matches, dim=1) > 0, dim=0) / dist.shape[0]
                cross_acc[p_idx, pv_idx, gv_idx] = acc

    # include identical-view
    g1_table = Table(title="[bold]Gait Recognition Rank-1 (Include Identical-View)[/]")
    g1_table.add_column("Rank", justify="center")
    for name in group_names:
        g1_table.add_column(name, justify="center")
    for r in [0, 4]:
        g1_table.add_row(str(r + 1), *[f'{torch.mean(cross_acc[i, :, :, r]) * 100:.2f}' for i in range(len(group_names))])
    console.print('\n')
    console.print(g1_table)

    # wandb: only per-condition R1 (include identical-view)
    visual['gait'] = {
        n.lower(): torch.mean(cross_acc[i, :, :, 0]).item()
        for i, n in enumerate(group_names)
    }

    # detail: R1/R5 both with and without identical-view
    detail = {}
    for i, n in enumerate(group_names):
        key = n.lower()
        detail[f'{key}_r1'] = torch.mean(cross_acc[i, :, :, 0]).item()
        detail[f'{key}_r5'] = torch.mean(cross_acc[i, :, :, 4]).item()
        detail[f'{key}_r1_exd'] = exclude_diag(cross_acc[i, :, :, 0]).item()
        detail[f'{key}_r5_exd'] = exclude_diag(cross_acc[i, :, :, 4]).item()

    # exclude identical-view
    g2_table = Table(title="[bold]Gait Recognition Rank-1 (Exclude Identical-View)[/]")
    g2_table.add_column("Rank", justify="center")
    for name in group_names:
        g2_table.add_column(name, justify="center")
    for r in [0, 4]:
        g2_table.add_row(str(r + 1), *[f'{exclude_diag(cross_acc[i, :, :, r]) * 100:.2f}' for i in range(len(group_names))])
    console.print('\n')
    console.print(g2_table)

    return {'visual': visual, 'detail': detail, 'score': visual['gait']['cl']}
