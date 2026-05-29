"""
CCGR Mini dataset evaluation.

Protocol loaded from data/partition/protocol/ccgrm.json.
Gallery defined by slug list, rest are probes.
Metrics: Rank-1, Rank-5, mAP, mINP.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from gaitlight.metrics.cmc import cmc
from gaitlight.metrics.distance import euc_dist

logger = logging.getLogger(__name__)

_PROTOCOL_PATH = Path(__file__).parents[1] / 'partition' / 'protocol' / 'ccgrm.json'


@torch.inference_mode()
def ccgr_mini_evaluate(u_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate CCGR mini dataset.

    :param u_data: dict with 'embed' [n, c, p], 'label' [n], 'meta' [list of SequenceMeta]
    :return: dict with 'visual' and 'score'
    """
    console = Console()

    embed: torch.Tensor = u_data['embed']
    label: torch.Tensor = u_data['label']

    # build slug from meta (strip dataset prefix: ccgrm_601 → 601)
    def _strip_prefix(subject: str) -> str:
        return subject.split('_', 1)[-1] if '_' in subject else subject

    slug = [f'{_strip_prefix(m.subject)}-{m.caption}-{m.view}' for m in u_data['meta']]

    # load gallery list from protocol
    protocol = json.loads(_PROTOCOL_PATH.read_text())
    gallery = set(protocol['gallery_slugs'])

    # split
    import numpy
    g_mask = numpy.array([x in gallery for x in slug])
    g_embed, g_label = embed[g_mask], label[g_mask]
    p_embed, p_label = embed[~g_mask], label[~g_mask]

    logger.info(f'CCGR mini: probe={p_embed.shape[0]}, gallery={g_embed.shape[0]}')

    # evaluate
    dist = euc_dist(p_embed, g_embed)
    rst = cmc(dist, p_label, g_label)

    # table
    g_table = Table(title="[bold]CCGR Mini Gait Recognition[/]")
    g_table.add_column("R1", justify="center")
    g_table.add_column("R5", justify="center")
    g_table.add_column("mAP", justify="center")
    g_table.add_column("mINP", justify="center")
    g_table.add_row(
        f"{rst['cmc'][0] * 100:.2f}",
        f"{rst['cmc'][4] * 100:.2f}",
        f"{rst['mAP'] * 100:.2f}",
        f"{rst['mINP'] * 100:.2f}",
    )
    console.print('\n')
    console.print(g_table)

    detail = {
        'r1': rst['cmc'][0].item(),
        'r5': rst['cmc'][4].item(),
        'mAP': rst['mAP'].item(),
        'mINP': rst['mINP'].item(),
    }
    visual = {'gait': {'r1': detail['r1']}}
    return {'visual': visual, 'detail': detail, 'score': detail['r1']}
