"""
GaitEval: Lightning callback for gait recognition evaluation.

Collects EmbedBatch outputs from validation, groups by dataset prefix,
and dispatches to dataset-specific evaluation functions.

Supported datasets (detected by subject prefix):
    - ccpg_*    → ccpg_evaluate
    - ccgrm_*   → ccgr_mini_evaluate
    - casiab_*  → casiab_evaluate
    - sus_*     → sustech1k_evaluate
"""

import json
import logging
from collections import defaultdict
from functools import reduce
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from gaitlight.types import EmbedBatch

logger = logging.getLogger(__name__)

# prefix → evaluate function name
_DATASET_PREFIX = {
    'ccpg': 'ccpg',
    'ccgrm': 'ccgrm',
    'casiab': 'casiab',
    'sus': 'sustech1k',
}


def _detect_dataset(subject: str) -> str | None:
    """Detect dataset from subject prefix (e.g., 'ccpg_001' → 'ccpg')."""
    for prefix in _DATASET_PREFIX:
        if subject.startswith(prefix):
            return prefix
    return None


class GaitEval(pl.Callback):
    def __init__(self):
        super().__init__()
        self.v_cache: list[EmbedBatch] = []

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT, batch, batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, EmbedBatch):
            return
        self.v_cache.append(outputs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.v_cache:
            logger.info('GaitEval skipped: no EmbedBatch outputs collected')
            return

        # gather across all ranks
        all_v_cache = [None for _ in range(trainer.world_size)]
        torch.distributed.all_gather_object(all_v_cache, self.v_cache)
        torch.distributed.barrier()

        # only rank 0 runs evaluation
        if trainer.global_rank != 0:
            self.v_cache.clear()
            return

        # merge and deduplicate
        all_v_cache = reduce(lambda x, y: x + y, all_v_cache)
        seen = set()
        embeds, labels, metas = [], [], []
        for batch in all_v_cache:
            for meta, embed, label in zip(batch.meta, batch.embed, batch.label):
                key = f'{meta.subject}-{meta.caption}-{meta.view}'
                if key not in seen:
                    seen.add(key)
                    embeds.append(embed.cpu())
                    labels.append(label.cpu())
                    metas.append(meta)

        embed = torch.stack(embeds, dim=0)
        label = torch.stack(labels, dim=0)
        logger.info(f'GaitEval: collected {len(metas)} unique sequences')

        # group by dataset
        groups: dict[str, dict] = defaultdict(lambda: {'embed': [], 'label': [], 'meta': []})
        for i, meta in enumerate(metas):
            ds = _detect_dataset(meta.subject)
            if ds is None:
                logger.debug(f'unknown dataset for subject: {meta.subject}')
                continue
            groups[ds]['embed'].append(embed[i])
            groups[ds]['label'].append(label[i])
            groups[ds]['meta'].append(meta)

        # dispatch per dataset
        all_details = {}
        for ds, data in groups.items():
            n_seq = len(data['meta'])
            n_ids = len(set(m.subject for m in data['meta']))
            data['embed'] = torch.stack(data['embed'], dim=0)
            data['label'] = torch.stack(data['label'], dim=0)
            logger.info(f'GaitEval: evaluating {ds} ({n_seq} sequences, {n_ids} identities)')

            # skip if too few identities or sequences for meaningful eval
            if n_ids < 2:
                logger.warning(f'GaitEval: {ds} has only {n_ids} identity, skipping')
                continue
            if n_seq < 4:
                logger.warning(f'GaitEval: {ds} has only {n_seq} sequences, skipping')
                continue

            try:
                result = _dispatch_evaluate(ds, data)
            except NotImplementedError:
                logger.warning(f'GaitEval: no evaluate function for {ds}, skipping')
                continue
            except Exception as e:
                logger.error(f'GaitEval: {ds} evaluation failed: {e}')
                continue

            # log score to wandb (minimal)
            if 'score' in result:
                pl_module.log(f'val/{ds}/score', result['score'], sync_dist=False, rank_zero_only=True)

            # log summary metrics to wandb (per-condition R1 only)
            visual = result.get('visual', {})
            for group_name, metrics in visual.items():
                if isinstance(metrics, dict):
                    pl_module.log_dict(
                        {f'val/{ds}/{group_name}/{k}': v for k, v in metrics.items()},
                        sync_dist=False, rank_zero_only=True,
                    )

            # collect detail for file output
            if 'detail' in result:
                all_details[ds] = result['detail']
                all_details[ds]['score'] = result.get('score', None)

        # save detail JSON next to checkpoints
        if all_details:
            self._save_detail(trainer, all_details)

        # clear
        self.v_cache.clear()

    @staticmethod
    def _save_detail(trainer: pl.Trainer, details: dict):
        """Save detailed eval results as JSON next to checkpoints."""
        try:
            ckpt_dir = None
            for cb in trainer.checkpoint_callbacks:
                if hasattr(cb, 'dirpath') and cb.dirpath:
                    ckpt_dir = Path(cb.dirpath)
                    break
            if ckpt_dir is None:
                return

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            step = trainer.global_step
            path = ckpt_dir / f'eval-{step:05d}.json'
            path.write_text(json.dumps(details, indent=2))
            logger.info(f'GaitEval: detail saved to {path}')
        except Exception as e:
            logger.warning(f'GaitEval: failed to save detail: {e}')


def _dispatch_evaluate(dataset: str, data: dict) -> dict:
    """Dispatch to dataset-specific evaluation function."""
    if dataset == 'ccpg':
        from gaitlight.data.evaluate.ccpg_evaluate import ccpg_evaluate
        return ccpg_evaluate(data)
    elif dataset == 'ccgrm':
        from gaitlight.data.evaluate.ccgr_evaluate import ccgr_mini_evaluate
        return ccgr_mini_evaluate(data)
    elif dataset == 'casiab':
        from gaitlight.data.evaluate.casiab_evaluate import casiab_evaluate
        return casiab_evaluate(data)
    elif dataset == 'sus':
        from gaitlight.data.evaluate.sustech1k_evaluate import sustech1k_evaluate
        return sustech1k_evaluate(data)
    else:
        raise NotImplementedError(f'No evaluate function for dataset: {dataset}')
