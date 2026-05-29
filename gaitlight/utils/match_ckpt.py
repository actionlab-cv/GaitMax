import logging
from typing import Any

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


def match_ckpt(model: pl.LightningModule, checkpoint: dict[str, Any]) -> dict[str, Any]:
    # state dict
    curr_state = model.state_dict()
    ckpt_state = checkpoint['state_dict']

    # filter
    new_state = {
        k: v for k, v in ckpt_state.items()
        if k in curr_state and v.shape == curr_state[k].shape
    }
    unmatch = [k for k in ckpt_state.keys() if k not in new_state.keys()]
    if len(unmatch) > 0:
        logger.warning(f'ignore unmatched keys in checkpoint: {unmatch}')
    checkpoint['state_dict'] = new_state

    return checkpoint
