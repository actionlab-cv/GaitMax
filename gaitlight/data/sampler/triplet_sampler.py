import logging
import random
from typing import Literal

import torch
import torch.distributed
from torch.utils.data import Sampler

from gaitlight.data.dataset.ug_set import UGSet

logger = logging.getLogger(__name__)


class TripletSampler(Sampler):
    def __init__(self, dataset: UGSet, identity_size: int, sequence_size: int, shuffle: bool = True, max_steps: int = 10 ** 18):
        super().__init__()

        # dataset
        self.dataset = dataset

        # batch size & shuffle
        self.identity_size = identity_size
        self.sequence_size = sequence_size
        self.shuffle = shuffle

        # distribution info
        is_dist = torch.distributed.is_initialized()
        self.world_s = torch.distributed.get_world_size() if is_dist else 1
        self.rank = torch.distributed.get_rank() if is_dist else 0

        # batch size
        batch_size = self.identity_size * self.sequence_size
        if batch_size % self.world_s != 0:
            raise ValueError(f"batch size {batch_size} is not divisible by world size {self.world_s}")
        self.batch_size = batch_size // self.world_s

        # max steps
        self.max_steps = max_steps * self.batch_size

    def __iter__(self):
        while True:
            l_idx = []

            # select subject
            l_sub_used = self._sync_idx(len(self.dataset.l_sub), self.identity_size)

            # select sequence
            for s_idx in l_sub_used:
                sub, l_meta = self.dataset.l_sub[s_idx]  # (subject, seq_idx)
                l_sel = self._sync_idx(len(l_meta), self.sequence_size)
                l_meta = [l_meta[i] for i in l_sel]
                logger.debug(f'selected sequences: {l_meta} from subject {sub} [idx: {s_idx}]')
                l_idx += l_meta
            l_idx = torch.tensor(l_idx, dtype=torch.int)

            # shuffle l_idx if needed
            if self.shuffle:
                l_idx = l_idx[self._sync_idx(len(l_idx), len(l_idx))]

            # yield
            l_idx = l_idx[self.rank::self.world_s]
            for _idx in l_idx.tolist():
                yield _idx

    def __len__(self):
        return self.max_steps

    @staticmethod
    def _sync_idx(m: int, n: int, mode: Literal['random', 'perm'] = 'perm') -> list:
        """
        Gain synchronized indices across all processes. Select n indices from m indices.
        :param m: total number of indices
        :param n: number of indices to select
        :param mode: selection mode, random or randperm
        :return: list of selected indices
        """

        # select index
        if mode == 'random' or m < n:
            idx = torch.tensor(random.choices(range(m), k=n), dtype=torch.int)
        else:
            idx = torch.randperm(m, dtype=torch.int)[:n]

        # broadcast to all (skip if not distributed)
        if torch.distributed.is_initialized():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            idx = idx.to(device)
            torch.distributed.broadcast(idx, src=0)
            idx = idx.cpu()

        return idx.tolist()
