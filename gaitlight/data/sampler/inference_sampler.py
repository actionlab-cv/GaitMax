import math

import torch
from torch.utils.data import Sampler

from gaitlight.data.dataset.ug_set import UGSet


class InferenceSampler(Sampler):
    def __init__(self, dataset: UGSet, sequence_size: int):
        super().__init__()

        # dataset
        self.dataset = dataset

        # batch size
        self.sequence_size = sequence_size

        # distribution info
        is_dist = torch.distributed.is_initialized()
        self.world_s = torch.distributed.get_world_size() if is_dist else 1
        self.rank = torch.distributed.get_rank() if is_dist else 0

        # sanity check
        if sequence_size % self.world_s != 0:
            raise ValueError(f"sequence size {sequence_size} is not divisible by world size {self.world_s}")

        # sample nums
        self.num_samples = math.ceil(len(self.dataset.l_meta) / sequence_size)

    def __iter__(self):
        # total number of sequences
        tot = len(self.dataset.l_meta)
        l_idx = list(range(tot))

        # expand l_idx if needed
        req = self.num_samples * self.sequence_size
        l_idx = l_idx + l_idx[:req - tot]

        # broadcast index
        l_idx = l_idx[self.rank::self.world_s]
        assert len(l_idx) == self.num_samples * self.batch_size

        return iter(l_idx)

    def __len__(self) -> int:
        return self.num_samples * self.batch_size

    @property
    def batch_size(self) -> int:
        return self.sequence_size // self.world_s
