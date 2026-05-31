import dataclasses
import random
from dataclasses import dataclass

import torch

from gaitlight.types import (
    InputBatch, SampleInfo, SequenceBatch, SequenceData, SequenceItem,
)


@dataclass
class SequenceCollateConfig:
    sample: list[str]  # [mode, order]: e.g. ['natural', 'ordered']
    frame_num: int | list[int]  # int for fixed/natural/interval, [min, max] for unfixed
    frame_buf: int = 0  # buffer for natural mode


class SequenceCollate:
    def __init__(self, sample: list[str], frame_num: int | list[int], frame_buf: int = 0):
        self.sample = sample
        self.frame_num = frame_num
        self.frame_buf = frame_buf

        assert self.sample[0] in ['fixed', 'unfixed', 'natural', 'interval']
        assert self.sample[1] in ['ordered', 'unordered']
        if self.sample[0] == 'unfixed':
            assert isinstance(self.frame_num, list), 'frame_num must be [min, max] for unfixed mode'

    def __call__(self, items: list[SequenceItem]) -> InputBatch:
        # sample frames
        sampled_items, sample_infos = zip(*[self._sample_item(x) for x in items])

        # pad to max length
        max_t = max(s.num for s in sample_infos)
        padded_items = [self._pad_item(item, max_t) for item in sampled_items]
        padded_infos = [self._pad_info(info, max_t) for info in sample_infos]

        # stack into batch
        seq_batch = SequenceBatch(**{
            f.name: self._stack_field(padded_items, f.name)
            for f in dataclasses.fields(SequenceData)
        })

        # per-sequence caption embeddings (not frame-sampled); [b, l, d] or None
        cpt = (torch.stack([item.cpt for item in padded_items])
               if padded_items and padded_items[0].cpt is not None else None)

        return InputBatch(
            seq=seq_batch,
            meta=[item.meta for item in padded_items],
            label=torch.tensor([item.label for item in padded_items]),
            sample=padded_infos,
            cpt=cpt,
        )

    def _sample_item(self, item: SequenceItem) -> tuple[SequenceItem, SampleInfo]:
        # get temporal length from first non-None field
        t = next(
            getattr(item.seq, f.name).shape[0]
            for f in dataclasses.fields(item.seq)
            if getattr(item.seq, f.name) is not None
        )

        frame_num, l_idx = self._compute_idx(t)

        match self.sample[1]:
            case 'ordered':
                l_idx.sort()
            case 'unordered':
                random.shuffle(l_idx)

        new_seq = SequenceData(**{
            f.name: getattr(item.seq, f.name)[l_idx] if getattr(item.seq, f.name) is not None else None
            for f in dataclasses.fields(item.seq)
        })

        info = SampleInfo(type=self.sample, num=frame_num, index=l_idx)
        return dataclasses.replace(item, seq=new_seq), info

    def _compute_idx(self, t: int) -> tuple[int, list]:
        match self.sample[0]:
            case 'fixed':
                return self.frame_num, self._random_idx(t, self.frame_num)
            case 'unfixed':
                n = random.choice(range(self.frame_num[0], self.frame_num[1] + 1))
                return n, self._random_idx(t, n)
            case 'natural':
                return self.frame_num, self._continuous_idx(t, self.frame_num, self.frame_buf)
            case 'interval':
                return self.frame_num, self._interval_idx(t, self.frame_num)
            case _:
                raise NotImplementedError(f'unsupported sample mode: {self.sample[0]}')

    @staticmethod
    def _pad_item(item: SequenceItem, t: int) -> SequenceItem:
        def pad(v):
            if v is None or v.shape[0] >= t:
                return v
            return torch.cat([v, torch.zeros(t - v.shape[0], *v.shape[1:], dtype=v.dtype)], dim=0)

        new_seq = SequenceData(**{
            f.name: pad(getattr(item.seq, f.name))
            for f in dataclasses.fields(item.seq)
        })
        return dataclasses.replace(item, seq=new_seq)

    @staticmethod
    def _pad_info(info: SampleInfo, t: int) -> SampleInfo:
        if len(info.index) >= t:
            return info
        return dataclasses.replace(info, index=info.index + [None] * (t - len(info.index)))

    @staticmethod
    def _stack_field(items: list[SequenceItem], name: str):
        tensors = [getattr(item.seq, name) for item in items]
        if all(v is None for v in tensors):
            return None
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _random_idx(m: int, n: int) -> list:
        l_idx = torch.arange(0, m)
        l_idx = l_idx.repeat(n // m + 1) if n > m else l_idx
        return l_idx[torch.randperm(len(l_idx))][:n].tolist()

    def _continuous_idx(self, m: int, n: int, buf: int) -> list:
        if n >= m:
            return list(range(m))
        s = n + buf
        if s >= m:
            return torch.linspace(0, m - 1, steps=n, dtype=torch.int).tolist()
        st = random.randint(0, m - s)
        window = list(range(st, st + s))
        return [window[i] for i in self._random_idx(s, n)]

    @staticmethod
    def _interval_idx(m: int, n: int) -> list:
        l_idx = torch.arange(0, m)
        l_idx = l_idx.repeat(n // m + 1) if n > m else l_idx
        l_idx, _ = l_idx.sort()
        return l_idx[torch.linspace(0, len(l_idx) - 1, n, dtype=torch.int)].tolist()
