import hashlib
import json
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch
import torch.distributed
from rich.progress import Progress
from torch.utils.data import Dataset

from gaitlight.data.dataset.loader import load_caption, load_mp4, load_pt
from gaitlight.types import SequenceData, SequenceItem, SequenceMeta

logger = logging.getLogger(__name__)

loader_func = {
    'frame.mp4': load_mp4,
    'mask.mp4': lambda p: load_mp4(p)[:, 0] > 0,
    'pattern.mp4': lambda p: (load_mp4(p)[:, 0].float() / 9).round().to(torch.uint8),
    'pose.pt': load_pt,
}


class UGSet(Dataset):
    """
    Unified Gait Dataset.
    """

    def __init__(self, root: Path, partition: Path, using: list[str], mode: str = Literal['train', 'val', 'pred'], caption: bool = False):
        super().__init__()

        self.root = root / 'sequences'
        self.partition = partition
        self.using = using
        self.mode = mode
        self.caption = caption  # also load co-located caption.pt -> SequenceItem.cpt (for CDLoss)

        # load subject list from partition
        part = json.load(self.partition.open('rb'))
        l_t_sub = part['TRAIN']
        l_v_sub = part['TEST']
        match self.mode:
            case 'train':
                self.l_sub = l_t_sub
            case 'val':
                self.l_sub = l_v_sub
            case 'pred':
                logger.warning(f'pred mode: using all subjects from train and val partitions')
                self.l_sub = l_t_sub + l_v_sub
            case _:
                raise ValueError(f'invalid mode: {self.mode}')

        # find missing subjects
        a_sub = {x.name for x in self.root.iterdir() if x.is_dir()}
        missing = set(self.l_sub) - a_sub
        if missing:
            logger.fatal(f'{self.mode}: missing {len(missing)} subjects: {missing}')
        logger.info(f'{self.mode}: {len(self.l_sub)} subjects')

        # cache folder
        self.p_cache = root / 'cache'
        self.p_cache.mkdir(parents=True, exist_ok=True)

        # build meta & subject lists
        self.l_meta: list[SequenceMeta] = self._build_meta_list()
        self.l_sub: list[tuple[str, list[int]]] = self._build_subject_list()
        self.m_sub: dict[str, int] = {sub: idx for idx, (sub, _) in enumerate(self.l_sub)}

    def __len__(self) -> int:
        return len(self.l_meta)

    def __getitem__(self, idx: int) -> SequenceItem:
        meta = self.l_meta[idx]
        seq = SequenceData(**{
            Path(f).stem: loader_func[f](meta.path / f)
            for f in self.using
        })
        cpt = load_caption(meta.path / 'caption.pt') if self.caption else None
        return SequenceItem(seq=seq, meta=meta, label=self.m_sub[meta.subject], cpt=cpt)

    @property
    def signature(self) -> str:
        c = {
            'root': str(self.root),
            'root_st_mtime': self.root.stat().st_mtime,
            'partition': str(self.partition),
            'partition_st_mtime': self.partition.stat().st_mtime,
            'using': self.using,
            'mode': self.mode,
            'caption': self.caption,
        }
        return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()

    def _touch(self, path: Path) -> bool:
        return all((path / f).exists() for f in self.using) and (
            not self.caption or (path / 'caption.pt').exists()
        )

    @staticmethod
    def _is_rank0() -> bool:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    @staticmethod
    def _barrier():
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_meta_list(self) -> list[SequenceMeta]:
        f_cache = self.p_cache / f'meta_{self.signature}.pkl'

        # rank 0 builds cache if missing, other ranks wait
        if self._is_rank0():
            if not f_cache.exists():
                logger.info('no cache found, building meta list')
                l_meta = []
                with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() + 4)) as executor:
                    with Progress() as pbar:
                        _id = pbar.add_task('build meta list', total=len(self.l_sub))
                        pcs = [executor.submit(self._build_meta_subject, sub) for sub in sorted(self.l_sub)]
                        for pc in pcs:
                            l_meta.extend(pc.result())
                            pbar.update(_id, advance=1)
                logger.info(f'built meta list ({len(l_meta)} sequences)')
                pickle.dump(l_meta, f_cache.open('wb'))
        self._barrier()

        l_meta = pickle.load(f_cache.open('rb'))
        logger.info(f'loaded meta list from cache ({len(l_meta)} sequences)')
        return l_meta

    def _build_meta_subject(self, sub: str) -> list[SequenceMeta]:
        l_meta = []
        p_sub = self.root / sub
        for p_cap in sorted(x for x in p_sub.iterdir() if x.is_dir()):
            for p_view in sorted(x for x in p_cap.iterdir() if x.is_dir()):
                if not self._touch(p_view):
                    logger.warning(f'{sub}: missing modality files in {p_view}')
                    continue
                l_meta.append(SequenceMeta(
                    subject=sub, caption=p_cap.name,
                    view=p_view.name, path=p_view,
                ))
        return l_meta

    def _build_subject_list(self) -> list[tuple[str, list[int]]]:
        f_cache = self.p_cache / f'subject_{self.signature}.pkl'

        # rank 0 builds cache if missing, other ranks wait
        if self._is_rank0():
            if not f_cache.exists():
                logger.info('no cache found, building subject list')
                grouped: dict[str, list[int]] = defaultdict(list)
                for idx, meta in enumerate(self.l_meta):
                    grouped[meta.subject].append(idx)
                l_sub = sorted(grouped.items())
                logger.info(f'built subject list ({len(l_sub)} subjects)')
                pickle.dump(l_sub, f_cache.open('wb'))
        self._barrier()

        l_sub = pickle.load(f_cache.open('rb'))
        logger.info(f'loaded subject list from cache ({len(l_sub)} subjects)')
        return l_sub
