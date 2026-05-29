import logging
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from gaitlight.data.collate.sequence_collate import SequenceCollate
from gaitlight.data.dataset.ug_set import UGSet
from gaitlight.data.sampler.inference_sampler import InferenceSampler
from gaitlight.data.sampler.triplet_sampler import TripletSampler

logger = logging.getLogger(__name__)


class UGModule(pl.LightningDataModule):
    """
    Unified Gait Data Module.
    """

    @classmethod
    def from_dict(cls, d: dict) -> 'UGModule':
        c = d['common']
        t = d['loader']['train']
        v = d['loader']['val']
        return cls(
            root=Path(c['root']), partition=Path(c['partition']), using=c['using'], num_workers=c.get('num_workers', 0),
            train_identity_size=t['identity_size'], train_sequence_size=t['sequence_size'], train_max_steps=t['max_steps'],
            train_collate_sample=t['sample'], train_collate_frame_num=t['frame_num'], train_collate_frame_buf=t.get('frame_buf', 0),
            val_sequence_size=v['sequence_size'], val_collate_sample=v['sample'], val_collate_frame_num=v['frame_num'],
            val_collate_frame_buf=v.get('frame_buf', 0),
        )

    def __init__(
            self,
            root: Path, partition: Path, using: list[str], num_workers: int = 0,
            train_identity_size: int | None = None, train_sequence_size: int | None = None, train_max_steps: int | None = None,
            train_collate_sample: list[str] | None = None, train_collate_frame_num: int | list[int] | None = None, train_collate_frame_buf: int | None = None,
            val_sequence_size: int | None = None,
            val_collate_sample: list[str] | None = None, val_collate_frame_num: int | list[int] | None = None, val_collate_frame_buf: int | None = None,
    ):
        super().__init__()

        # meta
        self.root = root
        self.partition = partition
        self.using = using
        self.num_workers = num_workers

        # train set
        self.t_set = None
        self.t_identity_size = train_identity_size
        self.t_sequence_size = train_sequence_size
        self.t_max_steps = train_max_steps
        self.t_collate_sample = train_collate_sample
        self.t_collate_frame_num = train_collate_frame_num
        self.t_collate_frame_buf = train_collate_frame_buf

        # val set
        self.v_set = None
        self.v_sequence_size = val_sequence_size
        self.v_collate_sample = val_collate_sample
        self.v_collate_frame_num = val_collate_frame_num
        self.v_collate_frame_buf = val_collate_frame_buf

        # pred set
        self.p_set = None

    def setup(self, stage: str) -> None:
        if stage in ['fit', None]:
            self.t_set = UGSet(self.root, self.partition, self.using, mode='train')
            logger.info('train set loaded')
        if stage in ['fit', 'validate', 'test', None]:
            self.v_set = UGSet(self.root, self.partition, self.using, mode='val')
            logger.info('val set loaded')
        if stage in ['predict', None]:
            self.p_set = UGSet(self.root, self.partition, self.using, mode='pred')
            logger.info('pred set loaded')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = TripletSampler(
            self.t_set,
            identity_size=self.t_identity_size,
            sequence_size=self.t_sequence_size,
            shuffle=True,
            max_steps=self.t_max_steps,
        )
        logger.info(f'train sampler initialized, batch size (each rank): {sampler.batch_size}')
        return DataLoader(
            self.t_set, batch_size=sampler.batch_size, sampler=sampler,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SequenceCollate(self.t_collate_sample, self.t_collate_frame_num, self.t_collate_frame_buf),
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = InferenceSampler(self.v_set, sequence_size=self.v_sequence_size)
        logger.info(f'val sampler initialized, batch size (each rank): {sampler.batch_size}')
        return DataLoader(
            self.v_set, batch_size=sampler.batch_size, sampler=sampler,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SequenceCollate(self.v_collate_sample, self.v_collate_frame_num, self.v_collate_frame_buf),
            persistent_workers=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        sampler = InferenceSampler(self.p_set, sequence_size=self.v_sequence_size)
        logger.info(f'pred sampler initialized, batch size (each rank): {sampler.batch_size}')
        return DataLoader(
            self.p_set, batch_size=sampler.batch_size, sampler=sampler,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SequenceCollate(self.v_collate_sample, self.v_collate_frame_num, self.v_collate_frame_buf),
            persistent_workers=True,
        )
