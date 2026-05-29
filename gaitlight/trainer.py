import argparse
import os
import tomllib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='Grad strides do not match bucket view strides')
warnings.filterwarnings('ignore', message='The AccumulateGrad node')
warnings.filterwarnings('ignore', message='`isinstance(treespec, LeafSpec)`')

import lightning.pytorch as pl
import torch
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from gaitlight import models
from gaitlight.data.dataset.ug_module import UGModule
from gaitlight.data.evaluate.gait_eval import GaitEval
from gaitlight.utils.init_logger import init_logger
from gaitlight.utils.match_ckpt import match_ckpt

os.environ['MASTER_PORT'] = '29520'
os.environ['NCCL_IB_DISABLE'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Lightning")

    # config file
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')

    # phase
    parser.add_argument('-p', '--phase', default='train', choices=['train', 'val', 'pred'], help='Phase of the model')

    # resume & load
    parser.add_argument('-r', '--resume', help='Resume from checkpoint')
    parser.add_argument('-l', '--load', help='Load from checkpoint')

    # version label
    parser.add_argument('-v', '--version', type=str, default='default', help='Version label for ckpt')

    # offline
    parser.add_argument('--offline', action='store_true', help='Use offline mode for wandb')

    # save strategy: 'default' for periodic only, or a metric name (e.g. 'val/iou') to save best
    parser.add_argument('-s', '--save', default='default', help='Save strategy: default or metric name (e.g. val/iou)')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # load toml config
    _c = tomllib.load(Path(args.config).open('rb'))
    config = OmegaConf.create(_c)

    # init logger
    logger = init_logger(config.common.log_level)
    logger.info(f"config loaded from {args.config}")

    # matmul precision
    torch.set_float32_matmul_precision("medium")

    # fix seed
    pl.seed_everything(config.common.seed, verbose=False)
    logger.info(f'seed fixed: {config.common.seed}')

    # connect wandb
    _offline = config.lightning.wandb.offline if args.phase in ['train'] else True
    if args.offline:
        _offline = True
    wandb_l = WandbLogger(
        offline=_offline,
        project=config.lightning.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        log_model=False,
    )
    logger.info(f'wandb connected')

    # find model
    _model = getattr(models, config.pipeline.main.type)
    logger.info(f'model found: {_model.__name__}')

    # init model
    if args.resume:
        "resume from checkpoint"
        model = _model.load_from_checkpoint(args.resume, map_location='cpu', config=config.pipeline)
        logger.info(f'model resumed from {args.resume}')
    elif args.load:
        "load from checkpoint"
        model: pl.LightningModule = _model(config.pipeline)
        _ckpt = torch.load(args.load, map_location='cpu', weights_only=False)
        ckpt = match_ckpt(model, _ckpt)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info(f'model loaded from {args.load}')
    else:
        "init from config"
        model: pl.LightningModule = _model(config.pipeline)
        logger.info(f'model initialized')

    # init data module
    data_module = UGModule.from_dict(OmegaConf.to_container(config.data, resolve=True))
    logger.info(f'data module initialized')

    # init callbacks
    callbacks: list[Callback] = [LearningRateMonitor(), RichProgressBar(), GaitEval()]

    # save callback
    _ckpt_dir = Path(config.common.ckpt) / config.common.label / args.version
    save_ckpt = ModelCheckpoint(
        dirpath=_ckpt_dir,
        filename='back-{step:05d}',
        save_top_k=-1,
        every_n_train_steps=config.lifecycle.save_interval,
    )
    callbacks.append(save_ckpt)

    # metrics related callbacks
    if args.save != 'default':
        _metric = args.save
        _metric_tag = _metric.replace('/', '_')
        save_best = ModelCheckpoint(
            dirpath=_ckpt_dir,
            filename=f'best-{{step:05d}}-{{{_metric_tag}:.4f}}',
            monitor=_metric,
            save_top_k=3,
            mode='max',
        )
        early_stop = EarlyStopping(
            monitor=_metric,
            mode='max',
        )
        callbacks.extend([save_best, early_stop])

    # trainer
    trainer = Trainer(
        # train
        max_epochs=1,
        max_steps=config.lifecycle.total,
        val_check_interval=config.lifecycle.eval_interval,
        callbacks=callbacks,
        # logger & profiler
        logger=wandb_l,
        # profiler=profiler,
        # sampler
        use_distributed_sampler=False,
        # others
        **config.lightning.common,
    )
    logger.info(f'trainer initialized')

    # train phase
    if args.phase in ['train']:
        logger.info(f'{args.phase} phase: starting fit')
        trainer.fit(model, datamodule=data_module)

    # val phase
    if args.phase in ['val']:
        logger.info('eval phase: starting test')
        trainer.validate(model, datamodule=data_module)

    # predict phase
    if args.phase in ['pred']:
        logger.info('predict phase: starting predict')
        trainer.predict(model, datamodule=data_module)


if __name__ == '__main__':
    main()
