"""
GaitMax: dual-branch gait recognition combining semantic and kinematic cues.

Semantic branch  (BigGait):  DINO multi-layer features + mask
    → denoising / appearance → fusion ResNet → horizontal pooling → embed
Kinematic branch (ours):     DINO last-layer features + pattern attention
    → per-part temporal attention → cross-part union → per-pattern embed
IQA:    learned per-frame quality scoring → soft temporal weighting + attention bias
Router: per-sample branch confidence from embedding structure analysis
    → pairwise-weighted distance fusion for retrieval

External dependencies (frozen, not trained):
    - DINOWrapper  → f_v  [k, b, t, c, h, w]
    - HumanPrior   → mask [b, t, h, w], attn [b, t, m, h, w], conf [b, t, m]
"""

from functools import reduce
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn
from transformers import get_cosine_schedule_with_warmup

from gaitlight.data.transform.fetch_transform import fetch_transform
from gaitlight.losses.cross_entropy import CrossEntropy
from gaitlight.losses.disentangle import DisentangleLoss
from gaitlight.losses.triplet import TripletLoss
from gaitlight.models.common.dino_wrapper import DINOWrapper
from gaitlight.models.common.human_prior import HumanPrior
from gaitlight.models.gaitmax.iqa import FrameQuality
from gaitlight.models.gaitmax.kinematic.kinematic import KinematicBranch
from gaitlight.models.gaitmax.router import Router
from gaitlight.models.gaitmax.semantic.semantic import SemanticBranch
from gaitlight.types import InputBatch, EmbedBatch

__all__ = ['GaitMax']


class GaitMax(pl.LightningModule):
    """
    Dual-branch gait recognition model with IQA and adaptive routing.

    :param config: full experiment config (backbone, pipeline, train, transform, etc.)
    """

    def __init__(self, config: DictConfig | ListConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # frozen feature extractors
        self.dino = DINOWrapper(**config.backbone)
        self.prior = HumanPrior(
            s_dim=int(self.dino.embed_dim),
            n_slots=config.get('n_slots', 11),
            n_iter=config.get('n_iter', 3),
        )
        if 'prior_weights' in config:
            sd = torch.load(config.prior_weights, map_location='cpu', weights_only=True)
            self.prior.load_state_dict(sd, strict=True)

        # frame quality estimator
        self.iqa = FrameQuality(**config.get('iqa', {}))

        # trainable branches
        self.semantic = SemanticBranch(**config.semantic)
        kin_cfg = {k: v for k, v in config.kinematic.items() if k != 'eval_window'}
        self.kinematic = KinematicBranch(**kin_cfg)

        # branch confidence router (optional, disable for from-scratch training)
        self.use_router = 'router' in config and config.router is not None
        if self.use_router:
            self.router = Router(
                sem_dim=config.semantic.embed_dim,
                kin_dim=config.kinematic.embed_dim,
                **config.router,
            )

        # losses
        self.triplet = TripletLoss(config.train.get('triplet', None))
        self.ce = CrossEntropy(scale=config.train.get('ce_scale', 16.0))
        self.loss_w = config.train.loss

        # conditional decorrelation loss (enabled when 'cd' weight is set; requires caption data)
        self.use_cd = 'cd' in config.train.loss
        if self.use_cd:
            self.cd = DisentangleLoss()

        # transforms (ModuleList so buffers move to device with the model)
        t_transform, v_transform = fetch_transform(config.transform)
        self.t_transform = nn.ModuleList(t_transform)
        self.v_transform = nn.ModuleList(v_transform)

    # ---- core forward ----

    def _forward_single(self, frame: Tensor) -> dict[str, Any]:
        """
        Run full pipeline on a single clip.

        :param frame: normalized RGB [b, t, 3, h, w]
        :return: sem, kin, conf_sem, conf_kin, quality
        """
        f_v = self.dino(frame)  # [k, b, t, c, h, w]
        prior_out = self.prior(f_v[-1:])  # mask, attn, conf, ...

        mask = prior_out['mask']  # [b, t, h, w] bool
        attn = prior_out['attn']  # [b, t, m, h, w]
        conf = prior_out['conf']  # [b, t, m]

        quality = self.iqa(conf, mask)  # [b, t]

        sem = self.semantic(f_v, mask, quality)
        kin = self.kinematic(f_v[-1], attn, quality)

        # router (optional)
        if self.use_router:
            routed = self.router(sem['embed'], kin['embed'])
            conf_sem, conf_kin = routed['conf_sem'], routed['conf_kin']
        else:
            conf_sem, conf_kin = None, None

        return {
            'sem': sem,
            'kin': kin,
            'conf_sem': conf_sem,
            'conf_kin': conf_kin,
            'quality': quality,
        }

    def _forward(self, frame: Tensor) -> dict[str, Any]:
        """
        Run full pipeline.  During training or short sequences, run once.
        During inference with long sequences, split into clips and average.

        :param frame: normalized RGB [b, t, 3, h, w]
        """
        window = self.config.kinematic.get('eval_window', 30)
        t = frame.shape[1]

        # training or short sequence: single pass
        if self.training or t <= window:
            return self._forward_single(frame)

        # inference with long sequence: multi-clip averaging
        return self._forward_multi_clip(frame, window)

    @torch.no_grad()
    def _forward_multi_clip(self, frame: Tensor, window: int) -> dict[str, Any]:
        """
        Multi-clip inference: DINO + HumanPrior + IQA run once on all frames,
        then branches run per-clip with IQA renormalized.  Avoids redundant
        frozen-model computation.

        :param frame: [b, t, 3, h, w] where t > window
        :param window: clip length (same as training frame count)
        """
        # frozen modules: run once on all frames
        f_v = self.dino(frame)  # [k, b, t, c, h, w]
        prior_out = self.prior(f_v[-1:])

        mask = prior_out['mask']  # [b, t, h, w]
        attn = prior_out['attn']  # [b, t, m, h, w]
        conf = prior_out['conf']  # [b, t, m]

        # clip indices
        t = frame.shape[1]
        clip_starts = list(range(0, t - window + 1, window))
        if not clip_starts:
            clip_starts = [0]

        sem_embeds, kin_embeds = [], []
        conf_sems, conf_kins = [], []

        for start in clip_starts:
            s = slice(start, start + window)

            # IQA per clip (softmax over window frames, matches training distribution)
            q_clip = self.iqa(conf[:, s], mask[:, s])

            # branches per clip
            sem_out = self.semantic(f_v[:, :, s], mask[:, s], q_clip)
            kin_out = self.kinematic(f_v[-1, :, s], attn[:, s], q_clip)

            sem_embeds.append(sem_out['embed'])
            kin_embeds.append(kin_out['embed'])

            if self.use_router:
                routed = self.router(sem_out['embed'], kin_out['embed'])
                conf_sems.append(routed['conf_sem'])
                conf_kins.append(routed['conf_kin'])

        return {
            'sem': {'embed': torch.stack(sem_embeds).mean(dim=0),
                    'logits': None, 'loss': {}},
            'kin': {'embed': torch.stack(kin_embeds).mean(dim=0),
                    'logits': None},
            'conf_sem': torch.stack(conf_sems).mean(dim=0) if conf_sems else None,
            'conf_kin': torch.stack(conf_kins).mean(dim=0) if conf_kins else None,
            'quality': None,
        }

    # ---- training ----

    def training_step(self, batch: InputBatch, batch_idx: int) -> STEP_OUTPUT:
        for t in self.t_transform:
            batch = t(batch)

        frame = batch.seq.frame  # [b, t, 3, h, w]
        label = batch.label  # [b]

        out = self._forward(frame)

        # --- losses ---
        loss: dict[str, Tensor] = {}

        # semantic auxiliary losses
        loss['sem_con'] = out['sem']['loss']['loss_con']
        loss['sem_div'] = out['sem']['loss']['loss_div']

        # gather for distributed training
        all_label = self.all_gather(label).flatten(0, 1)

        # per-branch triplet losses (independent training signal)
        all_sem_embed = self.all_gather(out['sem']['embed'], sync_grads=True).flatten(0, 1)
        all_kin_embed = self.all_gather(out['kin']['embed'], sync_grads=True).flatten(0, 1)

        trip_sem = self.triplet({'embed': all_sem_embed, 'label': all_label})
        trip_kin = self.triplet({'embed': all_kin_embed, 'label': all_label})
        loss['triplet_sem'] = trip_sem['loss']['triplet']
        loss['triplet_kin'] = trip_kin['loss']['triplet']

        # weighted fused triplet (trains router confidence, skip if no router)
        if self.use_router:
            all_conf_sem = self.all_gather(out['conf_sem'], sync_grads=True).flatten(0, 1)
            all_conf_kin = self.all_gather(out['conf_kin'], sync_grads=True).flatten(0, 1)

            trip_fused = self.triplet({
                'embed_sem': all_sem_embed.detach(),
                'embed_kin': all_kin_embed.detach(),
                'label': all_label,
                'conf_sem': all_conf_sem,
                'conf_kin': all_conf_kin,
            })
            loss['triplet_fused'] = trip_fused['loss']['triplet']

            # confidence regularization: prevent conf from collapsing to 0
            loss['conf_reg'] = -(out['conf_sem'].mean().clamp(min=1e-8).log()
                                 + out['conf_kin'].mean().clamp(min=1e-8).log())

        # cross-entropy on both branches
        # before ce_start_step: detach embed → only PartHead learns new class boundaries
        # after ce_start_step: full gradient → CE also finetunes embedding space
        ce_start = self.config.train.get('ce_start_step', 0)
        ce_detach = self.global_step < ce_start

        all_sem_logits = self.all_gather(
            self.semantic.head(out['sem']['embed'].detach())['logits'] if ce_detach
            else out['sem']['logits'],
            sync_grads=True,
        ).flatten(0, 1)
        all_kin_logits = self.all_gather(
            self.kinematic.head(out['kin']['embed'].detach())['logits'] if ce_detach
            else out['kin']['logits'],
            sync_grads=True,
        ).flatten(0, 1)

        ce_sem = self.ce({'logits': all_sem_logits, 'label': all_label})
        ce_kin = self.ce({'logits': all_kin_logits, 'label': all_label})
        loss['ce_sem'] = ce_sem['loss']['cross_entropy']
        loss['ce_kin'] = ce_kin['loss']['cross_entropy']

        # conditional decorrelation: decorrelate the comprehensive embedding R from caption attributes
        if self.use_cd:
            all_cpt = self.all_gather(batch.cpt).flatten(0, 1)  # [n, l, d] (fixed text features, no grad)
            # comprehensive R = concat(sem, kin) along part dim, same as validation_step
            embed_r = torch.cat([all_sem_embed, all_kin_embed], dim=-1)  # [n, d, p_total]
            loss['cd'] = self.cd({'embed': embed_r, 'label': all_label, 'cpt': all_cpt})['loss']['cd']

        # weighted total
        l_tot = reduce(lambda a, b: a + b, [loss[k] * self.loss_w[k] for k in loss])

        # logging - losses
        self.log('train/loss', l_tot, prog_bar=True)
        self.log_dict({f'train/{k}': v for k, v in loss.items()})

        # logging - classification accuracy
        self.log('train/acc_sem', ce_sem['acc'], prog_bar=True)
        self.log('train/acc_kin', ce_kin['acc'])

        # logging - triplet health
        self.log('train/active_sem', trip_sem['visual']['active'])
        self.log('train/active_kin', trip_kin['visual']['active'])
        self.log('train/triplet_sem_ap', trip_sem['visual']['ap'])
        self.log('train/triplet_sem_an', trip_sem['visual']['an'])
        self.log('train/triplet_kin_ap', trip_kin['visual']['ap'])
        self.log('train/triplet_kin_an', trip_kin['visual']['an'])

        # logging - IQA quality
        quality = out['quality']
        self.log('train/quality_max', quality.max())
        self.log('train/quality_min', quality.min())
        self.log('train/quality_entropy', -(quality * quality.clamp(min=1e-8).log()).sum(dim=-1).mean())
        self.log('train/quality_tau', self.iqa.tau)

        # logging - router confidence
        if self.use_router:
            self.log('train/conf_sem', out['conf_sem'].mean())
            self.log('train/conf_kin', out['conf_kin'].mean())

        if hasattr(self.triplet, 'step'):
            self.triplet.step()

        return {'loss': l_tot}

    # ---- validation ----

    def validation_step(self, batch: InputBatch, batch_idx: int) -> EmbedBatch:
        for t in self.v_transform:
            batch = t(batch)

        out = self._forward(batch.seq.frame)

        # concat sem + kin along part dimension for retrieval
        embed = torch.cat([out['sem']['embed'], out['kin']['embed']], dim=-1)  # [b, d, p_total]

        return EmbedBatch(embed=embed, label=batch.label, meta=batch.meta)

    # ---- optimizer ----

    def configure_optimizers(self) -> OptimizerLRScheduler:
        cfg = self.config.train
        wd = cfg.weight_decay

        # per-module LR groups
        param_groups = [
            {'params': [p for p in self.semantic.parameters() if p.requires_grad],
             'lr': cfg.get('lr_semantic', cfg.lr)},
            {'params': [p for p in self.kinematic.parameters() if p.requires_grad],
             'lr': cfg.get('lr_kinematic', cfg.lr)},
            {'params': [p for p in self.iqa.parameters() if p.requires_grad],
             'lr': cfg.get('lr_other', cfg.lr)},
        ]
        if self.use_router:
            param_groups.append(
                {'params': [p for p in self.router.parameters() if p.requires_grad],
                 'lr': cfg.get('lr_other', cfg.lr)},
            )

        opt = torch.optim.AdamW(param_groups, weight_decay=wd)
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=cfg.num_training_steps,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sch, 'interval': 'step', 'frequency': 1},
        }

    # ---- misc ----

    def state_dict(self, *args, **kwargs):
        """Exclude frozen DINO backbone weights from checkpoint."""
        return {
            k: v for k, v in super().state_dict(*args, **kwargs).items()
            if not k.startswith('dino.dino.')
        }
