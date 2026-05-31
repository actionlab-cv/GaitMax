from pathlib import Path

import torch
from torch import Tensor
from torchcodec.decoders import VideoDecoder


def load_mp4(path: Path) -> Tensor:
    return VideoDecoder(path, device='cpu')[:]


def load_pt(path: Path) -> Tensor:
    return torch.load(path, weights_only=True)


def load_caption(path: Path) -> Tensor:
    # caption.pt = {emb: fp16 [l, d], label, meta}; return the per-sequence attribute embedding
    return torch.load(path, weights_only=False)['emb'].float()
