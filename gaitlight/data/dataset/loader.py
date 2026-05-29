from pathlib import Path

import torch
from torch import Tensor
from torchcodec.decoders import VideoDecoder


def load_mp4(path: Path) -> Tensor:
    return VideoDecoder(path, device='cpu')[:]


def load_pt(path: Path) -> Tensor:
    return torch.load(path, weights_only=True)
