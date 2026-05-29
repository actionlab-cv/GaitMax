import torch
from torch import nn


def net_speed(net: nn.Module, device: torch.device):
    # params
    num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{net.__class__.__name__}: {num_p / 1e6:.2f}M parameters')
    # max memory
    print(f'{net.__class__.__name__}: {torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB memory')
