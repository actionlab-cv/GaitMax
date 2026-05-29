import logging

import torch

logger = logging.getLogger(__name__)


def nan_hook(module: torch.nn.Module, inputs, output):
    def _check_tensor(x):
        if not isinstance(x, torch.Tensor):
            return False, False
        return torch.isnan(x).any().item(), torch.isinf(x).any().item()

    # check input
    nan_in, inf_in = False, False
    if isinstance(inputs, torch.Tensor):
        nan_in, inf_in = _check_tensor(inputs)
    elif isinstance(inputs, (list, tuple)):
        for i in inputs:
            n, f = _check_tensor(i)
            nan_in |= n
            inf_in |= f

    # check output
    nan_out, inf_out = False, False
    if isinstance(output, torch.Tensor):
        nan_out, inf_out = _check_tensor(output)
    elif isinstance(output, (list, tuple)):
        for o in output:
            n, f = _check_tensor(o)
            nan_out |= n
            inf_out |= f

    if nan_in or inf_in or nan_out or inf_out:
        logger.fatal(f'found NaN in {module.__class__.__name__} [in(nan={nan_in}, inf={inf_in}) -> out(nan={nan_out}, inf={inf_out})]')
        raise ValueError('NaN or Inf detected, stop training')
