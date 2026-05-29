from typing import Any

import torch
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self, scale: float = 16.0):
        super().__init__()

        self.scale = scale

    def forward(self, u_data: Any) -> dict[str, Any]:
        """
        Cross entropy loss for classification.

        :param u_data: {'logits': [n, k (class), p], 'label': [n]}
        :return:
            - loss: {'cross_entropy'}
        """

        # logits [n, k, p]
        assert 'logits' in u_data, 'logits not found in batch'
        logits: torch.Tensor = u_data['logits']

        # expand part (if needed)
        logits = logits.unsqueeze(-1) if logits.dim() == 2 else logits
        n, k, p = logits.shape

        # unpack labels
        assert 'label' in u_data, 'label not found in batch'
        label: torch.Tensor = u_data['label'].unsqueeze(1)

        # meta holder
        loss, visual = {}, {}

        # cross entropy
        _ce = nn.functional.cross_entropy(logits * self.scale, label.repeat(1, p), label_smoothing=0.1)
        visual |= {'cross_entropy': _ce.detach()}
        loss |= {'cross_entropy': _ce}

        # accuracy
        pred = logits.argmax(dim=1)
        acc = (pred == label).float().mean()

        return {'loss': loss, 'visual': visual, 'acc': acc}
