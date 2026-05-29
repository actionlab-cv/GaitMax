import dataclasses

import torch
import torch.nn.functional as func
from torch import Tensor, nn

from gaitlight.types import InputBatch, SequenceData


class AutoZoom(nn.Module):
    """
    Detect four-sided zero padding in mask, crop to content bounding box,
    then resize back to original dimensions while preserving aspect ratio.
    The result has at most one axis padded (letterbox or pillarbox).

    Automatically handles all spatial modalities in SequenceData:
    - 4D [t, c, h, w] tensors: bilinear interpolation (e.g. frame)
    - 3D [t, h, w] tensors with matching spatial dims: nearest interpolation (e.g. mask, pattern)
    - pose [t, k, 3] with (x, y, conf) in pixel coords: coordinate transform
    """

    def __init__(self, border: int = 1):
        super().__init__()
        self.border = border

    @torch.no_grad()
    def forward(self, batch: InputBatch) -> InputBatch:
        mask = batch.seq.mask
        if mask is None:
            return batch

        h, w = mask.shape[-2], mask.shape[-1]
        b = mask.shape[0]

        results: dict[str, list[Tensor]] = {
            f.name: [] for f in dataclasses.fields(SequenceData)
            if getattr(batch.seq, f.name) is not None
        }

        for i in range(b):
            m_i = mask[i]  # [t, h, w]
            y1, y2, x1, x2 = self._bbox(m_i)

            if self._is_padded(m_i, y1, y2, x1, x2):
                # compute transform params once, reuse for all modalities
                ch, cw = y2 - y1 + 1, x2 - x1 + 1
                scale = min(h / ch, w / cw)
                nh, nw = round(ch * scale), round(cw * scale)
                pt, pl = (h - nh) // 2, (w - nw) // 2

                for name in results:
                    v = getattr(batch.seq, name)[i]
                    results[name].append(self._zoom(v, h, w, y1, y2, x1, x2, scale, nh, nw, pt, pl))
            else:
                for name in results:
                    results[name].append(getattr(batch.seq, name)[i])

        new_seq = dataclasses.replace(
            batch.seq,
            **{name: torch.stack(tensors) for name, tensors in results.items()},
        )
        return dataclasses.replace(batch, seq=new_seq)

    @staticmethod
    def _zoom(
            v: Tensor, h: int, w: int,
            y1: int, y2: int, x1: int, x2: int,
            scale: float, nh: int, nw: int, pt: int, pl: int,
    ) -> Tensor:
        # 4D [t, c, h, w] - frame-like
        if v.ndim == 4 and v.shape[2] == h and v.shape[3] == w:
            cropped = v[:, :, y1:y2 + 1, x1:x2 + 1]
            resized = func.interpolate(cropped.float(), size=(nh, nw), mode='bilinear', align_corners=False).to(v.dtype)
            return func.pad(resized, (pl, w - nw - pl, pt, h - nh - pt))

        # 3D [t, h, w] - mask/label-like
        if v.ndim == 3 and v.shape[1] == h and v.shape[2] == w:
            cropped = v[:, y1:y2 + 1, x1:x2 + 1]
            resized = func.interpolate(cropped.float().unsqueeze(1), size=(nh, nw), mode='nearest').squeeze(1)
            padded = func.pad(resized, (pl, w - nw - pl, pt, h - nh - pt))
            return padded.bool() if v.dtype == torch.bool else padded.to(v.dtype)

        # 3D [t, k, 3] - pose keypoints (x, y, conf)
        if v.ndim == 3 and v.shape[2] == 3 and v.shape[1] != h:
            out = v.clone().float()
            out[..., 0] = (v[..., 0] - x1) * scale + pl  # x
            out[..., 1] = (v[..., 1] - y1) * scale + pt  # y
            return out.to(v.dtype)

        return v

    def _is_padded(self, mask: Tensor, y1: int, y2: int, x1: int, x2: int) -> bool:
        h, w = mask.shape[1], mask.shape[2]
        n = self.border
        return y1 >= n and y2 <= h - 1 - n and x1 >= n and x2 <= w - 1 - n

    @staticmethod
    def _bbox(mask: Tensor) -> tuple[int, int, int, int]:
        union = mask.any(dim=0)
        rows = union.any(dim=1)
        cols = union.any(dim=0)
        h, w = mask.shape[1], mask.shape[2]

        y_idx = rows.nonzero(as_tuple=False).view(-1)
        x_idx = cols.nonzero(as_tuple=False).view(-1)

        if y_idx.numel() == 0 or x_idx.numel() == 0:
            return 0, h - 1, 0, w - 1

        return y_idx[0].item(), y_idx[-1].item(), x_idx[0].item(), x_idx[-1].item()
