"""Autocrop black padding via the silhouette mask, then aspect-preserving resize back to original size (letterbox/pillarbox) so the person fills the frame.

Standalone mirror of gaitlight/data/transform/auto_zoom.py:AutoZoom (the frame / 4D branch), kept separate to avoid gaitlight.data's eager package imports.
Keep in sync with that file.

Dims: t = frames, c = channels, h/w = height/width.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _bbox(masks: Tensor) -> tuple[int, int, int, int]:
    """Union content bounding box over time.

    Shapes
    ------
    masks  : [t, h, w] bool
    returns: (y1, y2, x1, x2) inclusive pixel bounds
    """
    h, w = masks.shape[1], masks.shape[2]
    # shape: [t, h, w] -> [h, w]
    union = masks.any(dim=0)
    # shape: [h, w] -> [h] -> [n_rows]; [h, w] -> [w] -> [n_cols]
    rows = union.any(dim=1).nonzero(as_tuple=False).view(-1)
    cols = union.any(dim=0).nonzero(as_tuple=False).view(-1)
    if rows.numel() == 0 or cols.numel() == 0:
        return 0, h - 1, 0, w - 1
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def autocrop(frames: Tensor, masks: Tensor, border: int = 1) -> Tensor:
    """Zoom the masked content to fill the frame, preserving aspect ratio.

    No-op unless the content is padded on all four sides (matches AutoZoom).

    Shapes
    ------
    frames : [t, c, h, w] uint8
    masks  : [t, h, w]    bool
    returns: [t, c, h, w] uint8 (same h, w)
    """
    h, w = masks.shape[-2], masks.shape[-1]
    y1, y2, x1, x2 = _bbox(masks)
    if not (y1 >= border and y2 <= h - 1 - border and x1 >= border and x2 <= w - 1 - border):
        return frames
    ch, cw = y2 - y1 + 1, x2 - x1 + 1
    scale = min(h / ch, w / cw)
    nh, nw = round(ch * scale), round(cw * scale)
    pt, pl = (h - nh) // 2, (w - nw) // 2
    # interpolate requires float input
    # shape: [t, c, h, w] -> [t, c, ch, cw]
    cropped = frames[:, :, y1 : y2 + 1, x1 : x2 + 1].float()
    # shape: [t, c, ch, cw] -> [t, c, nh, nw]
    resized = F.interpolate(cropped, size=(nh, nw), mode="bilinear", align_corners=False)
    # shape: [t, c, nh, nw] -> [t, c, h, w] (letterbox/pillarbox)
    out = F.pad(resized, (pl, w - nw - pl, pt, h - nh - pt))
    # restore the uint8 image dtype after float resize
    return out.round().clamp(0, 255).to(torch.uint8)
