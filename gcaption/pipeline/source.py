"""Sequence discovery + frame sampling over the unified gait pool.

Dataset-agnostic: a "sequence" is any leaf dir containing `frame.mp4` under `<seq_root>/<dataset>_<sid>/<cond>/<view>/`.
Frames are autocropped via the co-located `mask.mp4` silhouette so the person fills the frame for the VLM.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from gcaption.pipeline.autocrop import autocrop

logger = logging.getLogger(__name__)

FRAME_FILE = "frame.mp4"
MASK_FILE = "mask.mp4"


@dataclass(frozen=True)
class SequenceRef:
    rel: Path  # relative to seq_root, e.g. ccpg_007/CL/090
    frame_path: Path  # absolute path to frame.mp4

    @property
    def meta(self) -> dict[str, str]:
        parts = self.rel.parts
        subject = parts[0] if parts else ""
        return {
            "rel": str(self.rel),
            "dataset": subject.rsplit("_", 1)[0] if "_" in subject else subject,
            "subject": subject,
            "condition": parts[1] if len(parts) > 1 else "",
            "view": parts[2] if len(parts) > 2 else "",
        }


def discover_sequences(
    seq_root: Path,
    datasets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[SequenceRef]:
    """Enumerate all sequences (sorted for deterministic resume)."""
    seq_root = Path(seq_root)
    refs: list[SequenceRef] = []
    for frame_path in sorted(seq_root.rglob(FRAME_FILE)):
        rel = frame_path.parent.relative_to(seq_root)
        if datasets and not rel.parts[0].startswith(tuple(datasets)):
            continue
        refs.append(SequenceRef(rel=rel, frame_path=frame_path))
        if limit and len(refs) >= limit:
            break
    return refs


def discover_captions(
    out_root: Path,
    datasets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[SequenceRef]:
    """Enumerate sequences by existing caption.json (embed needs no video)."""
    out_root = Path(out_root)
    refs: list[SequenceRef] = []
    for jp in sorted(out_root.rglob("caption.json")):
        rel = jp.parent.relative_to(out_root)
        if datasets and not rel.parts[0].startswith(tuple(datasets)):
            continue
        refs.append(SequenceRef(rel=rel, frame_path=jp.parent / FRAME_FILE))
        if limit and len(refs) >= limit:
            break
    return refs


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _decode(path: Path, idxs: np.ndarray, gray: bool) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    code = cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB
    try:
        out: list[np.ndarray] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                out.append(cv2.cvtColor(bgr, code))
        return out
    finally:
        cap.release()


def sample_frames(frame_path: Path, n: int, autocrop_frames: bool = True) -> list[Image.Image]:
    """Uniformly sample n ordered frames as RGB PIL images, mask-autocropped."""
    total = _frame_count(frame_path)
    if total <= 0:
        raise RuntimeError(f"empty/undecodable video: {frame_path}")
    idxs = np.linspace(0, total - 1, num=min(n, total)).round().astype(int)

    rgb = _decode(frame_path, idxs, gray=False)
    if not rgb:
        raise RuntimeError(f"no frames decoded: {frame_path}")

    if autocrop_frames:
        mask_path = frame_path.parent / MASK_FILE
        masks = _decode(mask_path, idxs, gray=True) if mask_path.exists() else []
        if len(masks) == len(rgb) and all(
            m.shape == r.shape[:2] for m, r in zip(masks, rgb, strict=False)
        ):
            # shape: t*[h, w, c] -> [t, h, w, c] -> [t, c, h, w]
            frames_t = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)
            # shape: t*[h, w] -> [t, h, w] bool
            masks_t = torch.from_numpy(np.stack(masks)) > 127
            zoomed = autocrop(frames_t, masks_t)
            # shape: [t, c, h, w] -> t*[h, w, c]; contiguous() gives .numpy() a C-contiguous buffer after permute
            rgb = [zoomed[i].permute(1, 2, 0).contiguous().numpy() for i in range(zoomed.shape[0])]
        elif mask_path.exists():
            logger.warning("mask/frame mismatch, skip autocrop: %s", frame_path.parent)

    return [Image.fromarray(np.ascontiguousarray(x)) for x in rgb]
