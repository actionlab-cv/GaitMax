from dataclasses import dataclass
from pathlib import Path

from torch import Tensor


@dataclass
class SequenceMeta:
    subject: str
    caption: str
    view: str
    path: Path


@dataclass
class SampleInfo:
    type: list[str]  # e.g. ['natural', 'ordered']
    num: int  # frames sampled
    index: list[int | None]  # frame indices (None = padding)


@dataclass
class SequenceData:
    frame: Tensor | None = None  # [t, 3, h, w] uint8
    pose: Tensor | None = None  # [t, 17, 3]
    mask: Tensor | None = None  # [t, h, w] bool
    pattern: Tensor | None = None  # [t, h, w] uint8 (body part labels)


@dataclass
class SequenceItem:
    seq: SequenceData
    label: int
    meta: SequenceMeta


@dataclass
class SequenceBatch:
    frame: Tensor | None  # [b, t, 3, h, w]
    pose: Tensor | None  # [b, t, 17, 3]
    mask: Tensor | None  # [b, t, h, w]
    pattern: Tensor | None = None  # [b, t, h, w] uint8 (body part labels)


@dataclass
class InputBatch:
    seq: SequenceBatch
    label: Tensor  # [b]
    meta: list[SequenceMeta]
    sample: list[SampleInfo]


@dataclass
class EmbedBatch:
    embed: Tensor  # [b, c] or [b, c, p]
    label: Tensor  # [b]
    meta: list[SequenceMeta]
