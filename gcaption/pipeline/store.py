"""Output read/write + resume checks.

Outputs mirror the input relative path under a persistent `out_root`:
    <out_root>/<rel>/caption.json   per-frame captions (+ meta)
    <out_root>/<rel>/caption.pt     {emb: float16[7,d], label, meta}
"""

import json
from pathlib import Path

import torch

from gcaption.pipeline.source import SequenceRef
from gcaption.schema import FrameCaption, SequenceCaption

JSON_NAME = "caption.json"
PT_NAME = "caption.pt"


def out_dir(out_root: Path, ref: SequenceRef) -> Path:
    return Path(out_root) / ref.rel


def json_path(out_root: Path, ref: SequenceRef) -> Path:
    return out_dir(out_root, ref) / JSON_NAME


def pt_path(out_root: Path, ref: SequenceRef) -> Path:
    return out_dir(out_root, ref) / PT_NAME


def write_json(out_root: Path, ref: SequenceRef, caption: SequenceCaption) -> Path:
    p = json_path(out_root, ref)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": ref.meta, "frames": [f.model_dump(mode="json") for f in caption.frames]}
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return p


def read_frames(out_root: Path, ref: SequenceRef) -> list[FrameCaption]:
    payload = json.loads(json_path(out_root, ref).read_text())
    return [FrameCaption.model_validate(f) for f in payload["frames"]]


def write_pt(out_root: Path, ref: SequenceRef, emb: torch.Tensor, label: dict) -> Path:
    p = pt_path(out_root, ref)
    p.parent.mkdir(parents=True, exist_ok=True)
    # store fp16 to halve size; embeddings are L2-normalized so fp16 precision is ample
    torch.save({"emb": emb.half(), "label": label, "meta": ref.meta}, p)
    return p


def merge_sequence(out_root: Path, ref: SequenceRef, label: dict) -> None:
    """Add the aggregated sequence-level caption into caption.json (readable)."""
    p = json_path(out_root, ref)
    payload = json.loads(p.read_text())
    payload["sequence"] = label
    p.write_text(json.dumps(payload, ensure_ascii=False))
