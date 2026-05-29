"""Runtime config for the gcaption CLI (plain dataclass, no Lightning)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GCaptionConfig:
    # io
    seq_root: Path  # input pool: <seq_root>/<dataset>_<sid>/<cond>/<view>/frame.mp4
    out_root: Path  # persistent output (NFS), mirrors relative path
    # selection
    datasets: tuple[str, ...] | None = None  # filter by top-level prefix, e.g. ("ccpg", "ccgrm")
    limit: int | None = None  # cap number of sequences (testing)
    frame_num: int = 8  # frames sampled per sequence (paper Supp B)
    autocrop: bool = True  # mask-based autocrop of black padding before VLM
    # annotate (stage 1)
    model: str = "gemini-2.5-flash-lite"
    media_res: str = "default"  # default|low|medium|high (low only helps 3.x; hurts color)
    workers: int = 8
    max_retries: int = 5
    # embed (stage 2)
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "laion2b_s32b_b82k"
    device: str = "cuda"
    shard: str = "0/1"  # "i/n" -> this process handles refs[i::n] (parallel embed)
    # behavior
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.seq_root = Path(self.seq_root)
        self.out_root = Path(self.out_root)
        if isinstance(self.datasets, list):
            self.datasets = tuple(self.datasets)
