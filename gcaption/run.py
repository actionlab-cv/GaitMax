"""gcaption CLI.

    python -m gcaption.run list     --seq-root ... [--datasets ccpg ccgrm] [--limit N]
    python -m gcaption.run annotate --seq-root ... --out-root ... [--model M] [--workers W] [--overwrite]
    python -m gcaption.run embed    --out-root ... [--device cuda] [--shard i/n] [--overwrite]

Stage 1 (annotate) is CPU/network bound; Stage 2 (embed) is GPU bound.
Both are idempotent: existing caption.json / caption.pt are skipped unless --overwrite.
"""

import argparse
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.logging import RichHandler
from rich.progress import Progress

from gcaption.config import GCaptionConfig
from gcaption.pipeline import source, store
from gcaption.pipeline.aggregate import aggregate
from gcaption.pipeline.annotate import Annotator, OpenAIAnnotator
from gcaption.pipeline.embed import Embedder

logger = logging.getLogger("gcaption")


def _refs(cfg: GCaptionConfig) -> list[source.SequenceRef]:
    return source.discover_sequences(cfg.seq_root, cfg.datasets, cfg.limit)


def cmd_list(cfg: GCaptionConfig) -> None:
    refs = _refs(cfg)
    by = Counter(r.meta["dataset"] for r in refs)
    for k, v in sorted(by.items()):
        logger.info("%-12s %d", k, v)
    logger.info("%-12s %d", "TOTAL", len(refs))


def cmd_annotate(cfg: GCaptionConfig) -> None:
    refs = _refs(cfg)
    todo = [r for r in refs if cfg.overwrite or not store.json_path(cfg.out_root, r).exists()]
    logger.info(
        "annotate: %d total, %d todo, %d skipped", len(refs), len(todo), len(refs) - len(todo)
    )
    if not todo:
        return
    if cfg.model.startswith(("gpt", "o1", "o3", "o4")):
        ann = OpenAIAnnotator(cfg.model, max_retries=cfg.max_retries)
    else:
        ann = Annotator(cfg.model, max_retries=cfg.max_retries, media_res=cfg.media_res)

    def work(r: source.SequenceRef) -> None:
        frames = source.sample_frames(r.frame_path, cfg.frame_num, cfg.autocrop)
        store.write_json(cfg.out_root, r, ann.annotate(frames))

    ok = fail = 0
    with Progress() as prog, ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        task = prog.add_task("annotate", total=len(todo))
        futs = {ex.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            try:
                fu.result()
                ok += 1
            except Exception as e:
                fail += 1
                logger.error("FAIL %s: %s", futs[fu].rel, e)
            prog.advance(task)
    logger.info("annotate done: ok=%d fail=%d", ok, fail)


def cmd_embed(cfg: GCaptionConfig) -> None:
    i, n = (int(x) for x in cfg.shard.split("/"))
    refs = source.discover_captions(cfg.out_root, cfg.datasets, cfg.limit)[i::n]
    todo = [r for r in refs if cfg.overwrite or not store.pt_path(cfg.out_root, r).exists()]
    logger.info("embed: shard %d/%d -> %d caption.json, %d todo", i, n, len(refs), len(todo))
    if not todo:
        return
    emb = Embedder(cfg.clip_model, cfg.clip_pretrained, cfg.device)
    ok = fail = 0
    with Progress() as prog:
        task = prog.add_task("embed", total=len(todo))
        for r in todo:
            try:
                frames = store.read_frames(cfg.out_root, r)
                label, seq = aggregate(frames, emb.encode_sequence(frames))
                store.write_pt(cfg.out_root, r, seq, label)
                store.merge_sequence(cfg.out_root, r, label)
                ok += 1
            except Exception as e:
                fail += 1
                logger.error("FAIL %s: %s", r.rel, e)
            prog.advance(task)
    logger.info("embed done: ok=%d fail=%d", ok, fail)


def _cfg(args: argparse.Namespace) -> GCaptionConfig:
    return GCaptionConfig(
        seq_root=getattr(args, "seq_root", None) or getattr(args, "out_root", None) or ".",
        out_root=getattr(args, "out_root", None) or ".",
        datasets=tuple(args.datasets) if args.datasets else None,
        limit=args.limit,
        frame_num=getattr(args, "frame_num", 8),
        autocrop=not getattr(args, "no_autocrop", False),
        model=getattr(args, "model", GCaptionConfig.model),
        media_res=getattr(args, "media_res", GCaptionConfig.media_res),
        workers=getattr(args, "workers", GCaptionConfig.workers),
        clip_model=getattr(args, "clip_model", GCaptionConfig.clip_model),
        clip_pretrained=getattr(args, "clip_pretrained", GCaptionConfig.clip_pretrained),
        device=getattr(args, "device", GCaptionConfig.device),
        shard=getattr(args, "shard", GCaptionConfig.shard),
        overwrite=getattr(args, "overwrite", False),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gcaption")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser, seq: bool = True, out: bool = True) -> None:
        if seq:
            sp.add_argument("--seq-root", required=True)
        if out:
            sp.add_argument("--out-root", required=True)
        sp.add_argument("--datasets", nargs="*", default=None)
        sp.add_argument("--limit", type=int, default=None)

    lp = sub.add_parser("list")
    add_common(lp, seq=True, out=False)
    lp.set_defaults(fn=cmd_list)

    ap = sub.add_parser("annotate")
    add_common(ap, seq=True, out=True)
    ap.add_argument("--frame-num", type=int, default=8)
    ap.add_argument("--no-autocrop", action="store_true")
    ap.add_argument("--model", default=GCaptionConfig.model)
    ap.add_argument(
        "--media-res",
        default=GCaptionConfig.media_res,
        choices=["default", "low", "medium", "high"],
    )
    ap.add_argument("--workers", type=int, default=GCaptionConfig.workers)
    ap.add_argument("--overwrite", action="store_true")
    ap.set_defaults(fn=cmd_annotate)

    ep = sub.add_parser("embed")
    add_common(ep, seq=False, out=True)  # embed reads only caption.json; no video needed
    ep.add_argument("--clip-model", default=GCaptionConfig.clip_model)
    ep.add_argument("--clip-pretrained", default=GCaptionConfig.clip_pretrained)
    ep.add_argument("--device", default=GCaptionConfig.device)
    ep.add_argument(
        "--shard", default=GCaptionConfig.shard, help="'i/n' -> handle refs[i::n] (parallel embed)"
    )
    ep.add_argument("--overwrite", action="store_true")
    ep.set_defaults(fn=cmd_embed)
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)]
    )
    args = build_parser().parse_args()
    args.fn(_cfg(args))


if __name__ == "__main__":
    main()
