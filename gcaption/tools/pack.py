"""Package captions + embeddings into release artifacts (for HuggingFace).

    python -m gcaption.tools.pack --src <sequences_root> --out <release_dir>

Produces under <release_dir>:
    captions/<dataset>.jsonl                one sequence/line (meta + frames + sequence)
    embeddings/gcaption_emb.safetensors     {"emb": [N, 7, 768] fp16}  (safe, no pickle)
    embeddings/ids.json                     ordered ids aligned to emb rows
    index.csv                               id,dataset,subject,condition,view
    README.md, SCHEMA.md
Also scans free-text fields for inappropriate terms and reports hits.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from gcaption.schema import ATTR_ORDER

DATASET_MAP = {"casiab": "casiab", "ccgrm": "ccgrm", "ccpg": "ccpg", "sus": "sustech1k"}

# heuristic blocklist (word-boundary) — catches egregious VLM output for human review
BLOCKLIST = [
    "nude",
    "naked",
    "nsfw",
    "sexual",
    "porn",
    "genital",
    "nipple",
    "blood",
    "gore",
    "weapon",
    "gun",
    "knife",
    "corpse",
    "dead body",
    "fuck",
    "shit",
    "bitch",
    "slut",
    "whore",
    "nigger",
    "faggot",
    "retard",
]
_BLOCK_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BLOCKLIST) + r")\b", re.I)

# plain string (literal braces) so the f-string README template can interpolate it safely
CITATION = """## Citation

If you use GCaption, please cite **GaitMax**:

```bibtex
@InProceedings{Huang_2026_CVPR,
    author    = {Huang, Zhanbo and Ye, Dingqiang and Liu, Xiaoming and Kong, Yu},
    title     = {Unlocking Motion from Large Vision Models with a Semantic and Kinematic Duality for Gait Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {28379-28390}
}
```

Please also cite the source gait datasets (CASIA-B, CCPG, CCGR, SUSTech1K)."""


def _seqtext(seq: dict) -> str:
    parts = [str(seq.get("attire", "")), str(seq.get("location", ""))]
    parts += seq.get("related_item", []) or []
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="sequences root holding caption.json/caption.pt")
    ap.add_argument("--out", required=True, help="release output dir")
    ap.add_argument("--license", default="cc-by-nc-4.0")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "captions").mkdir(parents=True, exist_ok=True)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)

    jsons = sorted(src.rglob("caption.json"))
    print(f"packing {len(jsons)} sequences")

    writers: dict = {}
    ids, embs, rows = [], [], []
    counts, flagged = Counter(), []

    for jp in jsons:
        d = json.loads(jp.read_text())
        m = d["meta"]
        ds = DATASET_MAP.get(m["dataset"], m["dataset"])
        rid = m["rel"]
        rec = {
            "id": rid,
            "dataset": ds,
            "subject": m["subject"],
            "condition": m["condition"],
            "view": m["view"],
            "frames": d.get("frames", []),
            "sequence": d.get("sequence"),
        }
        if ds not in writers:
            writers[ds] = (out / "captions" / f"{ds}.jsonl").open("w")
        writers[ds].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[ds] += 1
        rows.append((rid, ds, m["subject"], m["condition"], m["view"]))

        pt = jp.with_name("caption.pt")
        if pt.exists():
            embs.append(torch.load(pt, map_location="cpu", weights_only=False)["emb"])
            ids.append(rid)

        if d.get("sequence") and _BLOCK_RE.search(_seqtext(d["sequence"])):
            flagged.append((rid, _BLOCK_RE.findall(_seqtext(d["sequence"]))))

    for f in writers.values():
        f.close()

    emb = torch.stack(embs).to(torch.float16) if embs else torch.empty(0)
    save_file({"emb": emb}, str(out / "embeddings" / "gcaption_emb.safetensors"))
    (out / "embeddings" / "ids.json").write_text(json.dumps(ids))

    with (out / "index.csv").open("w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["id", "dataset", "subject", "condition", "view"])
        cw.writerows(rows)

    total = sum(counts.values())
    (out / "SCHEMA.md").write_text(_schema_md())
    (out / "README.md").write_text(_readme_md(args.license, total, counts, tuple(emb.shape)))

    print("\n=== summary ===")
    for ds, n in sorted(counts.items()):
        print(f"  {ds:10s} {n}")
    print(f"  TOTAL      {total}")
    print(f"  embeddings {tuple(emb.shape)} {emb.dtype}  ids={len(ids)}")
    print(f"\n=== free-text sensitive-term scan: {len(flagged)} flagged ===")
    for rid, terms in flagged[:25]:
        print(f"  {rid}: {terms}")


def _schema_md() -> str:
    order = ", ".join(f"{i}:{a}" for i, a in enumerate(ATTR_ORDER))
    return f"""# GCaption Schema

Seven attributes, fixed order (embedding axis `l=7`): {order}

| attribute | type | format / values |
|---|---|---|
| age | multi-choice | youth / young adult / adult / mid-ages / senior |
| attire | free-text | `[top color + top type] [bottom color + bottom type]` |
| action | multi-choice | stand / walk / run |
| related_item | list | objects carried, each `[adjective] [noun]`; `[]` if none |
| location | free-text | short scene/background description |
| viewpoint | structured | `{{roll: level/tilted, pitch: overhead/eye-level/low-angle, yaw: front/side/back}}` |
| lighting | multi-choice | bright / dim / shadowed / natural light / artificial light |

## Files
- `captions/<dataset>.jsonl` — one sequence per line: `id, dataset, subject, condition, view, frames[], sequence{{}}`.
  `frames` = per-frame captions (8 sampled frames, fewer for short clips); `sequence` = aggregated label.
- `embeddings/gcaption_emb.safetensors` — `emb` `[N, 7, 768]` fp16, OpenCLIP ViT-L-14 (laion2b_s32b_b82k)
  text features per attribute, L2-normalized. Row `i` ↔ `ids[i]`.
- `embeddings/ids.json` — ordered ids aligned to embedding rows.
- `index.csv` — sequence index.

Join with the source datasets by `id` (= `<dataset>_<subject>/<condition>/<view>`).
"""


def _readme_md(license_: str, total: int, counts: Counter, shape: tuple) -> str:
    rows = "\n".join(f"| {ds} | {n} |" for ds, n in sorted(counts.items()))
    return f"""---
license: {license_}
language: [en]
pretty_name: GCaption
size_categories: [100K<n<1M]
tags: [gait-recognition, vision-language, captions, attributes]
---

# GCaption

Natural-language attribute annotations for multiple RGB gait datasets, introduced in
**GaitMax** (CVPR'26). Each walking sequence is described by 7 attributes (see `SCHEMA.md`)
plus a precomputed OpenCLIP text embedding per attribute, to support context-aware gait
research and the Conditional Decorrelation Loss (CDLoss).

**{total:,} sequences** across 4 datasets:

| dataset | sequences |
|---|---|
{rows}

Embeddings: `{shape}` fp16.

## What this is / is NOT
- ✅ Includes: text captions + per-attribute OpenCLIP embeddings, keyed by sequence `id`.
- ❌ Does NOT include source frames/masks/poses. Obtain CASIA-B, CCPG, CCGR, SUSTech1K from
  their original providers under their licenses, and join by `id`.

## Usage
```python
import json, torch
from safetensors.torch import load_file

caps = [json.loads(l) for l in open("captions/ccpg.jsonl")]
emb = load_file("embeddings/gcaption_emb.safetensors")["emb"]   # [N,7,768] fp16
ids = json.load(open("embeddings/ids.json"))
row = {{i: k for k, i in enumerate(ids)}}["ccpg_007/CL/090".replace("ccpg","ccpg")]
```

## Provenance
- Captions: Gemini-2.5-flash-lite (structured output); a small fraction blocked by the VLM's
  PROHIBITED_CONTENT filter were recovered with GPT-4o. Frames mask-autocropped before captioning.
- Embeddings: OpenCLIP ViT-L-14 (laion2b_s32b_b82k), per-attribute, sequence-level aggregated.

## Ethics & limitations
- Attributes (esp. **age**) are **model-inferred apparent attributes, not verified ground truth
  or demographic data**. Do not use for identification of individuals.
- Source data is face–de-identified; these annotations add no PII.
- Known limits: color ambiguity under dim lighting; long-tailed attribute distribution;
  viewpoint yaw is only weakly separated in text-embedding space.

{CITATION}
"""


if __name__ == "__main__":
    main()
