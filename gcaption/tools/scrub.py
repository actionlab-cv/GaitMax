"""Scrub inappropriate VLM artifacts from captions before public release.

Drops weapon / explicit terms from `related_item` (per-frame and sequence-level), rewrites caption.json, and deletes the affected caption.pt so a subsequent `embed` run regenerates them from the cleaned text.
Free-text hits in attire/location are reported for manual review (not auto-edited).

    python -m gcaption.tools.scrub --src <sequences_root>
    # then: python -m gcaption.run embed --out-root <sequences_root>   (regenerates cleaned .pt)
"""

import argparse
import json
import re
from pathlib import Path

# weapon + explicit terms; benign colors like "nude"/"blood" intentionally excluded
SCRUB = [
    "gun",
    "guns",
    "handgun",
    "pistol",
    "revolver",
    "rifle",
    "shotgun",
    "firearm",
    "knife",
    "knives",
    "dagger",
    "blade",
    "machete",
    "sword",
    "grenade",
    "bomb",
    "weapon",
    "weapons",
    "ammo",
    "ammunition",
    "naked",
    "porn",
    "genital",
    "genitals",
    "nipple",
    "nipples",
]
RE = re.compile(r"\b(" + "|".join(map(re.escape, SCRUB)) + r")\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    root = Path(args.src)

    changed: list[str] = []
    manual: list[tuple] = []

    for jp in sorted(root.rglob("caption.json")):
        d = json.loads(jp.read_text())
        rid = d["meta"]["rel"]
        dirty = False

        def clean(lst):
            nonlocal dirty
            if not lst:
                return lst
            kept = [x for x in lst if not RE.search(str(x))]
            if len(kept) != len(lst):
                dirty = True
            return kept

        for fr in d.get("frames", []):
            fr["related_item"] = clean(fr.get("related_item", []))
            manual.extend(
                (rid, k, fr.get(k)) for k in ("attire", "location") if RE.search(str(fr.get(k, "")))
            )
        seq = d.get("sequence")
        if seq:
            seq["related_item"] = clean(seq.get("related_item", []))
            manual.extend(
                (rid, "seq." + k, seq.get(k))
                for k in ("attire", "location")
                if RE.search(str(seq.get(k, "")))
            )

        if dirty:
            jp.write_text(json.dumps(d, ensure_ascii=False))
            pt = jp.with_name("caption.pt")
            if pt.exists():
                pt.unlink()
            changed.append(rid)

    print(f"cleaned related_item in {len(changed)} sequences (caption.pt deleted for re-embed):")
    for r in changed:
        print(f"  {r}")
    print(f"\nfree-text (attire/location) hits needing MANUAL review: {len(manual)}")
    for rid, k, v in manual[:50]:
        print(f"  {rid} [{k}]: {v}")


if __name__ == "__main__":
    main()
