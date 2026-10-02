#!/usr/bin/env python
"""Name-paired symlink layout of an official DPDD 1680x1120 split (input for build_native_set.py, dp_maps.py).

  pair_official.py --root dd_dp_dataset_png --split train --out dpdd_1680/train

DPDD names the f/22 target by its own capture number (1 or 2 above the f/4 source, e.g. source 1P0A0917,
target 1P0A0916), so pairs are formed by sorted order within the split and checked: the capture numbers
must differ by <= 3. Writes OUT/inputs/<s>.png, OUT/targets/<s>.png (-> target file, so the target CR2
stem is the link name), OUT/inputs_l|inputs_r/<s>.png (if the _l/_r views exist) and OUT/pairs.csv
(input_stem,target_stem) — the same layout as dpdd_1680/test used for the native test set.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def pair_stems(src: list[str], tgt: list[str], max_gap: int = 3) -> list[tuple[str, str]]:
    if len(src) != len(tgt):
        raise ValueError(f"{len(src)} sources vs {len(tgt)} targets")
    num = lambda s: int(re.search(r"(\d+)$", s).group(1))
    pairs = list(zip(sorted(src), sorted(tgt)))
    bad = [(s, t) for s, t in pairs if abs(num(s) - num(t)) > max_gap]
    if bad:
        raise ValueError(f"{len(bad)} pairs with capture gap > {max_gap}, e.g. {bad[:3]}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir with <split>_c/{source,target}, <split>_l/source, ...")
    ap.add_argument("--split", required=True, help="train | val | test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out)
    c = root / f"{a.split}_c"
    pairs = pair_stems([p.stem for p in (c / "source").glob("*.png")], [p.stem for p in (c / "target").glob("*.png")])
    links = {"inputs": lambda s, t: c / "source" / f"{s}.png", "targets": lambda s, t: c / "target" / f"{t}.png"}
    for v in ("l", "r"):
        if (root / f"{a.split}_{v}" / "source").is_dir():
            links[f"inputs_{v}"] = lambda s, t, v=v: root / f"{a.split}_{v}" / "source" / f"{s}_{v.upper()}.png"
    for sub, fn in links.items():
        (out / sub).mkdir(parents=True, exist_ok=True)
        for s, t in pairs:
            dst, src = out / sub / f"{s}.png", fn(s, t)
            if not src.exists():
                sys.exit(f"missing {src}")
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    with open(out / "pairs.csv", "w", newline="") as f:
        csv.writer(f).writerows(pairs)
    print(f"{a.split}: {len(pairs)} pairs -> {out} ({', '.join(links)})")


if __name__ == "__main__":
    main()
