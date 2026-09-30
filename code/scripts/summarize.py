#!/usr/bin/env python
"""Collect metrics_*.json summaries into a Markdown table (for handoff reports / the paper).

  summarize.py RES/x1/*/metrics_dpdd.json [path.json=Label ...] [--cols psnr,pm_psnr,ssim,lpips,seam_ratio,time_s]
               [--std] [--latex]

Best value per column in bold (direction from uhdd.metrics.DIRECTION).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uhdd.metrics import DIRECTION  # noqa: E402

ARROW = {1: "↑", -1: "↓"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cols", help="comma-separated metric columns (default: all found)")
    ap.add_argument("--std", action="store_true", help="append ± std")
    ap.add_argument("--latex", action="store_true", help="LaTeX tabular rows instead of Markdown")
    ap.add_argument("--digits", type=int, default=3)
    a = ap.parse_args()

    runs = []
    for f in a.files:  # "metrics.json" or "metrics.json=Row label"
        path, _, label = f.partition("=")
        r = json.loads(Path(path).read_text())
        if label:
            r["label"] = label
        runs.append(r)
    cols = a.cols.split(",") if a.cols else list(dict.fromkeys(k for r in runs for k in r["metrics"]))
    best = {}
    for c in cols:
        vals = [r["metrics"][c]["mean"] for r in runs if c in r["metrics"]]
        d = DIRECTION.get(c)
        if vals and d:
            best[c] = max(vals) if d > 0 else min(vals)

    def cell(r, c):
        if c not in r["metrics"]:
            return "–"
        m = r["metrics"][c]
        s = f"{m['mean']:.{a.digits}f}" + (f" ± {m['std']:.{a.digits}f}" if a.std else "")
        if c in best and m["mean"] == best[c] and len(runs) > 1:
            s = f"\\textbf{{{s}}}" if a.latex else f"**{s}**"
        return s

    head = ["Method", "N"] + [f"{c} {ARROW.get(DIRECTION.get(c), '')}".strip() for c in cols]
    body = [[r["label"], str(r["n_images"])] + [cell(r, c) for c in cols] for r in runs]
    if a.latex:
        print(" & ".join(head) + r" \\ \midrule")
        for row in body:
            print(" & ".join(row) + r" \\")
    else:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in body:
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
