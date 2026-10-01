#!/usr/bin/env python
"""Collect metrics_*.json summaries into a Markdown table (for handoff reports / the paper).

  summarize.py RES/x1/*/metrics_dpdd.json [path.json=Label ...] [--cols psnr,pm_psnr,ssim,lpips,seam_ratio,time_s]
               [--std] [--latex]

Best value per column in bold (direction from uhdd.metrics.DIRECTION).
--ci    append a 95 % bootstrap CI of the mean over images (needs the per-image CSV next to
        the JSON, or the "csv" field of run_matrix pipeline JSONs).
--ref L second table: paired per-image differences to the row labelled L (mean, 95 % paired
        bootstrap CI, % of images where the row is better). Use this before claiming that one
        method beats another: n is small (DPDD native: 37-76 images).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uhdd.metrics import DIRECTION  # noqa: E402

import numpy as np  # noqa: E402

B = 2000  # bootstrap resamples


def per_image(r: dict, path: str) -> dict[str, dict[str, float]]:
    """{column: {image: value}} from the per-image CSV belonging to a summary JSON."""
    p = Path(r.get("csv") or Path(path).with_suffix(".csv"))
    if not p.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x):
                    out.setdefault(k, {})[row["name"]] = x
    return out


def boot_ci(v: np.ndarray, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

ARROW = {1: "↑", -1: "↓"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cols", help="comma-separated metric columns (default: all found)")
    ap.add_argument("--std", action="store_true", help="append ± std")
    ap.add_argument("--latex", action="store_true", help="LaTeX tabular rows instead of Markdown")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--ci", action="store_true", help="95%% bootstrap CI of the mean")
    ap.add_argument("--ref", help="label of the reference row for paired differences")
    a = ap.parse_args()

    runs = []
    for f in a.files:  # "metrics.json" or "metrics.json=Row label"
        path, _, label = f.partition("=")
        r = json.loads(Path(path).read_text())
        if label:
            r["label"] = label
        r["_img"] = per_image(r, path) if (a.ci or a.ref) else {}
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
        if a.ci and len(r["_img"].get(c, {})) > 1:
            lo, hi = boot_ci(np.array(list(r["_img"][c].values())))
            s += f" [{lo:.{a.digits}f}, {hi:.{a.digits}f}]"
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

    if a.ref:
        ref = next((r for r in runs if r["label"] == a.ref), None)
        if ref is None:
            sys.exit(f"--ref: no row labelled {a.ref!r}")
        print(f"\nPaired differences to **{a.ref}** (row minus reference; mean [95% paired bootstrap CI], % images better):\n")
        print("| Method | " + " | ".join(f"Δ{c}" for c in cols) + " |")
        print("|" + "---|" * (len(cols) + 1))
        for r in runs:
            if r is ref:
                continue
            cells = []
            for c in cols:
                x, y = r["_img"].get(c, {}), ref["_img"].get(c, {})
                common = sorted(set(x) & set(y))
                if len(common) < 2:
                    cells.append("–")
                    continue
                d = np.array([x[k] - y[k] for k in common])
                lo, hi = boot_ci(d)
                better = np.mean(d * (DIRECTION.get(c, 1)) > 0) * 100
                sig = "*" if lo > 0 or hi < 0 else ""
                cells.append(f"{d.mean():+.{a.digits}f}{sig} [{lo:+.{a.digits}f}, {hi:+.{a.digits}f}] {better:.0f}%")
            print(f"| {r['label']} | " + " | ".join(cells) + " |")
        print("\n`*` = CI excludes 0.")


if __name__ == "__main__":
    main()
