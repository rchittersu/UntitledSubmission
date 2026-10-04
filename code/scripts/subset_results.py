#!/usr/bin/env python
"""Write copies of metrics JSON + per-image CSV restricted to a set of image names (e.g. the indoor scenes).

  subset_results.py --names indoor.txt --out DIR  RES/steps/*/metrics_dpdd3.json  RES/pipelines/*.json

Each input JSON gets DIR/<parent-or-stem>__<file>.json and a matching .csv; means/stds/n of every column found in the
CSV are recomputed on the subset (columns that only exist in the JSON, e.g. time/memory, are kept). `summarize.py` then
works on DIR/*.json unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def subset_one(json_path: Path, names: set[str], out_dir: Path) -> Path | None:
    j = json.loads(json_path.read_text())
    csv_path = Path(j.get("csv") or json_path.with_suffix(".csv"))
    if not csv_path.exists():
        return None
    rows = [r for r in csv.DictReader(open(csv_path)) if r["name"] in names]
    if not rows:
        return None
    m = j.get("metrics", {})
    for k in rows[0]:
        if k == "name":
            continue
        try:
            v = np.array([float(r[k]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            continue
        v = v[np.isfinite(v)]
        if len(v) and (k in m or isinstance(m, dict)):
            m[k] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    j["metrics"], j["n_images"] = m, len(rows)
    tag = f"{json_path.parent.name}__{json_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    j["csv"] = str(out_csv)
    out_json = out_dir / f"{tag}.json"
    out_json.write_text(json.dumps(j, indent=1))
    return out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--names", required=True, help="text file, one image stem per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = {l.strip() for l in open(a.names) if l.strip()}
    done = [subset_one(Path(f), names, Path(a.out)) for f in a.files]
    print(f"{sum(d is not None for d in done)}/{len(done)} files written to {a.out}")


if __name__ == "__main__":
    main()
