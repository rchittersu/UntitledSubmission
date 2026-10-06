#!/usr/bin/env python
"""Set up and verify the inputs of the results root (uhdd/layout.py): $UHDD_RESULTS/dpdd/inputs/.

Links every input source into the root (symlinks, no copies) and checks, before anything is run on it:
  - every source has the expected number of scenes (76) and inputs / targets / masks / DP maps carry the SAME names;
  - names agree ACROSS sources (ours_x1, ours_x4, official_x4, ...);
  - sizes: ours_x4 = ours_x1 / 4, official_x4 = ours_x4, targets = inputs, per scene;
  - content: same name = same scene. official_x4 vs ours_x4 per-scene PSNR (expected ~39 dB, the calibration of
    our renderings; < --min-psnr flags a mismatch), and a nearest-thumbnail check over all scenes that would catch
    a permutation of names.
Writes inputs/manifest.json (all checks, ok flag). launch.py refuses to run on a source that is not ok.

Legacy roots ($UHDD_RESULTS/dpdd_v2, dpdd_official_x4, dpdd_p1) are never written; --protect-legacy removes write
permission from them (undo: chmod -R u+w <dir>).

  python code/scripts/setup_inputs.py [--sources ours_x1,ours_x4,official_x4] [--expect 76] [--protect-legacy]
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uhdd import layout  # noqa: E402
from uhdd.io import list_images  # noqa: E402

THUMB = (105, 70)   # (w, h) thumbnail for the permutation check


def names_report(sets: dict[str, set[str]], ref: str) -> list[str]:
    """Problems with name sets compared to sets[ref] (empty list = identical)."""
    out = []
    for k, s in sets.items():
        if k == ref:
            continue
        miss, extra = sorted(sets[ref] - s), sorted(s - sets[ref])
        if miss:
            out.append(f"{k}: missing {len(miss)} vs {ref}: {miss[:8]}{' ...' if len(miss) > 8 else ''}")
        if extra:
            out.append(f"{k}: {len(extra)} not in {ref}: {extra[:8]}{' ...' if len(extra) > 8 else ''}")
    return out


def nearest_mismatches(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    """Names in `a` whose nearest thumbnail in `b` (MSE) carries a different name: [(name, nearest name)]."""
    names = sorted(set(a) & set(b))
    B = np.stack([b[n].ravel() for n in names])
    out = []
    for n in names:
        d = ((B - a[n].ravel()[None]) ** 2).mean(1)
        j = names[int(d.argmin())]
        if j != n:
            out.append((n, j))
    return out


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    m = float(((x - y) ** 2).mean())
    return 99.0 if m == 0 else 10 * np.log10(1.0 / m)


def read01(p: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(p)
    return im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)


def size_of(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size[1], im.size[0]


def link(dst: Path, target: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a symlink; not touching it")
    dst.symlink_to(target.resolve(), target_is_directory=True)


def protect(d: Path) -> None:
    for base, dirs, files in os.walk(d):
        for f in files + dirs:
            p = Path(base) / f
            if not p.is_symlink():
                p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    d.chmod(d.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="ours_x1,ours_x4,official_x4",
                    help="sources to (re)check; train / val: train_x1,train_x4 / val_x1,val_x4")
    ap.add_argument("--expect", type=int, default=None, help="scenes per source (default: the split's count)")
    ap.add_argument("--min-psnr", type=float, default=30.0, help="official vs ours x4, per scene (dB)")
    ap.add_argument("--protect-legacy", action="store_true")
    a = ap.parse_args()

    data, root = layout.data_root(), layout.root()
    srcs = a.sources.split(",")
    problems: list[str] = []
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": layout.ROOT_NAME, "sources": {}, "checks": {}}
    splits = sorted({layout.split_of(s) for s in srcs})
    for sp in splits:    # the native source of a split is always (re)checked with it
        if layout.SPLITS[sp]["native"] not in srcs:
            srcs.insert(0, layout.SPLITS[sp]["native"])

    def flag(src: str, msg: str) -> None:   # every problem belongs to a source: launch.py refuses that source
        rec = man["sources"].get(src)
        (rec["problems"] if rec is not None else problems).append(msg)

    # 1. links + per-source name/size checks
    names: dict[str, set[str]] = {}
    files: dict[str, dict[str, dict[str, Path]]] = {}
    for s in srcs:
        spec = layout.source(s)
        rec = {"scale": spec["scale"], "data": {}, "problems": []}
        files[s] = {}
        for kind in ("inputs", "targets", "masks"):
            if not spec.get(kind):
                continue
            d = data / spec[kind]
            rec["data"][kind] = spec[kind]          # relative to $UHDD_DATA
            if not d.is_dir():
                rec["problems"].append(f"{kind}: missing ({spec[kind]})")
                continue
            link(layout.input_dir(s, kind), d)
            files[s][kind] = list_images(d)
        kinds = {f"{s}/{k}": set(v) for k, v in files[s].items()}
        if f"{s}/inputs" in kinds:
            rec["problems"] += names_report(kinds, f"{s}/inputs")
            names[s] = kinds[f"{s}/inputs"]
            rec["n"] = len(names[s])
            expect = a.expect or layout.SPLITS[layout.split_of(s)]["n"]
            if rec["n"] != expect:
                rec["problems"].append(f"{rec['n']} scenes, expected {expect}")
            sizes = {n: size_of(p) for n, p in files[s]["inputs"].items()}
            rec["sizes"] = sorted({f"{w}x{h}" for h, w in sizes.values()})
            for kind in ("targets", "masks"):
                for n, p in files[s].get(kind, {}).items():
                    if n in sizes and size_of(p) != sizes[n]:
                        rec["problems"].append(f"{kind}/{n}: size {size_of(p)} != input {sizes[n]}")
            rec["_hw"] = sizes
        man["sources"][s] = rec

    for sp in splits:
        nat = layout.SPLITS[sp]["native"]
        dp = data / layout.SPLITS[sp]["dp_maps"]
        if dp.is_dir():
            link(layout.dp_maps_dir(sp), dp)
            have = {p.name[: -len("_disp.png")] for p in dp.glob("*_disp.png")}
            miss = sorted(names.get(nat, set()) - have)
            man["checks"][f"dp_maps_{sp}"] = {"n": len(have), "missing": miss}
            if miss:
                flag(nat, f"dp_maps: missing for {miss}")
        else:
            flag(nat, f"dp_maps: missing ({layout.SPLITS[sp]['dp_maps']})")

        # 2. names and sizes across the sources of this split
        if nat in names:
            group = {k: v for k, v in names.items() if layout.split_of(k) == sp}
            cross = names_report(group, nat)
            man["checks"][f"names_across_sources_{sp}"] = cross or "identical"
            for c in cross:
                flag(c.split(":")[0], c)
            hw1 = man["sources"][nat]["_hw"]
            for s_ in group:
                if s_ == nat or "_hw" not in man["sources"][s_]:
                    continue
                sc = layout.source(s_)["scale"]
                bad = [n for n, (h, w) in man["sources"][s_]["_hw"].items()
                       if n in hw1 and (h, w) != (hw1[n][0] // sc, hw1[n][1] // sc)]
                if bad:
                    flag(s_, f"{len(bad)} scenes not native/{sc} in size: {bad[:8]}")

    # 3. content: official vs ours at x4 (same name = same scene)
    if {"official_x4", "ours_x4"} <= set(files):
        import cv2
        for kind in ("inputs", "targets"):
            fo, fu = files["official_x4"].get(kind, {}), files["ours_x4"].get(kind, {})
            common = sorted(set(fo) & set(fu))
            if not common:
                continue
            per, th_o, th_u = {}, {}, {}
            for n in common:
                x, y = read01(fo[n]), read01(fu[n])
                if x.shape == y.shape:
                    per[n] = round(psnr(x, y), 2)
                th_o[n] = cv2.resize(x, THUMB, interpolation=cv2.INTER_AREA)
                th_u[n] = cv2.resize(y, THUMB, interpolation=cv2.INTER_AREA)
            low = {n: v for n, v in per.items() if v < a.min_psnr}
            perm = nearest_mismatches(th_o, th_u)
            v = np.array(list(per.values()))
            man["checks"][f"official_vs_ours_x4_{kind}"] = {
                "psnr_median": float(np.median(v)) if len(v) else None, "psnr_min": float(v.min()) if len(v) else None,
                "below_min": low, "nearest_is_other_scene": perm, "per_scene": per}
            if low and kind == "inputs":   # targets: informational only (our targets are registered to the input, the official ones are not)
                flag("official_x4", f"official vs ours x4 {kind}: {len(low)} scenes below {a.min_psnr} dB: {sorted(low)[:8]}")
            if perm:
                flag("official_x4", f"official vs ours x4 {kind}: nearest scene has another name for {perm[:8]}")

    # 4. legacy roots
    res = root.parent
    man["legacy_untouched"] = [n for n in layout.LEGACY if (res / n).exists()]
    if a.protect_legacy:
        for n in man["legacy_untouched"]:
            print(f"protecting {res / n} (read-only)")
            protect(res / n)
        man["legacy_protected"] = True

    for s, rec in man["sources"].items():
        rec.pop("_hw", None)
        rec["ok"] = not rec["problems"]
        problems += [f"{s}: {p}" for p in rec["problems"]]
    # merge with sources checked earlier (e.g. test, then train / val): re-checked sources replace their entry
    mpath = root / "inputs" / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        for s, rec in old.get("sources", {}).items():
            if s not in man["sources"]:
                man["sources"][s] = rec
                problems += [f"{s}: {p}" for p in rec.get("problems", [])]
        for k, v in old.get("checks", {}).items():
            man["checks"].setdefault(k, v)
    man["problems"] = problems
    man["ok"] = not problems
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, indent=1))

    for s, rec in man["sources"].items():
        print(f"{s:12s} n={rec.get('n', 0):3d} sizes={rec.get('sizes')} {'ok' if rec['ok'] else 'PROBLEMS'}")
    for k in ("inputs", "targets"):
        c = man["checks"].get(f"official_vs_ours_x4_{k}")
        if c:
            print(f"official vs ours x4 {k}: PSNR median {c['psnr_median']:.2f} dB, min {c['psnr_min']:.2f} dB, "
                  f"nearest-scene mismatches {len(c['nearest_is_other_scene'])}")
    print("\n".join(["", "PROBLEMS:"] + problems) if problems else "\nall checks passed")
    print(f"manifest: {root / 'inputs' / 'manifest.json'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
