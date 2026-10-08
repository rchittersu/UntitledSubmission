#!/usr/bin/env python
"""Tile study: reference-based x4 upsamplers on selected 512 px native tiles (fast feedback, no full images).

  python code/scripts/tile_study.py select [--split val] [--scenes 8] [--per-bin 2]
  python code/scripts/tile_study.py prep   --anchor drbnet [--modes self,retrieved] [--features dinov2]
  python code/scripts/tile_study.py run    --anchor drbnet --model refir_seesr --ref retrieved [--set k=v ...]
  python code/scripts/tile_study.py report --anchor drbnet

select  fixed tiles (anchor-independent): per scene, 512 px native tiles on a 512 grid inside the 64 px border,
        valid (mask) >= 98 %, texture of the x4 TARGET >= the scene median (tiles with detail to recover; the blurry
        input would reject every defocused tile); `per-bin` tiles per DP defocus bin
        (b0 focal plane < 0.4 <= b1 < 1.5 <= b2 < 3.5 <= b3, DP px at 1680). -> tilestudy/<split>/tiles.json
prep    crops of target / input / mask / anchor per tile and the references (uhdd/refsr.py): `self` (native input at
        the tile), `retrieved` (mosaic of the 16 best in-focus exemplars from the DINOv2 memory of the whole anchor,
        tile region excluded). -> tilestudy/<split>/{crops,refs}/
run     one model (configs/refsr.yaml) x one reference mode on every tile (--shard i/n for several GPUs).
        -> tilestudy/<split>/runs/<anchor>/<model>[+overrides]__<ref>/<tile>.png + meta.json
report  per run, raw and with the anchor lock (down4(y) = anchor, as our method): PSNR / SSIM (mask) and LPIPS /
        DISTS per tile, means per bin and the gain over bicubic of the same anchor; contact sheets per tile.
        -> tilestudy/<split>/report_<anchor>.md, metrics_<anchor>.csv, sheets/<anchor>/<tile>.jpg
Everything lives under $UHDD_RESULTS/dpdd/tilestudy/ (or --root). Development on val; the test set only at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from uhdd import layout  # noqa: E402
from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.io import read_image, read_mask, write_image  # noqa: E402

BINS = (0.4, 1.5, 3.5)
DEBLUR = {"input": "identity", "drbnet": "drbnet_single", "bokehlicious": "bokehlicious_deblur",
          "restormer": "restormer_dpdd", "lakdnet": "lakdnet_dpdd_l", "ifan": "ifan"}


# ---------------------------------------------------------------- helpers
def to_t(img: np.ndarray) -> torch.Tensor:
    m = 65535.0 if img.dtype == np.uint16 else 255.0
    return torch.from_numpy(img.astype(np.float32) / m).permute(2, 0, 1)[None]


def to_u16(t: torch.Tensor) -> np.ndarray:
    return (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 65535 + 0.5).astype(np.uint16)


def bin_of(b: float) -> int:
    return int(np.searchsorted(BINS, b, side="right"))


def anchor_tag(a: str) -> str:
    return layout.tag(DEBLUR.get(a, a))


class Study:
    def __init__(self, a):
        self.split = a.split
        self.root = Path(a.root) if a.root else layout.root() / "tilestudy" / a.split
        self.native = layout.SPLITS[a.split]["native"]
        self.x4 = {"test": "ours_x4", "train": "train_x4", "val": "val_x4"}[a.split]

    @property
    def tiles(self) -> list[dict]:
        return json.loads((self.root / "tiles.json").read_text())["tiles"]

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def dirs(self, anchor: str | None = None) -> dict:
        d = {"inputs": layout.input_dir(self.native, "inputs"), "targets": layout.input_dir(self.native, "targets"),
             "masks": layout.input_dir(self.native, "masks"), "x4": layout.input_dir(self.x4, "inputs"),
             "x4_targets": layout.input_dir(self.x4, "targets"),
             "dp": layout.dp_maps_dir(self.split)}
        if anchor:
            d["anchor"] = layout.deblur_dir(self.x4, anchor_tag(anchor))
        return d


# ---------------------------------------------------------------- select
def cmd_select(a, st: Study):
    d = st.dirs()
    names = sorted(p.stem for p in Path(d["x4"]).glob("*.png"))
    if not names:
        sys.exit(f"no inputs in {d['x4']}")
    rng = np.random.default_rng(a.seed)
    if a.names:
        names = a.names.split(",")
    elif a.scenes and a.scenes < len(names):
        names = sorted(rng.choice(names, a.scenes, replace=False).tolist())
    T4, B4 = a.tile // 4, a.border // 4
    out = []
    for n in names:
        x4 = read_image(d["x4_targets"] / f"{n}.png")     # texture of the target: where there is detail to recover
        H4, W4 = x4.shape[:2]
        blur = load_blur_map(d["dp"], n, hw=(H4, W4), win=15)
        if blur is None:
            print(f"skip {n}: no DP map")
            continue
        mp = d["masks"] / f"{n}.png"
        mask = cv2.resize(read_mask(mp).astype(np.float32), (W4, H4), interpolation=cv2.INTER_AREA) if mp.exists() else np.ones((H4, W4), np.float32)
        luma = to_t(x4).mean(1, keepdim=True)
        hp = (luma - F.avg_pool2d(luma, 5, 1, 2, count_include_pad=False))[0, 0].numpy()
        cand = []
        for y in range(B4, H4 - B4 - T4 + 1, T4):
            for x in range(B4, W4 - B4 - T4 + 1, T4):
                if mask[y:y + T4, x:x + T4].mean() < 0.98:
                    continue
                cand.append((y, x, float(blur[y:y + T4, x:x + T4].mean()), float(hp[y:y + T4, x:x + T4].std())))
        if not cand:
            continue
        med = np.median([c[3] for c in cand])
        for k in range(4):
            pool = [c for c in cand if bin_of(c[2]) == k and c[3] >= med]
            for i in rng.permutation(len(pool))[:a.per_bin]:
                y, x, b, t = pool[i]
                out.append({"id": f"{n}_y{4 * y}_x{4 * x}", "scene": n, "y": 4 * y, "x": 4 * x, "size": a.tile,
                            "bin": k, "blur": round(b, 3), "texture": round(t, 5)})
    st.root.mkdir(parents=True, exist_ok=True)
    (st.root / "tiles.json").write_text(json.dumps({"split": st.split, "tile": a.tile, "seed": a.seed, "tiles": out,
                                                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    per = [sum(t["bin"] == k for t in out) for k in range(4)]
    print(f"{len(out)} tiles from {len(set(t['scene'] for t in out))} scenes; per bin b0..b3: {per} -> {st.root / 'tiles.json'}")


# ---------------------------------------------------------------- prep
def cmd_prep(a, st: Study):
    from uhdd import refsr
    d = st.dirs(a.anchor)
    at = anchor_tag(a.anchor)
    modes = a.modes.split(",")
    feats = None
    if "retrieved" in modes:
        from uhdd.train import memory as mm
        import os
        hub = a.dino_hub or (str(Path(os.environ["UHDD_WEIGHTS"]) / "vosr" / "torch_cache") if os.environ.get("UHDD_WEIGHTS") else None)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        feats = mm.Features(a.features, "dinov2_vitl14", hub if a.features == "dinov2" else None, dev)
    info = {}
    by_scene: dict[str, list[dict]] = {}
    for t in st.tiles:
        by_scene.setdefault(t["scene"], []).append(t)
    for n, tiles in by_scene.items():
        x = read_image(d["inputs"] / f"{n}.png")
        y = read_image(d["targets"] / f"{n}.png")
        mp = d["masks"] / f"{n}.png"
        m = read_mask(mp) if mp.exists() else np.ones(x.shape[:2], bool)
        an = read_image(d["anchor"] / f"{n}.png")
        xt = to_t(x)
        for t in tiles:
            ty, tx, s = t["y"], t["x"], t["size"]
            for kind, img in (("target", y), ("input", x)):
                f = st.p("crops", kind, f"{t['id']}.png")
                if not f.exists():
                    f.parent.mkdir(parents=True, exist_ok=True)
                    write_image(f, img[ty:ty + s, tx:tx + s])
            f = st.p("crops", "mask", f"{t['id']}.png")
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(f), (m[ty:ty + s, tx:tx + s] > 0).astype(np.uint8) * 255)
            f = st.p("crops", f"anchor@{at}", f"{t['id']}.png")
            f.parent.mkdir(parents=True, exist_ok=True)
            write_image(f, an[ty // 4:(ty + s) // 4, tx // 4:(tx + s) // 4])
            if "self" in modes:
                f = st.p("refs", "self", f"{t['id']}.png")
                if not f.exists():
                    f.parent.mkdir(parents=True, exist_ok=True)
                    write_image(f, to_u16(refsr.self_ref(xt, (ty, tx, s))))
        if "retrieved" in modes:
            from uhdd.train import memory as mm
            H4, W4 = an.shape[:2]
            blur = torch.from_numpy(load_blur_map(d["dp"], n, hw=(H4, W4), win=15))[None, None]
            at_t = to_t(an).to(feats.device)
            mem = mm.build_memory(at_t, blur.to(feats.device), feats, (1.0, 0.75, 0.5), value_px=a.size // a.grid)
            idx, sc = mm.retrieve(mem, 32)
            for t in tiles:
                ref, inf = refsr.retrieved_ref(xt, mem, idx, sc, (t["y"] // 4, t["x"] // 4, t["size"] // 4),
                                               size=a.size, grid=a.grid)
                f = st.p("refs", f"retrieved@{at}", f"{t['id']}.png")
                f.parent.mkdir(parents=True, exist_ok=True)
                write_image(f, to_u16(ref))
                info[t["id"]] = inf
        print(f"{n}: {len(tiles)} tiles", flush=True)
    if info:
        st.p("refs", f"retrieved@{at}", "info.json").write_text(json.dumps(info))


# ---------------------------------------------------------------- run
def parse_sets(sets):
    import yaml
    return {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in sets)}


def run_name(model: str, ref: str, sets: list[str]) -> str:
    return model + ("+" + "+".join(s.replace("=", "-") for s in sets) if sets else "") + f"__{ref}"


def load_ref(st: Study, ref: str, at: str, tid: str):
    if ref == "none":
        return None
    f = st.p("refs", "self" if ref == "self" else f"retrieved@{at}", f"{tid}.png")
    return to_t(read_image(f))


def cmd_run(a, st: Study):
    from uhdd import refsr
    at = anchor_tag(a.anchor)
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    model = refsr.load(a.model, dev, parse_sets(a.set))
    out = st.p("runs", at, run_name(a.model, a.ref, a.set))
    out.mkdir(parents=True, exist_ok=True)
    tiles = st.tiles
    if a.tiles:
        keep = set(a.tiles.split(","))
        tiles = [t for t in tiles if t["id"] in keep]
    i, n = map(int, a.shard.split("/"))
    tiles = tiles[i::n][: a.limit or None]
    times = []
    for t in tiles:
        f = out / f"{t['id']}.png"
        if f.exists() and not a.overwrite:
            continue
        anc = to_t(read_image(st.p("crops", f"anchor@{at}", f"{t['id']}.png")))
        ref = load_ref(st, a.ref, at, t["id"])
        t0 = time.time()
        y = model(anc, ref)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        write_image(f, to_u16(y.float().cpu()))
        print(f"{t['id']} bin {t['bin']} {times[-1]:.1f}s", flush=True)
    meta = {"model": a.model, "ref": a.ref, "anchor": at, "set": a.set, "n": len(list(out.glob("*.png"))),
            "time_s_mean": round(float(np.mean(times)), 2) if times else None, "device": dev,
            "load_report": getattr(model, "load_report", None)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- report
def lock(y: torch.Tensor, anc: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """Anchor lock alternated with [0, 1] clamping (POCS); the clamp breaks the lock a little each round, so it
    needs ~16-32 rounds to settle (4 rounds leave ~1e-2), cheap on a tile."""
    from uhdd.net.ours import anchor_lock
    for _ in range(iters):
        y = anchor_lock(y, anc).clamp(0, 1)
    return y


def cmd_report(a, st: Study):
    from uhdd.metrics.fidelity import psnr, ssim
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import importlib.util
    perc = ("lpips", "dists") if importlib.util.find_spec("pyiqa") and not a.no_perceptual else ()
    if perc:
        from uhdd.metrics.iqa import full_reference
    at = anchor_tag(a.anchor)
    runs = sorted(p for p in st.p("runs", at).glob("*") if p.is_dir()) if st.p("runs", at).exists() else []
    rows = []
    sheets = st.p("sheets", at)
    sheets.mkdir(parents=True, exist_ok=True)
    for t in st.tiles:
        tid = t["id"]
        gt = to_t(read_image(st.p("crops", "target", f"{tid}.png"))).to(dev)
        inp = to_t(read_image(st.p("crops", "input", f"{tid}.png"))).to(dev)
        m = torch.from_numpy(cv2.imread(str(st.p("crops", "mask", f"{tid}.png")), 0) > 0)[None, None].float().to(dev)
        anc = to_t(read_image(st.p("crops", f"anchor@{at}", f"{tid}.png"))).to(dev)
        bic = F.interpolate(anc, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        cands = [("input", "-", inp), ("bicubic", "-", bic)]
        for r in runs:
            f = r / f"{tid}.png"
            if f.exists():
                y = to_t(read_image(f)).to(dev)
                cands += [(r.name, "raw", y), (r.name, "lock", lock(y, anc))]
        for name, var, y in cands:
            row = {"tile": tid, "scene": t["scene"], "bin": t["bin"], "run": name, "variant": var,
                   "psnr": psnr(y, gt, m), "ssim": ssim(y, gt, m)}
            for k in perc:
                row[k] = full_reference(k, y, gt, tile=t["size"])
            rows.append(row)
        if not a.no_sheets:
            tiles = [("target", gt), ("input", inp), ("bicubic", bic)] + [(n, y) for n, v, y in cands[2:] if v == "raw"]
            ims = []
            for lab, y in tiles:
                im = cv2.cvtColor((y[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                im = cv2.resize(im, (a.sheet_px, a.sheet_px), interpolation=cv2.INTER_AREA) if a.sheet_px != im.shape[0] else im
                cv2.putText(im, lab[:40], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                ims.append(im)
            cols = min(len(ims), a.sheet_cols)
            while len(ims) % cols:
                ims.append(np.zeros_like(ims[0]))
            grid = np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])
            cv2.imwrite(str(sheets / f"{tid}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    keys = ["tile", "scene", "bin", "run", "variant", "psnr", "ssim", *perc]
    with open(st.p(f"metrics_{at}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    st.p(f"report_{at}.md").write_text(summary(rows, perc, at))
    print(st.p(f"report_{at}.md").read_text())


def summary(rows: list[dict], perc, at: str) -> str:
    """Per run / variant: mean PSNR gain over bicubic of the same tile per bin, and absolute metrics (all tiles)."""
    bic = {r["tile"]: r for r in rows if r["run"] == "bicubic"}
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["run"], r["variant"]), []).append(r)
    head = "| run | variant | n | ΔPSNR b0 | b1 | b2 | b3 | PSNR | SSIM |" + "".join(f" {k.upper()} |" for k in perc)
    lines = [f"# Tile study — anchor {at}", "", "ΔPSNR = mean paired gain over bicubic of the same anchor tile (dB, ↑); "
             "PSNR / SSIM ↑, LPIPS / DISTS ↓, means over all tiles.", "", head,
             "|" + "---|" * (head.count("|") - 1)]
    for (run, var), rs in groups.items():
        dd = []
        for k in range(4):
            v = [r["psnr"] - bic[r["tile"]]["psnr"] for r in rs if r["bin"] == k]
            dd.append(f"{np.mean(v):+.2f}" if v else "–")
        pm = "".join(f" {np.mean([r[k] for r in rs]):.4f} |" for k in perc)
        lines.append(f"| {run} | {var} | {len(rs)} | " + " | ".join(dd) + f" | {np.mean([r['psnr'] for r in rs]):.2f} | "
                     f"{np.mean([r['ssim'] for r in rs]):.4f} |{pm}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "prep", "run", "report"])
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--root", help="default: $UHDD_RESULTS/dpdd/tilestudy/<split>")
    ap.add_argument("--anchor", default="drbnet", help="short name (drbnet, bokehlicious, ...) or registry name")
    # select
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--names", help="explicit scene list (comma-separated)")
    ap.add_argument("--per-bin", type=int, default=2)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--border", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # prep
    ap.add_argument("--modes", default="self,retrieved")
    ap.add_argument("--features", default="dinov2", choices=["dinov2", "pixels"])
    ap.add_argument("--dino-hub")
    ap.add_argument("--size", type=int, default=512, help="reference size (native px)")
    ap.add_argument("--grid", type=int, default=4, help="retrieved mosaic: grid x grid exemplars")
    # run
    ap.add_argument("--model")
    ap.add_argument("--ref", default="retrieved", choices=["none", "self", "retrieved"])
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="model spec overrides, e.g. steps=20")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tiles", help="only these tile ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    # report
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--no-perceptual", action="store_true", help="skip LPIPS / DISTS (pyiqa)")
    ap.add_argument("--sheet-px", type=int, default=256)
    ap.add_argument("--sheet-cols", type=int, default=6)
    a = ap.parse_args(argv)
    st = Study(a)
    if a.cmd == "run" and not a.model:
        ap.error("run needs --model")
    {"select": cmd_select, "prep": cmd_prep, "run": cmd_run, "report": cmd_report}[a.cmd](a, st)
    return st


if __name__ == "__main__":
    main()
