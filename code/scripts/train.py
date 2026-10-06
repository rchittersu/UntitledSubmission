#!/usr/bin/env python
"""Train the native upsampler (plan/method_plan.md §4, docs/training.md).

  torchrun --nproc_per_node 8 code/scripts/train.py --config code/configs/train/v0.yaml
  torchrun --nproc_per_node 8 code/scripts/train.py --config code/configs/train/v0_oracle.yaml
  python code/scripts/train.py --config ... --set optim.steps=200 data.anchors.sim=1.0     # overrides

Runs go to $UHDD_RESULTS/dpdd/train/<name>/ (or --out): config.yaml, log.csv (+ TensorBoard events if available),
val.csv, ckpt_<step>.pt (model, EMA, optimizer, step), last.pt, model_ema.pt (EMA weights + model config; what the
registry entry `ours_*` loads). Resumes from last.pt automatically. Mixed precision bf16 on CUDA.
Validation (rank 0, EMA weights, fixed tiles of the val cache): PSNR on the validity mask, PSNR in defocused regions,
x4 PSNR, and the same for the bicubic-upsampled anchor (reference), every `val.every` steps.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402

from uhdd.models import _expand  # noqa: E402
from uhdd.net.ours import build_model, down4, up4  # noqa: E402
from uhdd.train.data import TileDataset  # noqa: E402
from uhdd.train.losses import Loss  # noqa: E402


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, sets: list[str]) -> dict:
    """YAML with `base:` inheritance, then the --set overrides, then environment variables."""
    def read(p: str) -> dict:
        c = yaml.safe_load(Path(p).read_text())
        return merge(read(str(Path(p).parent / c.pop("base"))), c) if "base" in c else c
    cfg = read(path)
    for s in sets:
        k, v = s.split("=", 1)
        d = cfg
        *ks, last = k.split(".")
        for kk in ks:
            d = d.setdefault(kk, {})
        d[last] = yaml.safe_load(v)
    return expand(cfg)


def expand(v):
    """${UHDD_RESULTS} / ${UHDD_WEIGHTS} / ... in config strings; an unset variable is an error, not a silent path."""
    if isinstance(v, dict):
        return {k: expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand(x) for x in v]
    if isinstance(v, str) and "$" in v:
        e = os.path.expandvars(v)
        if "$" in e:
            raise KeyError(f"unset environment variable in config value {v!r}")
        return e
    return v


def psnr(a, b, m=None):
    e = (a - b) ** 2
    mse = (e * m).sum() / (m.sum() * a.shape[1] + 1e-8) if m is not None else e.mean()
    return float(10 * torch.log10(1 / mse.clamp_min(1e-10)))


def lr_at(step: int, o: dict) -> float:
    if step < o.get("warmup", 0):
        return (step + 1) / o["warmup"]
    t = (step - o.get("warmup", 0)) / max(1, o["steps"] - o.get("warmup", 0))
    lo = o.get("min_lr", 0) / o["lr"]
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def to(b: dict, dev) -> dict:
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def validate(model, loader, dev, amp) -> dict:
    model.eval()
    rows = []
    for b in loader:
        b = to(b, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            p = model(b["a"], b["x"], b["d"], b["ex"], b["tok_map"], b["tok_scores"]).float().clamp(0, 1)
        base = up4(b["a"]).clamp(0, 1)
        for i in range(p.shape[0]):
            m, r = b["m"][i:i + 1], b["m"][i:i + 1] * b["regime"][i:i + 1]
            rows.append({"psnr": psnr(p[i:i + 1], b["y"][i:i + 1], m), "psnr_bicubic": psnr(base[i:i + 1], b["y"][i:i + 1], m),
                         "psnr_defocus": psnr(p[i:i + 1], b["y"][i:i + 1], r) if r.sum() > 0 else float("nan"),
                         "psnr_defocus_bicubic": psnr(base[i:i + 1], b["y"][i:i + 1], r) if r.sum() > 0 else float("nan"),
                         "psnr_x4": psnr(down4(p[i:i + 1]), down4(b["y"][i:i + 1]))})
    model.train()
    out = {}
    for k in rows[0] if rows else []:
        v = torch.tensor([r[k] for r in rows])
        v = v[torch.isfinite(v)]
        out[k] = round(float(v.mean()), 4) if len(v) else None
    out["n"] = len(rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="dotted overrides, e.g. optim.steps=200 (repeatable)")
    ap.add_argument("--out")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    cfg = _expand(load_config(a.config, a.set))

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    cuda = torch.cuda.is_available() and not a.cpu
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = cuda and cfg.get("amp", True)
    out = Path(a.out) if a.out else Path(os.environ["UHDD_RESULTS"]) / "dpdd" / "train" / cfg["name"]
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    torch.manual_seed(cfg.get("seed", 0) + rank)
    model = build_model(cfg["model"]).to(dev)
    o = cfg["optim"]
    bb = [p for n, p in model.named_parameters() if n.startswith("sr.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("sr.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": o.get("lr_backbone", o["lr"])}, {"params": rest, "lr": o["lr"]}],
                            betas=tuple(o.get("betas", (0.9, 0.99))), weight_decay=o.get("wd", 0.0))
    base_lr = [g["lr"] for g in opt.param_groups]
    ema = copy.deepcopy(model).eval()
    if rank == 0:
        n_all = sum(p.numel() for p in model.parameters())
        n_bb = sum(p.numel() for p in bb)
        print(f"params: {n_all / 1e6:.2f} M (backbone {n_bb / 1e6:.2f} M, new {(n_all - n_bb) / 1e6:.2f} M)", flush=True)
    for p in ema.parameters():
        p.requires_grad_(False)
    step = 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if rank == 0:
            print(f"resumed from {last} at step {step}", flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if cuda else None,
                                                    find_unused_parameters=cfg.get("find_unused", False)) if world > 1 else model
    crit = Loss(cfg.get("loss", {})).to(dev)

    dl = torch.utils.data.DataLoader(TileDataset({**cfg["data"], "seed": cfg["data"].get("seed", 0) + step}, rank, world),
                                     batch_size=cfg.get("batch", 4), num_workers=cfg.get("workers", 4),
                                     pin_memory=cuda, persistent_workers=cfg.get("workers", 4) > 0)
    vcfg = cfg.get("val", {})
    vl = None
    if rank == 0 and vcfg.get("tiles", 0):
        vd = TileDataset({**cfg["data"], "cache": vcfg["cache"], "names": vcfg.get("names"), "augment": False,
                          "oracle": vcfg.get("oracle", cfg["data"].get("oracle", False)), "drop_colocated": 0.0,
                          "anchors": vcfg.get("anchors", {"bokehlicious_deblur": 1.0}), "seed": 7}, fixed=vcfg["tiles"])
        vl = torch.utils.data.DataLoader(vd, batch_size=cfg.get("batch", 4), num_workers=min(2, cfg.get("workers", 4)))

    tb = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out / "tb"))
            print("logging: CSV + TensorBoard", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"logging: CSV only (TensorBoard unavailable: {type(e).__name__})", flush=True)
    logf = open(out / "log.csv", "a", newline="") if rank == 0 else None
    logw = None

    def run_val(step: int) -> None:
        v = {"step": step, **validate(ema, vl, dev, amp)}
        new = not (out / "val.csv").exists()
        with open(out / "val.csv", "a", newline="") as fv:
            w = csv.DictWriter(fv, fieldnames=list(v))
            if new:
                w.writeheader()
            w.writerow(v)
        if tb:
            for k, x in v.items():
                if k not in ("step", "n") and x is not None:
                    tb.add_scalar(f"val/{k}", x, step)
        print("val", json.dumps(v), flush=True)

    # step 0: the untrained model is the locked backbone (HAT-L x4 + anchor lock) -> its val row is the reference
    if rank == 0 and vl is not None and step == 0 and vcfg.get("at_start", True):
        run_val(0)

    model.train()
    t0, s0, it = time.time(), step, iter(dl)
    while step < o["steps"]:
        b = to(next(it), dev)
        f = lr_at(step, o)
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            pred = net(b["a"], b["x"], b["d"], b["ex"], b["tok_map"], b["tok_scores"])
        loss, logs = crit(pred.float(), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("clip", 1.0))
        opt.step()
        with torch.no_grad():
            d = o.get("ema", 0.999) if step > o.get("warmup", 0) else 0.0
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        if rank == 0 and (step % cfg.get("log_every", 50) == 0 or step == 1):
            row = {"step": step, "lr": round(opt.param_groups[1]["lr"], 8), "grad_norm": round(float(gn), 4),
                   "it_s": round((step - s0) / max(time.time() - t0, 1e-6), 3),
                   "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else 0.0, **{k: round(float(v), 6) for k, v in logs.items()}}
            if logw is None:
                logw = csv.DictWriter(logf, fieldnames=list(row) + ["cx"], extrasaction="ignore")
                if logf.tell() == 0:
                    logw.writeheader()
            logw.writerow(row)
            logf.flush()
            if tb:
                for k, v in row.items():
                    if k != "step":
                        tb.add_scalar(f"train/{k}", v, step)
            print(json.dumps(row), flush=True)
        if rank == 0 and vl is not None and (step % vcfg.get("every", 2000) == 0 or step == o["steps"]):
            run_val(step)
        if rank == 0 and (step % cfg.get("ckpt_every", 5000) == 0 or step == o["steps"]):
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}
            torch.save(ck, out / f"ckpt_{step}.pt")
            torch.save(ck, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", last)
            torch.save({"model": ema.state_dict(), "cfg": cfg["model"], "step": step}, out / "model_ema.pt")
            for p in sorted(out.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[: -cfg.get("keep", 3)]:
                p.unlink()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"done: {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
