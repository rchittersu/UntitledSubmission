#!/usr/bin/env python
"""Human-readable progress of training runs (used by code/experiments/train.sh status / watch).

  python code/scripts/train_status.py                      # every run under $UHDD_RESULTS/dpdd/train (one line each)
  python code/scripts/train_status.py v0                   # one run in detail: progress, ETA, loss, val table
  python code/scripts/train_status.py /path/to/run_dir

Reads only what train.py writes (config.yaml, log.csv, val.csv, model_ema.pt, train.pid / stdout.log from train.sh).
Validation rows are shown against bicubic of the same anchor and against step 0 (the untrained model = locked HAT-L).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import yaml


def runs_root() -> Path:
    return Path(os.environ.get("UHDD_RESULTS", ".")) / "dpdd" / "train"


def resolve(arg: str) -> Path:
    p = Path(arg)
    return p if p.is_dir() else runs_root() / arg


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: (float(v) if v not in ("", None) and _num(v) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def hms(s: float) -> str:
    s = int(max(s, 0))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    return f"{d}d {h:02d}h{m:02d}m" if d else f"{h}h{m:02d}m"


def state(run: Path) -> str:
    pid = run / "train.pid"            # written by train.sh, removed by `train.sh stop` / a foreground run's exit
    if pid.exists():
        try:
            os.kill(int(pid.read_text().split()[0]), 0)
            return "RUNNING"
        except (OSError, ValueError):
            return "DIED"              # process gone without `stop`: see the end of stdout.log
    if (run / "train.stopped").exists():
        return "stopped"
    log = run / "log.csv"
    if log.exists() and time.time() - log.stat().st_mtime < 600:
        return "RUNNING?"              # log written in the last 10 min, not started by train.sh
    return "stopped"


def info(run: Path) -> dict:
    cfg = yaml.safe_load((run / "config.yaml").read_text()) if (run / "config.yaml").exists() else {}
    total = int(cfg.get("optim", {}).get("steps", 0))
    log, val = rows(run / "log.csv"), rows(run / "val.csv")
    last = log[-1] if log else {}
    step = int(last.get("step", 0))
    it_s = float(last.get("it_s", 0) or 0)
    eta = (total - step) / it_s if it_s > 0 and total > step else 0
    done = (run / "model_ema.pt").exists() and step >= total > 0
    return {"cfg": cfg, "total": total, "log": log, "val": val, "step": step, "it_s": it_s, "eta": eta,
            "state": "DONE" if done else state(run), "age": time.time() - (run / "log.csv").stat().st_mtime
            if (run / "log.csv").exists() else None}


def bar(frac: float, n: int = 30) -> str:
    k = int(round(n * min(max(frac, 0), 1)))
    return "[" + "#" * k + "." * (n - k) + "]"


def one_line(run: Path) -> str:
    i = info(run)
    v = i["val"][-1] if i["val"] else {}
    gain = (v.get("psnr", 0) - v.get("psnr_bicubic", 0)) if v.get("psnr") is not None else None
    vs = f"val PSNR {v['psnr']:.2f} ({gain:+.2f} vs bicubic)" if gain is not None else "no val yet"
    frac = i["step"] / i["total"] if i["total"] else 0
    return (f"{run.name:<18} {i['state']:<9} {bar(frac, 20)} {i['step']:>7}/{i['total']:<7} "
            f"{i['it_s']:5.2f} it/s  ETA {hms(i['eta']) if i['eta'] and i['state'].startswith('RUN') else '-':>9}  {vs}")


def detail(run: Path, n_val: int = 12) -> str:
    i = info(run)
    out = [f"run      {run}", f"name     {i['cfg'].get('name', run.name)}   state {i['state']}"
           + (f"   (last log line {hms(i['age'])} ago)" if i["age"] is not None else "")]
    frac = i["step"] / i["total"] if i["total"] else 0
    out.append(f"progress {bar(frac)} {100 * frac:5.1f} %  step {i['step']} / {i['total']}")
    if i["it_s"]:
        out.append(f"speed    {i['it_s']:.2f} it/s  ->  ETA {hms(i['eta'])}" +
                   (f"   (finishes ~{time.strftime('%a %d %b %H:%M', time.localtime(time.time() + i['eta']))})"
                    if i["eta"] else ""))
    if i["log"]:
        L = i["log"][-1]
        first = i["log"][0]
        out.append(f"loss     {L.get('loss', float('nan')):.4f}  (first logged {first.get('loss', float('nan')):.4f})"
                   f"   l1 {L.get('l1', float('nan')):.4f}   cx {L['cx']:.4f}" if isinstance(L.get('cx'), float) else "")
        out.append(f"         lr {L.get('lr', 0):.2e}   grad norm {L.get('grad_norm', 0):.3f}   "
                   f"peak GPU mem {L.get('mem_gb', 0)} GB")
        bad = [r for r in i["log"] if any(isinstance(x, float) and x != x for x in r.values())]
        if bad:
            out.append(f"WARNING  NaN in {len(bad)} logged steps (first at step {int(bad[0]['step'])})")
    ck = sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    out.append("ckpts    " + (", ".join(p.stem.split("_")[1] for p in ck) if ck else "none yet")
               + ("   model_ema.pt ✓" if (run / "model_ema.pt").exists() else ""))
    if i["val"]:
        v0 = i["val"][0] if int(i["val"][0]["step"]) == 0 else None
        out.append("")
        out.append("validation (EMA, fixed val tiles; dB, higher is better)")
        out.append(f"  {'step':>7}  {'PSNR':>7} {'vs bic':>7} {'vs s0':>7}   {'defocus':>7} {'vs bic':>7}   {'x4':>6}")
        for v in i["val"][-n_val:]:
            g = lambda k, b: (v.get(k) - v.get(b)) if isinstance(v.get(k), float) and isinstance(v.get(b), float) else None
            s0 = (v["psnr"] - v0["psnr"]) if v0 and isinstance(v.get("psnr"), float) else None
            f = lambda x, w=7: f"{x:+{w}.2f}" if x is not None else " " * (w - 1) + "-"
            d = v.get("psnr_defocus")
            out.append(f"  {int(v['step']):>7}  {v.get('psnr', 0):7.2f} {f(g('psnr', 'psnr_bicubic'))} {f(s0)}   "
                       f"{d if isinstance(d, str) else f'{d:7.2f}'} {f(g('psnr_defocus', 'psnr_defocus_bicubic'))}   "
                       f"{v.get('psnr_x4', 0):6.2f}")
        out.append("  vs bic = gain over bicubic of the same anchor; vs s0 = gain over step 0 (locked HAT-L)")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        run = resolve(argv[0])
        if not run.is_dir():
            sys.exit(f"no run at {run}")
        print(detail(run))
        return
    root = runs_root()
    runs = sorted(p for p in root.glob("*") if (p / "config.yaml").exists()) if root.is_dir() else []
    if not runs:
        print(f"no runs under {root}")
    for r in runs:
        print(one_line(r))


if __name__ == "__main__":
    main()
