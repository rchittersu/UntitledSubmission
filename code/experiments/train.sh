#!/usr/bin/env bash
# =====================================================================================================================
# train.sh — manual training of the method (v0), step by step, with progress you can watch.
# Canonical description: docs/training.md. Evaluation runs stay in launch.sh (this script calls it for `anchors` / `eval`).
#
# THE STEPS (in order; each is safe to re-run: finished work is skipped, training resumes from last.pt)
#   1  train.sh inputs                  link + verify train / val inputs and DP maps (350 / 74 scenes)
#   2  train.sh check                   offline weights load (DINOv2-L, VGG19, HAT-L), GPUs, caches
#   3  train.sh anchors [GPU]           x4 anchors (bokehlicious, drbnet, restormer) on train_x4 and val_x4, scored
#   4  train.sh cache val [GPU]         training cache of val, then train (+ M1 / M2 measurements, pixel ablation)
#      train.sh cache train [GPU]
#   5  train.sh step0 [NGPU]            no training: validates the untrained model (= locked HAT-L), the reference row
#   6  train.sh overfit auto [NGPU]     3k steps on 2 train scenes (auto = the 2 most defocused); must fit them
#   7  train.sh smoke [NGPU] [--set k=v]  300 steps of v0: speed, memory, dataloader -> projected time of a full run
#   8  train.sh start v0_oracle [NGPU]  full run in the background (oracle exemplars: upper bound, go / no-go)
#      train.sh start v0 [NGPU]         full run, retrieved exemplars (the method)
#   9  train.sh eval v0 val_x4 bokehlicious [GPU]   full images through the launcher (val for selection;
#      train.sh eval v0 ours_x4 bokehlicious [GPU]  test = ours_x4, once, at the end)
#
# WATCHING
#   train.sh status                     one line per run: state, progress bar, it/s, ETA, latest val PSNR vs bicubic
#   train.sh status v0                  one run in detail: progress, ETA + finish time, loss, lr, grad norm, GPU
#                                       memory, checkpoints, validation table (vs bicubic, vs step 0)
#   train.sh watch v0 [SECONDS]         `status v0` refreshed every SECONDS (default 60); Ctrl-C to leave
#   train.sh log v0                     follow the raw output (tail -f stdout.log)
#   tensorboard --logdir $UHDD_RESULTS/dpdd/train                       (if TensorBoard is installed)
#
# OTHER
#   train.sh run CONFIG [NGPU] [--set k=v ...]    like start, but in the foreground (output also in stdout.log)
#   train.sh start CONFIG [NGPU] [--set k=v ...]  CONFIG = v0 | v0_oracle | tiny | path/to.yaml; --set overrides
#                                                 config values, e.g. --set optim.steps=50000 batch=2
#   train.sh stop NAME                  stop a background run; `train.sh start NAME` resumes it from the last
#                                       checkpoint with the run's saved settings (train/<name>/config.yaml)
#   train.sh env                        environment in use
#   NGPU: number of GPUs (default: all visible; restrict with CUDA_VISIBLE_DEVICES). GPU (anchors / cache / eval):
#   0 | 0,1,2,3 | all, as in launch.sh.
#
# WHERE THINGS GO ($UHDD_RESULTS/dpdd/)
#   cache/<split>/<scene>/ + manifest.json     training cache (≈155 GB train + val), M1 / M2 in manifest.json
#   train/<name>/                              config.yaml, log.csv (every 50 steps), val.csv (step 0, every 2k),
#                                              stdout.log, ckpt_<step>.pt (last 3), last.pt, model_ema.pt, tb/
#   train/_step0, _overfit, _smoke             the check runs (safe to delete)
#   upsample/<src>/<anchor>@whole/ours_v0@t128o16/   full-image results (scored like every baseline)
#
# ENVIRONMENT: UHDD_DATA, UHDD_RESULTS, UHDD_REPOS, UHDD_WEIGHTS, PY (as launch.sh).
#   Weights read offline: HAT-L (configs/models.yaml hat_l_x4), DINOv2-L ($UHDD_WEIGHTS/vosr/torch_cache),
#   VGG19 ($UHDD_WEIGHTS/vgg/vgg19-dcbb9e9d.pth, torchvision's file; copy it in if missing).
# =====================================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
LAUNCH_SH="$ROOT/code/experiments/launch.sh"
TRAIN="$ROOT/code/scripts/train.py"
STATUS="$ROOT/code/scripts/train_status.py"
CACHE="$ROOT/code/scripts/build_train_cache.py"
CONFIGS="$ROOT/code/configs/train"
ANCHORS=(bokehlicious drbnet restormer)

usage() { sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
need() { for v in "$@"; do [[ -n "${!v:-}" ]] || die "set \$$v (see: $0 help)"; done; }
step() { echo; echo ">> $*"; }
runs() { echo "$UHDD_RESULTS/dpdd/train"; }
ngpu() { [[ -n "${1:-}" && "${1:-}" != --* ]] && echo "$1" || "$PY" -c 'import torch; print(max(1, torch.cuda.device_count()))'; }

config_path() {
  local c="$1"
  [[ -f "$c" ]] && { echo "$c"; return; }
  [[ -f "$CONFIGS/$c.yaml" ]] && { echo "$CONFIGS/$c.yaml"; return; }
  die "no config '$c' (have: $(ls "$CONFIGS" | sed 's/\.yaml//' | tr '\n' ' '))"
}
config_name() {  # CONFIG [--set k=v ...] -> run name after the overrides (name=... renames the run folder)
  local c="$1"; shift
  "$PY" - "$ROOT" "$c" "$@" <<'EOF2'
import sys
sys.path.insert(0, sys.argv[1] + "/code/scripts")
import train
print(train.load_config(sys.argv[2], [a for a in sys.argv[3:] if a != "--set"])["name"])
EOF2
}

is_running() {  # run dir -> 0 if its recorded process is alive
  [[ -f "$1/train.pid" ]] && kill -0 "$(cut -d' ' -f1 < "$1/train.pid")" 2>/dev/null
}

# train_go MODE(fg|bg) CONFIG NGPU OUT [train.py args]
train_go() {
  local mode="$1" cfg="$2" n="$3" out="$4"; shift 4
  need UHDD_RESULTS
  mkdir -p "$out"
  is_running "$out" && die "already running: $out (train.sh status $(basename "$out"); stop with train.sh stop $(basename "$out"))"
  local cmd=("$PY" "$TRAIN" --config "$cfg" --out "$out" "$@")       # 1 GPU: plain python
  (( n > 1 )) && cmd=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$n" "$TRAIN" --config "$cfg" --out "$out" "$@")
  { echo; echo "=== $(date '+%F %T')  ${cmd[*]}"; } >> "$out/stdout.log"
  rm -f "$out/train.stopped"
  [[ -f "$out/last.pt" ]] && echo "resuming from $out/last.pt"
  if [[ "$mode" == bg ]]; then
    if command -v setsid >/dev/null; then nohup setsid "${cmd[@]}" >> "$out/stdout.log" 2>&1 &
    else nohup "${cmd[@]}" >> "$out/stdout.log" 2>&1 & fi
    echo "$! $(date '+%F %T')" > "$out/train.pid"
    echo "started in the background on $n GPU(s): $out"
    echo "  watch:  $0 watch $(basename "$out")      log: $0 log $(basename "$out")      stop: $0 stop $(basename "$out")"
  else
    echo "$$ $(date '+%F %T')" > "$out/train.pid"
    trap "rm -f '$out/train.pid'" EXIT   # $out expanded now: it is a function-local, gone (and `set -u` fatal) when the trap fires at exit
    "${cmd[@]}" 2>&1 | tee -a "$out/stdout.log"
    echo; "$PY" "$STATUS" "$out"
  fi
}

check() {
  need UHDD_RESULTS UHDD_WEIGHTS
  "$PY" - "$ROOT" <<'EOF'
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1]); sys.path[:0] = [str(root / "code"), str(root / "code" / "scripts")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
R, W = Path(os.environ["UHDD_RESULTS"]) / "dpdd", Path(os.environ["UHDD_WEIGHTS"])
ok = True
def chk(name, fn):
    global ok
    t = time.time()
    try:
        msg = fn() or ""
        print(f"  ok    {name:<44} {msg} ({time.time() - t:.0f} s)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FAIL  {name:<44} {type(e).__name__}: {str(e)[:160]}")
import torch
def gpus():
    n = torch.cuda.device_count()
    assert n > 0, "no CUDA device"
    return f"{n} x {torch.cuda.get_device_name(0)}"
def inputs():
    m = json.loads((R / "inputs" / "manifest.json").read_text())
    have = {s: m["sources"].get(s, {}).get("n") for s in ("train_x1", "train_x4", "val_x1", "val_x4")}
    assert all(have.values()), f"missing sources {have} (run: train.sh inputs)"
    return str(have)
def dino():
    from uhdd.train.memory import Features
    f = Features("dinov2", "dinov2_vitl14", str(W / "vosr" / "torch_cache"), "cuda" if torch.cuda.is_available() else "cpu")
    return "dinov2_vitl14 loaded offline"
def vgg():
    from uhdd.train.losses import VGGFeatures
    VGGFeatures(str(W / "vgg" / "vgg19-dcbb9e9d.pth"))
    return "vgg19 relu3_1"
def hat():
    import train
    from uhdd.net.ours import build_model
    cfg = train.load_config(str(root / "code" / "configs" / "train" / "v0.yaml"), [])
    m = build_model(cfg["model"])
    return f"v0 built, {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M params"
def caches():
    out = []
    for s in ("train", "val"):
        p = R / "cache" / s / "manifest.json"
        out.append(f"{s}: {json.loads(p.read_text())['n']} scenes" if p.exists() else f"{s}: not built")
    return ", ".join(out)
for n, f in [("GPUs", gpus), ("train / val inputs (manifest)", inputs), ("DINOv2-L offline", dino),
             ("VGG19 weights", vgg), ("v0 model with HAT-L weights", hat), ("caches", caches)]:
    chk(n, f)
print("all checks passed" if ok else "some checks failed")
EOF
}

overfit_scenes() {  # "auto" -> the 2 train scenes with the largest defocused share (M1 bins b2 + b3)
  "$PY" - "$UHDD_RESULTS/dpdd/cache/train" <<'EOF'
import json, sys
from pathlib import Path
s = sorted(((m["m1_blur_bins"][2] + m["m1_blur_bins"][3], m["name"]) for m in
           (json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*/meta.json"))), reverse=True)
if len(s) < 2:
    sys.exit("train cache not built (train.sh cache train)")
print(",".join(n for _, n in s[:2]))
EOF
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  help|-h|--help) usage ;;
  env)     for v in UHDD_DATA UHDD_RESULTS UHDD_REPOS UHDD_WEIGHTS PY CUDA_VISIBLE_DEVICES; do
             printf '  %-20s %s\n' "$v" "${!v:-<unset>}"; done ;;
  inputs)  "$LAUNCH_SH" setup inputs --sources train_x1,train_x4,val_x1,val_x4 "$@" ;;
  check)   check ;;
  anchors) gpu="${1:-0}"
           for s in train_x4 val_x4; do for m in "${ANCHORS[@]}"; do
             step "anchor $m on $s"; "$LAUNCH_SH" deblur "$s" "$m" "$gpu"; done; done
           "$LAUNCH_SH" summary --stage deblur ;;
  cache)   [[ $# -ge 1 ]] || die "cache train|val [GPU] [build_train_cache.py args]"
           split="$1" gpu="${2:-all}"; shift $(( $# >= 2 ? 2 : 1 ))
           "$PY" "$CACHE" --split "$split" --gpus "$gpu" --compare-pixels "$@" ;;
  step0)   need UHDD_RESULTS
           train_go fg "$CONFIGS/v0.yaml" "$(ngpu "${1:-}")" "$(runs)/_step0" --set optim.steps=0 name=_step0 ;;
  overfit) [[ $# -ge 1 ]] || die "overfit auto|SCENE_A,SCENE_B [NGPU]"
           need UHDD_RESULTS
           sc="$1"; [[ "$sc" == auto ]] && sc="$(overfit_scenes)"
           echo "overfitting on: $sc"
           train_go fg "$CONFIGS/v0.yaml" "$(ngpu "${2:-}")" "$(runs)/_overfit" --set name=_overfit optim.steps=3000 \
             optim.warmup=200 "data.names=[$sc]" "val.names=[$sc]" "val.cache=\${UHDD_RESULTS}/dpdd/cache/train" \
             val.every=500 val.tiles=64 ckpt_every=3000 ;;
  smoke)   need UHDD_RESULTS
           n="$(ngpu "${1:-}")"; [[ -n "${1:-}" && "${1:-}" != --* ]] && shift
           train_go fg "$CONFIGS/v0.yaml" "$n" "$(runs)/_smoke" --set name=_smoke optim.steps=300 \
             val.every=300 log_every=10 ckpt_every=300 "$@" ;;
  run|start)
           [[ $# -ge 1 ]] || die "$cmd CONFIG [NGPU] [--set k=v ...]"
           need UHDD_RESULTS
           cfg="$(config_path "$1")"; shift
           n="$(ngpu "${1:-}")"; [[ -n "${1:-}" && "${1:-}" != --* ]] && shift
           out="$(runs)/$(config_name "$cfg" "$@")"
           if [[ -f "$out/config.yaml" && $# -eq 0 ]]; then      # resume with the run's own settings (incl. its --set)
             cfg="$out/config.yaml"; echo "using the run's saved settings: $cfg"
           fi
           train_go "$([[ $cmd == start ]] && echo bg || echo fg)" "$cfg" "$n" "$out" "$@" ;;
  stop)    [[ $# -ge 1 ]] || die "stop NAME"
           need UHDD_RESULTS
           d="$(runs)/$1"; is_running "$d" || die "$1 is not running (no live process in $d/train.pid)"
           pid="$(cut -d' ' -f1 < "$d/train.pid")"
           kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
           for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
           kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
           rm -f "$d/train.pid"; date '+%F %T' > "$d/train.stopped"
           echo "stopped $1 (resume: $0 start $1 — continues from the last checkpoint, every ckpt_every steps)" ;;
  status)  need UHDD_RESULTS; "$PY" "$STATUS" "$@" ;;
  watch)   [[ $# -ge 1 ]] || die "watch NAME [SECONDS]"
           need UHDD_RESULTS
           while true; do clear; date '+%F %T'; "$PY" "$STATUS" "$1"; sleep "${2:-60}"; done ;;
  log)     [[ $# -ge 1 ]] || die "log NAME"
           need UHDD_RESULTS; tail -n 50 -f "$(runs)/$1/stdout.log" ;;
  eval)    [[ $# -ge 3 ]] || die "eval NAME SRC ANCHOR [GPU] [launch.py args]   (NAME: v0 | v0_oracle)"
           name="$1" src="$2" anchor="$3" gpu="${4:-0}"; shift $(( $# >= 4 ? 4 : 3 ))
           need UHDD_RESULTS
           [[ -f "$(runs)/$name/model_ema.pt" ]] || die "no $(runs)/$name/model_ema.pt yet (first checkpoint: ckpt_every steps)"
           "$LAUNCH_SH" upsample "$src" "$anchor" "ours_$name" "$gpu" "$@" ;;
  *)       die "unknown command '$cmd' (see: $0 help)" ;;
esac
