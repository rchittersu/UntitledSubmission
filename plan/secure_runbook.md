# Secure-env runbook — handoff 6: ReFIR (SeeSR) and iRAG on selected tiles, same-image references

**One runbook per handoff.** Replaces the handoff-5 runbook (in git history). The handoff-5 and figures patches were
applied (`handoff/LOG.md`).

Audience: the Claude agent in the secure environment. The user runs the commands by hand (`launch.sh tiles …`) and
validates; you check each step and report. Read first: `docs/baselines.md` §D3, `code/experiments/launch.sh help`
(TILE STUDY), `code/uhdd/adapters/{refir,irag}.py` docstrings.

## Update 2026-10-11 (after the partial h6 report) — continue with these first
Your partial report was applied (`handoff/from_secure/2026-10-09_h6.md`). Answers and fixes:
- **iRAG unblocked.** iRAG's config uses the OpenCLIP text encoder (`FrozenOpenCLIPEmbedder`, open_clip). The
  transformers-CLIP subclass that failed (`CLIPTextTransformer`) is only *imported* at module level for an encoder iRAG
  never builds, as is OpenAI `clip`. `compat.irag_imports()` now installs placeholders for both (building them raises),
  so no numerical check is needed: nothing of them runs. Checked outside: `ldm.modules.encoders.modules` imports, and the
  TTSR branch (7 M params) and the structure encoder (52 M) build from the config. No separate environment.
- **"SeeSR with retrieved = none" is by design**: `seesr` is ReFIR's base without reference injection (`refir_scale: 0`),
  so it ignores `--ref`. Only `refir_seesr` uses the reference.
- **Does the reference reach ReFIR's output?** `refir_seesr` runs now write `tile_stats.json` (also in `meta.json`):
  per tile `injected_calls` (attention calls that took the injection branch; expected 20 steps × 36 decoder layers = 720
  with the counter reset per tile), `mean_weight` (refir_scale × layer schedule × similarity mask), `steps_seen`.
  ReFIR mixes `(1 − m)·self + m·cross` and then AdaIN-normalises the result back to the self-attention statistics, so a
  small effect is plausible even when it runs.
- The test-scene benchmark (1P0A1046) is recorded as a user-authorised deviation; its numbers are not used for decisions.

Next, in this order (val tiles from your `tiles select` / `prep --anchor drbnet`):
```bash
git pull
A="--anchor drbnet --limit 16"                 # 4 tiles per bin first
code/experiments/launch.sh tiles run $A --model refir_seesr --ref retrieved --overwrite          # now with tile_stats
code/experiments/launch.sh tiles run $A --model refir_seesr --ref retrieved --set refir_scale=1.0
code/experiments/launch.sh tiles run $A --model refir_seesr --ref retrieved --set refir_scale=2.0
code/experiments/launch.sh tiles run $A --model irag --ref self --tiles <one b2 tile>          # smoke: load_report, time
for r in self retrieved; do
  code/experiments/launch.sh tiles run $A --model irag --ref $r
  code/experiments/launch.sh tiles run $A --model irag_inter --ref $r
done
code/experiments/launch.sh tiles run $A --model bicubic --ref none
code/experiments/launch.sh tiles run $A --model highband --ref self
code/experiments/launch.sh tiles run $A --model seesr --ref none
code/experiments/launch.sh tiles report --anchor drbnet
```
Report `injected_calls` / `mean_weight` per run (min / mean / max over tiles), the iRAG `load_report`, and the per-bin
table. Then the full 56 tiles, then Bokehlicious, as in §4–5 below.

## Why
User decision 2026-10-08: before building our own model, take the two most recent reference-based diffusion SR methods
with code and weights — **ReFIR on SeeSR** (NeurIPS 2024) and **iRAG** (ICCV 2025, 50 DDIM steps) — run them in our
setting (x4 deblurred anchor + same-image references) on **selected val tiles** for immediate feedback, find what breaks,
then fix it (handoff 7). No full images in this handoff.

## Standing rules
- **Val split only** (`--split val`, the default). The test set is not touched.
- One Python environment (the main env), as for S3Diff / VOSR. If something cannot be made to work there, report the
  exact error and versions; do not create a separate env without telling the user.
- Report outcomes and deviations; no internal paths / hostnames / proprietary names; images stay inside (describe what
  you see in words, numbers in tables).

## What changed outside
- `uhdd/refsr.py` (references: `self`, `retrieved`; model registry `configs/refsr.yaml`; non-learned `highband`),
  adapters `uhdd/adapters/refir.py`, `uhdd/adapters/irag.py`, shims `compat.seesr_imports()` / `compat.irag_imports()`,
  `scripts/tile_study.py` (select / prep / run / report), `launch.sh setup refir|irag`, `launch.sh tiles …`, new
  requirements (open_clip_torch, kornia, omegaconf, scikit-learn, matplotlib, accelerate, gdown).
- `adapters/ours.py`: lock rounds after blending 4 → 16 (alternating lock and clamp converges slowly; 4 rounds leave
  ~1e-2 at x4).
- Training (handoffs 5) is paused; the synthetic-training plan (`plan/method_plan.md`, 2026-10-07) waits for this study.

---

## 0. Update, install, test
```bash
git pull && git checkout -b handoff/h6
pip install -r code/requirements.txt
pytest -q code/tests                  # outside: 90 passed, 1 skipped (+ the ReFIR / iRAG import tests run when the repos exist)
```

## 1. Code and weights
```bash
code/experiments/launch.sh setup refir     # ReFIR repo, SD-2-base (NOT 2.1), SeeSR + DAPE (Google Drive via gdown)
code/experiments/launch.sh setup irag      # iRAG repo, iRAG.ckpt (Google Drive), StableSR CFW VQGAN, OpenCLIP ViT-H-14
code/experiments/launch.sh check           # new block "ReFIR / iRAG (tile study)"
```
If Google Drive is not reachable, the user carries the files in; the expected layout is in `launch.sh help`
(TILE STUDY) and `code/configs/refsr.yaml`. RAM (`ram_swin_large_14m.pth`) and the BERT tokenizer are the OSEDiff
ones. Report: every file found, its size, and where it came from (generically).

## 2. Tiles and references
```bash
code/experiments/launch.sh tiles select                       # 8 val scenes x 2 tiles per DP bin (seed 0) -> ~64 tiles
code/experiments/launch.sh tiles prep --anchor drbnet         # crops + refs self / retrieved (DINOv2 memory)
code/experiments/launch.sh tiles prep --anchor bokehlicious   # second anchor (needs deblur/val_x4/bokehlicious_deblur@whole)
```
Check: tiles per bin (b0..b3) and scenes; open 4 retrieved mosaics (one per bin) and describe them (same material as
the tile? colour? flat cells?). `refs/retrieved@*/info.json` has the exemplar scores.

## 3. Smoke (one tile each, then look at it)
```bash
T=<one b2 tile id from tiles.json>
for m in seesr refir_seesr irag irag_inter; do
  code/experiments/launch.sh tiles run --anchor drbnet --model $m --ref retrieved --tiles $T
done
```
Report per model: load errors / warnings (the iRAG `load_report` in `runs/.../meta.json`: missing / unexpected key
counts must be small and explained), time per tile, peak GPU memory, and what the output looks like.

## 4. The matrix (DRBNet anchor; Bokehlicious after, same commands)
```bash
A="--anchor drbnet"
code/experiments/launch.sh tiles run $A --model bicubic --ref none
code/experiments/launch.sh tiles run $A --model highband --ref self            # non-learned copy of the observation
code/experiments/launch.sh tiles run $A --model seesr --ref none               # ReFIR's base without reference
for r in self retrieved; do
  code/experiments/launch.sh tiles run $A --model refir_seesr --ref $r
  code/experiments/launch.sh tiles run $A --model irag --ref $r
  code/experiments/launch.sh tiles run $A --model irag_inter --ref $r
done
code/experiments/launch.sh tiles run $A --model refir_seesr --ref retrieved --set reset_editor=false   # official counter bug
code/experiments/launch.sh tiles report $A
```
Several GPUs: run independent commands on different GPUs (`--gpu N`), or split one run with `--shard i/n`.
If ReFIR / iRAG take more than ~1 min per tile, run them with `--limit 16` first (4 per bin) and say so.

## 5. What to report (`handoff/from_secure/<date>_h6.md`)
- The report table (`tilestudy/val/report_<anchor>.md`): per run, raw and anchor-locked: ΔPSNR vs bicubic per bin
  (b0..b3), PSNR, SSIM, LPIPS, DISTS; time per tile.
- **Look at the contact sheets** (`tilestudy/val/sheets/<anchor>/`) for at least 8 tiles (2 per bin) and describe, per
  model and reference mode: focal-plane damage (b0), invented texture vs texture that matches the reference, glyph /
  text changes, mosaic seams or exemplar boundaries copied into the output, colour shifts, tile-border artefacts.
- Does the reference matter? `refir_seesr` vs `seesr`; `irag` with `self` vs `retrieved`; `irag_inter` vs bicubic.
- Does the anchor lock fix the focal-plane damage, and what does it cost in perceptual metrics?
- The issues you would fix first (our hypotheses: ReFIR's always-on injection, no abstain, SD prior in focus, mosaic
  format), with evidence.
- Deviations table; anything that differs from the official scripts.
`handoff/make_patch.sh`.
