# Secure-env runbook — handoff 4: new results layout, fresh runs through the launcher, S3Diff / VOSR

**One runbook per handoff.** Replaces the handoff-3 runbook (in git history). Your handoff-3 patch was applied and
reviewed outside (review notes: `handoff/LOG.md`, `docs/evaluation.md` §7.7).

Audience: the Claude agent in the secure environment (the user triggers the long runs by hand with the launcher).
Read first: `code/experiments/launch.sh help`, `docs/evaluation.md` §5.0 (results layout), `docs/baselines.md`
("Manual launcher").

## Standing rules
- **All 76 test scenes**, always. **Evaluation only**, no training. **Protocol frozen** (dpdd4, decision log).
- Report results; keep interpretation short. Same rules as always: outcome of every step, deviations table, no
  internal paths / hostnames, hand off early if blocked.

## What changed outside
- **New results root `$UHDD_RESULTS/dpdd/`**, organised by stage → input source → chain (`code/uhdd/layout.py`):
  `inputs/<src>/` (symlinks + `manifest.json`), `deblur/<src>/<model>@<tiling>/`,
  `upsample/<src>/<anchor>@whole/<sr>@<tiling>/`, `fusion/…`, `scratch/`. Metrics stay next to the images; every run
  folder gets a `launch.json` (command, commit, scenes). Sources: `ours_x1`, `ours_x4`, `official_x4`, `ours_x2`.
- **Fresh start** (user decision): everything is re-run into the new root with `code/scripts/launch.py`
  (wrapper `code/experiments/launch.sh`). `launch_sr.py/.sh` were renamed to `launch.py/.sh` and now also run
  deblurrers. CLI: `launch.sh deblur SRC MODEL`, `launch.sh upsample SRC MODEL SR`; `upsample official_x4 …` upsamples an anchor
  computed on the original DPDD images (diagnostic).
- **Legacy roots** (`dpdd_v2`, `dpdd_official_x4`, `dpdd_p1`) are kept as they are and never written; they are only
  read by `launch.py compare`.
- `run_matrix.py`, `run_baselines_bd.sh`, `refsr_baseline.py` still write the legacy layout — **do not use them in
  this handoff** (fusion and DATSR are not ported to the new layout yet).

---

## 0. Update and test
```bash
git pull && git checkout -b handoff/h4        # base = origin/main
pytest -q code/tests                          # expect 71 passed
```

## 1. Inputs: link, verify, protect legacy
```bash
code/experiments/launch.sh setup inputs --protect-legacy      # = python code/scripts/setup_inputs.py
```
Checks (all must pass; a source that fails is refused by the launcher):
- 76 scenes per source; inputs / targets / masks / DP maps carry the same names;
- **names agree across sources** — the user believes the official and our scene names match; this is the check.
  It also verifies content: official vs ours ×4 per-scene PSNR (expected ≈ 39 dB, the calibration) and a nearest-
  thumbnail check that catches swapped names. If it fails, do not rename anything: report the mismatching names and
  the PSNR table, and stop at this step (the user decides);
- sizes: ours_x4 = native / 4, official_x4 = ours_x4.
Report: the printed summary + `checks` from `inputs/manifest.json` (PSNR median / min per kind, mismatches).
`--protect-legacy` removes write permission from the legacy roots (undo: `chmod -R u+w <dir>`).

## 2. S3Diff and VOSR 2.0 (registry models in the main env)
S3Diff and VOSR are now registry models (`s3diff_x4`, `vosr2_x4`, `vosr_0.5b_x4`; adapters in `code/uhdd/adapters/`)
run by `run_model.py` in the **one main env** with our tiling, like OSEDiff (docs/baselines.md, "S3Diff and VOSR in our
pipeline"). Update the main env and fetch code + weights:
```bash
pip install -r code/requirements.txt          # diffusers >= 0.35, transformers, peft, fairscale, ...
code/experiments/launch.sh setup s3diff       # $UHDD_REPOS/S3Diff, $UHDD_WEIGHTS/{sd-turbo,s3diff}
code/experiments/launch.sh setup vosr         # $UHDD_REPOS/VOSR, $UHDD_WEIGHTS/vosr
code/experiments/launch.sh check
```
Upgrading diffusers/transformers/peft can affect OSEDiff: re-run one OSEDiff scene and compare with its legacy output
(`compare` after step 3 covers it). If one library version cannot serve all three, report the conflict (versions,
error) — do not create separate envs without telling the user.
Smoke test, one scene each, then look at the output image (and `meta.json`: time, peak memory, tiling):
```bash
code/experiments/launch.sh upsample ours_x4 drbnet s3diff 0 --scenes <one test scene>
code/experiments/launch.sh upsample ours_x4 drbnet vosr2 0 --scenes <one test scene>
```

## 3. Fresh standard set (the user may launch these; check them)
```bash
code/experiments/launch.sh fresh 0          # deblur {input + 5} on ours_x4, official_x4, ours_x1; drbnet/bokehlicious x 5 upsamplers; then compare
code/experiments/launch.sh study ours_x4 s3diff 0
code/experiments/launch.sh study ours_x4 vosr2 1
code/experiments/launch.sh upsample official_x4 drbnet vosr2 0          # diagnostic
```
Multi-GPU: give a GPU list (`0,1,2,3` or `all`) as the GPU argument; the scenes are split over the GPUs (one
process per GPU; each loads its own model copy). Independent commands on disjoint GPUs also work (runs resume).

## 4. Correctness re-check
```bash
code/experiments/launch.sh compare
```
Fresh vs legacy (handoff 2/3) for PSNR / SSIM / LPIPS / DISTS. Same harness, same weights, same tiling → expect
agreement to ~0.01 dB (identity, bicubic, deblurrers at ×4) and small differences for tiled / stochastic models.
For every row that differs by more than 0.05 dB PSNR or 0.005 LPIPS: find out why (tiling tag, input source,
inputs bit depth, weights) and report. In particular re-check the handoff-3 anomaly: **LaKDNet / Bokehlicious ×4 +
HAT-L** (in-focus −1.0 dB, SSIM drop) — does it reproduce? Look at the images.

## 5. Results
```bash
code/experiments/launch.sh summary
```
Tables per stage / source with CIs (all 76): deblur at ×4 (ours and official, Table 1 both renderings), native
deblur, upsample (all anchors × upsamplers incl. S3Diff / VOSR 2.0), the official-anchor diagnostic. Runtime per
image and GPU type.

## 6. Handoff
Report `handoff/from_secure/<date>_h4.md` (template in CLAUDE.md): steps 0–5, the manifest checks, deviations,
compare table, summary tables. Update `docs/evaluation.md` §7 and `docs/baselines.md` status. `handoff/make_patch.sh`.
