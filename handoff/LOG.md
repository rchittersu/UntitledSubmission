# Handoff log

| Date | Direction | File | Summary | Status |
|---|---|---|---|---|
| 2026-10-01 | outside → secure | plan/secure_runbook_p1.md | P1 evaluation runbook: setup, baseline checks, DPDD registration/scales, Restormer reproduction, run_matrix | sent |
| 2026-10-02 | secure → outside | handoff/from_secure/2026-10-01_p1-eval.md (4 commits) | PARTIAL P1: Table 1 + G1 (4 models, 37 indoor scenes), G2 Restormer+{bicubic,SwinIR}; raw development (develop_raw.py), GT MTF measurement (estimate_gt_mtf.py); deviations D1-D6 | applied (git am, hashes verified, 39 tests pass); G2 rest + IFAN pending |
| 2026-10-02 | outside → secure | plan/secure_runbook_p1b.md | protocol v2: euclidean re-registration, gridshift/blurbins/pmg metrics, DP maps, focal-plane MTF, finish G2 + IFAN, paired CIs | sent |
| 2026-10-03 | outside → secure | plan/secure_runbook.md | handoff 2 (replaces P1b + P2a, never reported): native test 76 / val / train sets, anchors + train/test gap, VAE ceiling, protocol v3 (docs/evaluation.md), all baselines (docs/baselines.md), per-method paper tiling | sent |
| 2026-10-04 | secure → outside | handoff/from_secure/2026-10-04_h2.md (10 commits) | handoff 2: native test 76 / val / train built; protocol-v3 matrix (5 deblurrers, 6 upsamplers, 5 fusion baselines tuned on val), all-76 + indoor-37 with CIs; code: Swin mask cache, fail-loud DP maps, crash-tolerant run_model, subset tool; DATSR and VAE not run | applied (8 code commits exact; report transcribed from the pasted patch), 61 tests pass; review below |
| 2026-10-04 | outside → secure | plan/secure_runbook.md | handoff 3 (evaluation only): complete frozen protocol on 76 (dpdd4: LPIPS, MUSIQ, CLIPIQA, SSIM@x4, Table 1 deblur-only x4), DATSR mosaic + mosaic_focus, idle-node timing, agent visual review of 8 captures, data facts + val proxy (perspective experiments dropped); standing rule: always 76 | sent |
