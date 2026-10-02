# Handoff log

| Date | Direction | File | Summary | Status |
|---|---|---|---|---|
| 2026-10-01 | outside → secure | plan/secure_runbook_p1.md | P1 evaluation runbook: setup, baseline checks, DPDD registration/scales, Restormer reproduction, run_matrix | sent |
| 2026-10-02 | secure → outside | handoff/from_secure/2026-10-01_p1-eval.md (4 commits) | PARTIAL P1: Table 1 + G1 (4 models, 37 indoor scenes), G2 Restormer+{bicubic,SwinIR}; raw development (develop_raw.py), GT MTF measurement (estimate_gt_mtf.py); deviations D1-D6 | applied (git am, hashes verified, 39 tests pass); G2 rest + IFAN pending |
| 2026-10-02 | outside → secure | plan/secure_runbook_p1b.md | protocol v2: euclidean re-registration, gridshift/blurbins/pmg metrics, DP maps, focal-plane MTF, finish G2 + IFAN, paired CIs | sent |
| 2026-10-03 | outside → secure | plan/secure_runbook.md | handoff 2 (replaces P1b + P2a, never reported): native test 76 / val / train sets, anchors + train/test gap, VAE ceiling, protocol v3 (docs/evaluation.md), all baselines (docs/baselines.md), per-method paper tiling | sent |
