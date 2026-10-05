# UntitledSubmission — CVPR paper workspace

**Target venue: CVPR 2027** (deadline mid-November 2026; `\confYear` set to 2027).

Research project: **ultra-high-resolution (native sensor resolution, ~30 MP+) single-image defocus deblurring** via a blur-guided generative upsampler. Working method name is `\method{}` (placeholder, final name TBD).

## Repository layout

- `Template/` — CVPR author-kit LaTeX project.
  - `main.tex` — entry point (review mode). Working title set; authors / paper ID still template defaults.
  - `preamble.tex` — packages and project macros (`\todo` renders red; disable before submission).
  - `sec/0_abstract.tex` … `sec/5_conclusion.tex` — the paper draft (rewritten 2026-10-04 from the results so far):
    abstract, introduction, related work, method (draft, follows `plan/method_plan.md`), experiments (protocol +
    all baseline numbers from handoff 2; method rows are `\todo`), conclusion. Macros (`\method`, `\todo`, `\up`, …)
    are in `preamble.tex`; `cleveref` is loaded in `main.tex`.
  - `sec/X_suppl.tex`, `rebuttal.tex` — untouched template text.
  - `main.bib` — all cited entries (entries preceded by `% VERIFY` need a venue/author check).
  - `notes.txt` — informal research notes (inconsistency taxonomy for tiled inference, papers to read, open directions).
  - Build artifacts (`*.aux`, `*.log`, `main.pdf`, …) are git-ignored.
- `code/` — evaluation harness (`uhdd` package + scripts, see `code/README.md`). Written outside, run in the secure env; outside can only test on CPU (`pytest code/tests`).
  Results layout (`$UHDD_RESULTS/dpdd/`: inputs / deblur / upsample / fusion, by source) and the manual launcher
  (`code/experiments/launch.sh`): `docs/evaluation.md` §5.0.
- `docs/` — **canonical evaluation and baseline documentation**: `docs/evaluation.md` (protocol, every metric with definition + code, how to run, results, done / priority / todo) and `docs/baselines.md` (every baseline: purpose, setup, weights, commands, status, results). Keep them current when results arrive (both envs may edit them in patches).
- `plan/` — research plans shared by both envs: `plan/method_plan.md` (the method, variants A/B, timeline). **`plan/secure_runbook.md` is the single runbook for the current handoff** — one runbook per handoff, replaced by outside after the secure patch is applied (older runbooks live in git history).
- Build: `cd Template && latexmk -pdf main.tex`.

## Core idea (summary of the paper draft and `plan/method_plan.md`)

- **Finding** (handoff 2, 76 native DPDD scenes): native-resolution inference of DPDD deblurrers ≈ identity (blur extent
  out of distribution, not tiling); ×4 deblur + bicubic recovers structure but no native detail; generative SR invents
  detail and damages in-focus regions.
- **Method** (\method): blur-guided upsampler from the native blurry input + ×4 anchor (any deblurrer) + blur map
  (DP oracle or predicted). Copy path where the input is sharp; **global in-focus exemplar memory** (keys in the anchor
  domain, values = native in-focus texture, null entry) queried by every tile; generator fills the rest. Variants:
  feed-forward U-Net and one-step DiT (copy path in pixel space).
- **Evaluation**: protocol frozen 2026-10-04 (standard metrics headline; blur-stratified / no-harm / hallucination
  as diagnostics) — `docs/evaluation.md`.

---

## Two-environment workflow (IMPORTANT)

This project is worked on by **two Claude instances**:

| | **Outside env** (this checkout, origin of the repo) | **Secure env** |
|---|---|---|
| Has | Paper sources, notes, `code/` (written here, CPU-tested only), internet | Data, GPUs, experiments, a clone of this repo |
| Inbound | — | Can pull the latest version of this repo at any time |
| Outbound | Anything | **Exactly one text file per handoff: a git patch**, copy-pasted out by the user |

The secure env currently has only bits and pieces (some eval setups); implementation is effectively starting from scratch.

### Outside → secure

Nothing to do: the user pulls this repo into the secure env whenever needed, so the repo itself (paper, notes, `CLAUDE.md`, `handoff/LOG.md`) is the channel. Requests to the secure agent are given by the user directly.

### Secure → outside: one patch file

Everything the secure env wants to communicate travels as **one patch file against this repo**. The narrative report is not sent separately — it is a **new Markdown file inside the patch**, at `handoff/from_secure/YYYY-MM-DD_<topic>.md`. Paper edits (LaTeX, bib, notes, figures-as-code) are ordinary changes in the same patch.

Producing it in the secure env:

```bash
git pull                                   # start from the latest outside version (any branch)
git checkout -b handoff/<topic>
# ... write handoff/from_secure/<date>_<topic>.md, edit Template/..., commit (one or more commits) ...
handoff/make_patch.sh                      # -> ../handoff_<date>.patch
```

`make_patch.sh` works from **any origin branch**: the base is the current branch's `origin/*` upstream, else the `origin/*` branch HEAD is closest to, or explicitly `BASE=<branch>` (e.g. `BASE=paper-v2` = `origin/paper-v2`). The patch covers all commits since the fork point from that branch (`git format-patch --stdout --base=<fork-point>`), so the base branch moving on later is harmless. It refuses to write the patch if it touches paths outside `Template/`, `plan/`, `docs/`, `code/`, `handoff/from_secure/`, `CLAUDE.md`, `.gitignore`, contains binary files, or has no report in `handoff/from_secure/`, and warns about uncommitted changes. The report's `Context` section should name the base branch. The user copies the printed file out verbatim.

Applying it here: `git am -3 <file>.patch` (fallback: `git apply --3way`, or manual edit if the base diverged).

Patch rules:
- **Only paths of this repo** (`handoff/from_secure/`, `Template/`, `plan/`, `docs/`, `code/`, `CLAUDE.md`). Changes to `code/` (fixes, new metrics/baselines) are welcome; other secure-only code, data, logs and configs never leave. Keep code free of internal paths (use the `UHDD_*` env vars).
- **Text only**: no binary files (`make_patch.sh` checks this). Figures travel as plot data (CSV/table in the report or a `.dat`/pgfplots/TikZ file), not images.
- **No sensitive content**: no credentials, internal hostnames/paths, proprietary dataset names or internal identifiers unless the user explicitly says they are cleared to leave. Describe them generically (e.g. "internal 50 MP smartphone test set, N=120 images").
- Keep paper diffs minimal (no whitespace-only reflow of paragraphs) so they apply cleanly.
- Do not edit `handoff/LOG.md` (outside maintains it) to avoid conflicts.

### Report format (the `.md` inside the patch)

The outside agent has **no access** to the secure code, data, or run logs. Write as if the reader knows the paper (this repo) but nothing else:

````markdown
# <Topic> — <YYYY-MM-DD>

## TL;DR
3–6 bullets: what was done, headline result, what changes in the paper/plan.

## Context
What task this answers and the base commit of this repo.

## What was done
Methods, configs, and settings that matter for writing the paper
(architecture, backbone, resolution, patch size, steps, datasets/splits, #params, GPU, runtime).
Exact values, not "default".

## Results
Markdown tables with units, dataset/split, resolution, #images, metric direction (↑/↓).
Best per column in **bold**. Seeds / variance if run more than once.
Anything visual: describe in words what is seen, and give the plot data as a table/CSV block
so the figure can be regenerated outside. Never reference a figure by file path only.

## Decisions & findings
What worked, what did not, surprises, and anything that contradicts claims in the paper
(quote the sentence and say what should change). Mark **verified** results vs. **hypotheses**.

## Paper changes in this patch
One line per changed file: what changed and why.

## Code state (for the record only)
Short description of relevant secure-side code (module/function/config names) so it can be
referred to later. Small paper-relevant snippets (e.g. a loss or metric definition) may be
pasted inline; no full code.

## Open questions / needs from outside
Literature checks, writing help, citations to verify, decisions needed.

## Next steps
````

### Outside-agent duties when a patch arrives

1. Save it, apply with `git am -3`, resolve conflicts, check LaTeX still builds.
2. Read the report; update claims/`\todo{}`s the results resolve; flag claims the results contradict.
3. Replace `plan/secure_runbook.md` with the next handoff's runbook (one runbook per handoff; do not keep finished ones).
4. Add a line to `handoff/LOG.md` (date, direction, file, one-line summary, status) and commit, so the next pull brings the secure env up to date.

## Code conventions (`code/`)

- Efficiency first: one process per GPU (`--gpus all`), prefetching readers, async writers, GPU metrics, process pools for CPU work. Keep images in native 8/16-bit until on the device.
- Never resize to satisfy architecture size constraints: reflect-pad to the model's `multiple` and crop back (see `code/README.md`).
- Every script writes machine-readable results (CSV/JSON); `summarize.py` turns them into the Markdown tables that go into handoff reports.
- Add a CPU test in `code/tests` for new logic; run `pytest -q code/tests` before handing off.
