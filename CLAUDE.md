# UntitledSubmission — CVPR paper workspace

**Target venue: CVPR 2027** (the template files still say 2026; update `\confYear` when finalizing).

Research project: **ultra-high-resolution (native sensor resolution, ~30 MP+) single-image defocus deblurring** via a blur-guided generative upsampler. Working method name is `\method{}` (placeholder, final name TBD).

## Repository layout

- `Template/` — CVPR author-kit LaTeX project.
  - `main.tex` — entry point (review mode). Title/authors/paper ID still template defaults.
  - `preamble.tex` — extra packages/macros. Note: defines `\todo`, so the `\providecommand{\todo}` in `sec/1_intro.tex` is ignored.
  - `sec/1_intro.tex` — **the only real content**: Introduction, Related Work, and (appended at the end) a draft method subsection "Surrogate Interface for Off-the-Shelf Methods". A commented list of all citation keys used sits between Related Work and that subsection.
  - `sec/0_abstract.tex`, `sec/2_formatting.tex`, `sec/3_finalcopy.tex`, `sec/X_suppl.tex`, `rebuttal.tex` — untouched template text.
  - `main.bib` — template entries only; **none of the paper's citation keys exist yet** (all `\cite`s are undefined).
  - `notes.txt` — informal research notes (inconsistency taxonomy for tiled inference, papers to read, open directions).
  - Build artifacts (`*.aux`, `*.log`, `main.pdf`, …) are git-ignored.
- Build: `cd Template && latexmk -pdf main.tex`.

## Core idea (summary of `sec/1_intro.tex`)

- **Stage 1** (4× downsampled, whole image in one pass): outputs (i) an all-in-focus *anchor*, (ii) a *deblur-level map* (how much restoration + how reliable), (iii) a whole-image *affinity map* from self-attention.
- **Stage 2** (native resolution, patch-wise): one-step (distilled) conditional diffusion upsampler. Affinity routes *cross-patch attention* to related patches anywhere in the image; the deblur-level map drives *blur-aware copy/generate* (copy real detail where it survives in the blurry input, synthesize only where it does not).
- **Plug-in**: any off-the-shelf low-res deblurrer can be lifted to native res. Surrogates replace (ii) and (iii): re-blur agreement (+ input sharpness, optional scale-perturbation variance) → tiny calibration head; DINOv2 feature affinity. Trained with native/surrogate signals randomly swapped and a mixture of real + simulated anchors.
- **Data/eval**: synthetic defocus from sharp HR photos + real DPDD pairs; native-res DPDD protocol with registration, perceptual/no-reference metrics, patch-seam and cross-patch consistency metrics.

---

## Two-environment workflow (IMPORTANT)

This project is worked on by **two Claude instances**:

| | **Outside env** (this checkout) | **Secure env** |
|---|---|---|
| Has | Paper sources, notes, internet access | Code, data, GPUs, experiments, its own clone of this repo |
| Can send out | Anything | **Only text: patches (unified diffs) and Markdown** |

The secure env currently has only bits and pieces (some eval setups); implementation is effectively starting from scratch.

Nothing binary (images, PDFs, checkpoints, data, logs as files) can leave the secure env. Everything must be communicated as text that a human copies out. Therefore the secure-env agent must write **self-contained, copy-pasteable Markdown handoff reports** following the protocol below.

### Handoff directories

- `handoff/from_secure/` — reports written in the secure env, carried out and dropped here.
- `handoff/to_secure/` — task briefs / questions written here, carried into the secure env.

File name: `YYYY-MM-DD_<short-topic>.md` (e.g. `2026-10-04_stage1-affinity-ablation.md`). One report per logical unit of work. Transfer is plain copy-paste, so length is not a hard constraint, but keep reports focused.

### Git base

Both sides use git so patches apply cleanly. The secure env clones this repo; every report must state the **commit hash of this repo it was diffed against** (in `Context`). Paper patches must be produced with `git diff` against that commit so they can be applied here with `git apply`. When the outside side commits applied patches, the next brief states the new base commit.

### Handoff report format (secure → outside)

The outside agent has **no access** to the secure code, data, or run logs. Write as if the reader knows the paper (this repo) but nothing else. Use this template:

````markdown
# <Topic> — <YYYY-MM-DD>

## TL;DR
3–6 bullets: what was done, headline result, what changes in the paper/plan.

## Context
Which task brief this answers (link `handoff/to_secure/...` if any), and the base state
(repo version or last handoff this builds on).

## What was done
Methods, configs, and settings that matter for writing the paper
(architecture, backbone, resolution, patch size, steps, datasets/splits, #params, GPU, runtime).
Give exact values, not "default".

## Results
Numbers as Markdown tables with units, dataset/split, resolution, and number of images.
Mark best per column in **bold**. State seeds / variance if run more than once.
For anything visual (figures, qualitative results): describe in words what is seen, and
where useful provide the plot data as a small table or CSV block so a figure can be
regenerated here (pgfplots/matplotlib). Never reference a figure by file path only.

## Decisions & findings
What worked, what did not, surprises, and anything that contradicts claims in `sec/1_intro.tex`
(quote the sentence and say what should change).

## Paper patches
Changes to files in THIS repo (LaTeX, bib, notes) as unified diffs against the current
version, one fenced ```diff block per file, with paths relative to repo root, e.g.:

```diff
--- a/Template/sec/1_intro.tex
+++ b/Template/sec/1_intro.tex
@@ -115,3 +115,3 @@
...
```

New files: give the full content in a fenced block headed by its path.
Keep diffs minimal (no whitespace-only reflow). If a diff is large or fragile, give the
replacement text of the whole paragraph/section instead, with the exact first and last line
it replaces.

## Code state (for the record only)
Short description of relevant code changes in the secure repo (module names, key functions,
config names) — enough to refer to them later. Code diffs are optional; include only small,
paper-relevant snippets (e.g. a loss or metric definition) so they can be described accurately.

## Open questions / needs from outside
Literature checks, writing help, citations to verify, decisions needed.

## Next steps
What the secure env plans to do next.
````

Rules for the secure-env agent:
- **Text only.** No attachments, no base64 blobs of images/binaries.
- **No sensitive content**: no credentials, internal hostnames/paths, proprietary dataset names or internal identifiers unless the user explicitly says they are cleared to leave. Describe them generically (e.g. "internal 50 MP smartphone test set, N=120 images").
- Every number must say what it is measured on (dataset, split, resolution, metric direction ↑/↓).
- Distinguish clearly between **verified** results and **expectations/hypotheses**.
- Prefer many small reports over one huge report.

### Task brief format (outside → secure)

Files in `handoff/to_secure/` contain: goal, relevant paper section/claim (quote it), exact experiments/ablations requested, the metrics and tables expected back, and priority. The secure agent answers with a report that links the brief.

### Outside-agent duties when a report arrives

1. Read the whole report; apply the `Paper patches` (`git apply --3way`, or manual edit if the base diverged) and check LaTeX still builds.
2. Update claims/`\todo{}`s in the paper that the results resolve; flag claims the results contradict.
3. Keep `handoff/LOG.md` updated: one line per report/brief (date, file, one-line summary, status).
