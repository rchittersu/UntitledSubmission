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

| | **Outside env** (this checkout, origin of the repo) | **Secure env** |
|---|---|---|
| Has | Paper sources, notes, internet access | Code, data, GPUs, experiments, a clone of this repo |
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

`make_patch.sh` works from **any origin branch**: the base is the current branch's `origin/*` upstream, else the `origin/*` branch HEAD is closest to, or explicitly `BASE=<branch>` (e.g. `BASE=paper-v2` = `origin/paper-v2`). The patch covers all commits since the fork point from that branch (`git format-patch --stdout --base=<fork-point>`), so the base branch moving on later is harmless. It refuses to write the patch if it touches paths outside `Template/`, `handoff/from_secure/`, `CLAUDE.md`, `.gitignore`, contains binary files, or has no report in `handoff/from_secure/`, and warns about uncommitted changes. The report's `Context` section should name the base branch. The user copies the printed file out verbatim.

Applying it here: `git am -3 <file>.patch` (fallback: `git apply --3way`, or manual edit if the base diverged).

Patch rules:
- **Only paths of this repo** (`handoff/`, `Template/`, `CLAUDE.md`, …). Never include the secure codebase, data, logs, or configs. If secure code lives inside this clone, keep it out of the handoff commits.
- **Text only**: no binary files (check `git diff --stat origin/main` shows no `Bin`). Figures travel as plot data (CSV/table in the report or a `.dat`/pgfplots/TikZ file), not images.
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
3. Add a line to `handoff/LOG.md` (date, direction, file, one-line summary, status) and commit, so the next pull brings the secure env up to date.
