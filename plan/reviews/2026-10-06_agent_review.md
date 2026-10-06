# Internal review of the draft — 2026-10-06

Source: an AI agent's review of the paper draft (state of `main` around commit `8b49543`), pasted by the user.
Below: the review verbatim, then our assessment and the decisions taken. Open items are tracked at the end.

## Review (verbatim)

> The analysis and benchmark are strong. The main risks are that the method has no results yet, and that the
> native-resolution ground truth may limit what any method can show on fidelity metrics. Here they are, ranked by how
> likely they are to sink the paper.
>
> **Critical**
>
> 1. There are no METHOD numbers. The abstract, contributions, and Tables 2–5 are all placeholders, as are the teaser
>    and qualitative figures and the human study. The training run, about 14 ablations, 5 plug-in anchors, retrained
>    baselines and a 2AFC study is a lot of work before the deadline. Consider cutting the generative variant and the
>    per-band map ablation now.
> 2. The f/22 target may be diffraction-limited at native resolution. At f/22 with the 5D IV's roughly 5.4 µm pixels,
>    the diffraction cutoff is near 0.44 cycles/px. The diffraction MTF is about 0.65 at the anchor Nyquist (1/8) and
>    only about 0.2 at 0.3 cycles/px. Your paired edge measurement compares the target to the f/4 input, not to the true
>    scene, so a reviewer can argue the ground truth lacks exactly the high band you add. The consequence is that PSNR
>    will penalize real transferred texture. You already note that PSNR rewards smoothing. Expect a fight here, so do
>    two things:
>    * Add a diffraction/MTF paragraph.
>    * Make DISTS plus the human study the primary evidence, with fidelity framed as "no loss".
> 3. The anchor lock caps any PSNR gain. Since ŷ↓4 = a, METHOD can only win in the native high band, where the target
>    is weakest (see point 2). A realistic outcome is +0.0–0.2 dB over bicubic. The paper's own framing ("the empty
>    region in Fig. 3") is the right bar, but it needs CIs showing METHOD dominates both bicubic and the non-learned
>    exemplar transfer. That transfer already reaches DISTS 0.232 within 0.2 dB of bicubic. If the learned model only
>    modestly beats it, reviewers will ask why it needs to be learned.
>
> **High**
>
> 4. The obvious baseline is missing. The "retrained rows" are empty: deblurrers fine-tuned on native crops, ideally
>    with larger receptive fields or multi-scale inference. Reviewers will ask for this first. If a fine-tuned native
>    model gets close, the motivation weakens. Run it early.
> 5. The evaluation uses a single dataset and camera. Training and test are both DPDD on a Canon 5D IV with static
>    scenes. You need at least qualitative results on other cameras or phones, and ideally a second quantitative set.
>    DP access also limits the "any camera" framing. The predicted-map head is still a TODO, and its cost decides
>    whether the method is practical.
> 6. Your re-rendering shifts results and reorders models. Shifts range from −0.58 to +0.16 dB, and the best model at
>    ×4 flips from LaKDNet (original) to Bokehlicious (yours). Then Bokehlicious is chosen as the "strongest" anchor on
>    your rendering. Reviewers may read this as either a rendering bias or anchor cherry-picking. Report everything with
>    both renderings, and justify the anchor choice independently.
> 7. The method can invent detail too. Transferring in-focus texture or text into a defocused region is "real" but can
>    still be wrong: wrong glyphs, or brick from a different wall. Run the same text-crop and hallucination analysis on
>    METHOD that you used against OSEDiff. Otherwise the main criticism of the baselines turns back on you.
>
> **Medium**
>
> 8. Novelty may read as a combination of known parts. Back-projection, RefSR texture transfer, internal patch
>    recurrence, SFT conditioning, and retrieval in the style of ReFIR are all prior work. Make the contribution about
>    the insight: blur-dependent sources of detail and matching in the anchor across focus.
> 9. The claim that native-resolution deblurring is "not studied" is risky. Bokehlicious already trains on
>    multi-megapixel data, and UHD deblurring work exists. Do a literature check and soften the wording to
>    "not evaluated".
> 10. Coverage is unmeasured. How many test scenes actually have in-focus counterparts? Report memory abstention rates,
>     and show failure cases.
> 11. Runtime is missing. DINOv2 plus a U-Net with tile attention on 30 MP images could be slow, and the runtime column
>     is empty.
> 12. The DPDD raw data may have licensing limits. Check that you're allowed to redistribute the rebuilt benchmark
>     before claiming it as a contribution.
>
> **Before submission**
> Remove internal notes, which are still in the text in many places: "refresh from handoff", "which ×4 rendering leads
> is open", "only if the feed-forward model leaves a visible gap…", and "sizes; whether a larger model…". Also fill the
> "–" cells and rebuild Fig. 3, which currently shows as raw axis text. Watch number consistency too: the focal-plane
> damage figure appears as −1.6 dB in the intro and −1.0/−1.1 in Table 3 because they use different anchors, so state
> which anchor each number refers to.

## Assessment (outside agent)

| # | Verdict | Notes |
|---|---|---|
| 1 | agree | Scope is the main schedule risk. Proposed cuts: generative variant → future-work sentence; ablations ~14 → 5–6 that test claims (lock, native input, exemplars, oracle vs retrieved, DINOv2 vs pixels, anchor mix); plug-in anchors 5 → 2–3; 2AFC only if someone runs it. |
| 2 | agree on the paragraph; partly answered by data | Numbers check out (cutoff 0.44 c/px; MTF 0.65 at 0.125, 0.21 at 0.3, ≈0.04 at 0.4). Our paired edge MTF (336 edges, 37 scenes): f/22 / f/4 = 1.05 / 1.24 / 0.99 / 0.64 at 0.1 / 0.2 / 0.3 / 0.4 c/px — the target holds the detail the camera records in focus up to 0.3 c/px, which is the detail we transfer. PSNR penalises only above ≈ 0.3 c/px. PSF-matched sensitivity metric already in the protocol (`docs/evaluation.md` §2.5). |
| 3 | agree (by design) | The lock fixes everything the anchor determines; the gain over bicubic of the same anchor is the native high band only. The claim is the empty region of Fig. 3 with CIs. "Why learned?" vs the non-learned exemplar transfer (DISTS 0.232): first evidence from v0_oracle vs v0 (handoff 5). |
| 4 | agree — strongest point | Fine-tune Restormer / DRBNet on 512 px native crops (the training cache makes it cheap). |
| 5 | valid; needs the user | Other DP / high-MP captures for at least qualitative results; without them drop "any camera". Predicted blur map decides practicality. |
| 6 | agree | Justify Bokehlicious independently (not trained on DPDD → no train/test gap), not as "strongest"; both renderings stay in Table 1. |
| 7 | agree, cheap | Same text-crop / hallucination diagnostics on our outputs; log the attention on the null exemplar (also answers 10). |
| 8 | agree | Insight first: blur-dependent sources of detail, matching across focus in the anchor. |
| 9 | agree | The phrase "not studied" was in the intro ("to our knowledge, not been studied"). |
| 10 | agree | M2 coverage (cache) + null-attention rate at inference + failure cases. |
| 11 | agree | Handoff-5 V8 records time per image. |
| 12 | agree | Release rebuild scripts, not images; check the DPDD licence. |
| Before submission | agree | The internal notes are `\todo{}` markers (≈ 50), cleared before submission. Fig. 3 renders correctly in our build (checked page 7) — raw axis text is an extraction artefact on the reviewer's side. −1.6 dB = Restormer anchor + OSEDiff, −1.0 / −1.1 = DRBNet anchor. |

## Decisions (user, 2026-10-06)

| Item | Decision | Status |
|---|---|---|
| Diffraction / ground-truth sharpness paragraph (#2) | do it | **done** — `sec/4_experiments.tex` §4.1 "Ground-truth sharpness" (cutoff, diffraction MTF, paired MTF ratio, PSNR penalises correct detail above ≈ 0.3 c/px, PSF-matched sensitivity variant `\todo{numbers}`, DISTS / LPIPS measure restored detail) |
| Scope cuts (#1) | not now; revisit | open |
| Anchor labels on dB numbers (review "before submission") | do it | **done** — intro (focal plane: −1.0 to −1.1 dB DRBNet anchor, up to −1.6 dB Restormer anchor; +1.5–1.7 dB depending on the network), abstract (1.3 dB against bicubic of the same result), §4 ceilings (iii) |
| Novelty wording (#8, #9) | do it | **done** — intro: "not been studied" → "no defocus deblurring method has been evaluated against ground truth at the native resolution …", mentions larger-image training and UHD restoration; contribution bullet leads with the insight and names the known components |
| Fine-tuned native deblurrer baseline (#4), hallucination check on the method (#7) | later | open — candidates for handoff 6 |

## Open items (for later)
- #1 scope cuts (revisit).
- #4 fine-tuned native deblurrers (Restormer / DRBNet on native 512 px crops; larger receptive field / multi-scale).
- #7 text-crop + hallucination analysis on the method; #10 null-attention (abstention) rate, coverage, failure cases.
- #5 second camera / dataset (user to check what exists); predicted blur map.
- #6 anchor-choice justification in the text (not trained on DPDD).
- #11 runtime column (from handoff 5 V8).
- #12 DPDD licence check; release scripts rather than images.
- PSF-matched sensitivity numbers for the new paragraph.
