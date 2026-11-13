# Independent review of draft v1 (12 Nov 2026), by a separate agent that had not seen the work being built.
# It describes the state BEFORE the fixes; CHANGELOG.md ('Changes after the independent review') lists what changed.

# Independent review: geospatial-xai-attacks, branch paper-draft-v1 (draft v1, 12 Nov 2026)

Scope: main runs seeds 0-4, feature and location designs (complete). I edited nothing in the repo. My recomputation
script is `/tmp/claude-0/-home-claude/9d31b9fa-a301-5bca-8db9-187e7d5aad50/scratchpad/recompute.py`. It reads the
`.npz`/`.json` files directly. CONFIRMED = I reproduced it with code. PLAUSIBLE = I reasoned it out but did not
reproduce it.

---------------------------------------------------------------------------------------------------------------------

## Critical

### C-1. Rule C2 (feature design) says "GeoShapley keeps more", but the text says there is no difference (CONFIRMED)
- **Numbers.** `decisions.csv`, C2 feature: GeoShapley minus KernelSHAP in K = **+0.050 ± 0.029 (95% CI +0.021 to
  +0.078)**, p = 0.009. All five seeds are positive: +0.039, +0.090, +0.035, +0.040, +0.043.
- **The auxiliary contrasts agree.** Top-3 is +4.9 pp ± 4.5. The leak at 90% fidelity is +5.0 pp ± 3.6. AUC is
  -0.051 ± 0.006. In this design GeoShapley is measurably, if slightly, harder to fool.
- **Text that contradicts it:**
  - `abstract.tex:8` and `fooling_geoxai_draft_v1.tex:63`: "GeoShapley is no harder to fool than KernelSHAP".
  - `results.tex:80`, section title: "GeoShapley is fooled like KernelSHAP".
  - `results.tex:81`, paragraph heading: "No detectable difference from KernelSHAP (rule C2)".
  - `results.tex:86-88`: "does not translate into a detectable difference in what the attacker can hide".
  - `discussion.tex:12-13`: "Under rule C2 we find no detectable difference".
  - `conclusion.tex:4`: "does not change this".
- **How it shows in the PDF.** On p. 10 the heading "No detectable difference" sits directly above the macro output
  "feature design, GeoShapley keeps more".
- **Fix.** Rewrite the C2 claims by design: "In the feature design GeoShapley keeps slightly more (+0.05, CI
  0.02-0.08); in the location design there is no detectable difference (0.00, -0.05 to +0.04)". Drop "no harder to
  fool" from the abstract, introduction and conclusion. Better still, drive these sentences from `\RuleCtwo...`
  through a conditional so that hard-coded prose cannot contradict a rule outcome.

### C-2. "Trigger in top 3" counts all 200 explained homes, but the prose reads it as penalised homes (CONFIRMED)
- **What the code does.** `run_matched_attack.py:341-345` (`summarise`) computes top-3 over all explained homes. The
  Methods text defines it the same way. About 84% of those homes are not penalised. On them the trigger carries a
  positive "bonus" attribution of about +0.04, and that bonus drives most of the metric.
- **Recomputed on penalised homes only (feature design, matched scaffold).** The trigger is in the top 3 for
  **99.5% (KernelSHAP) and 100% (GeoShapley)** of penalised homes. The location design gives 98.1% / 98.6%. At 50%
  fidelity it is 54.7% / 58.0% (paper: 17.4% / 19.3%).
- **Statements that are wrong or misleading:**
  - `discussion.tex:18-19`: "would still find the trigger among the leading features for about half of the penalised
    homes". This is false: it is essentially all of them.
  - `results.tex:35-36`: "the trigger drops out of the top three for about half of the homes".
  - `results.tex:67-69`: "Even at 50% fidelity … 17.4% … 19.3%".
  - `appendix.tex:103-104`.
- **Failure scenario.** A reader concludes that the scaffold hides the trigger on half the targeted homes. In fact
  only unpenalised homes lose it.
- **Fix.** Report top-3 separately for penalised and unpenalised homes, or define it on P, consistent with `a` and K.
  The C1' outcomes are unchanged either way (LIME's penalised-home top-3 is 0%; KernelSHAP and GeoShapley stay "partly
  fooled").

---------------------------------------------------------------------------------------------------------------------

## Major

### M-1. Incomplete runs leak into tables, rules and figures (CONFIRMED)
- **Cause.** Neither `summarize_matched_attack.py:102-108` nor `make_paper_tables.py:127-139` (`runs`) checks
  `d["complete"]`. Meanwhile `run_matched_attack.py:641-642` writes the JSON at the end of every budget-limited
  invocation, even when the run is partial.
- **What has already leaked.** `feature_seed0_m-mlp.json` has `complete=False` (8 of 10 cells). It is used for:
  - `tab_robust.tex`: the "Neural net" rows with Seeds = 1;
  - three rule-R rows in `decisions.csv`;
  - `\RobHold`/`\RobCells`, printed as "8 of 9";
  - `\HonestRtwoNet`, printed as "neural networks (0.81)";
  - the Neural net row in F8.
- **What will leak next.** The MLP runs for seeds 1 and 2 are running now, and so will the main seeds 5-9. A partial
  main JSON would enter the main tables with different seed sets per explainer and per metric. `ci()` silently drops
  NaNs, the table would show "--" per-seed cells, and C1 would use a different n for each explainer.
  `verify_paper_numbers.py:167-169` flags incomplete runs only in the main designs, never in robustness rows.
- **Fix.** Skip runs with `complete != True`, or keep a run only when all needed cells are present for every seed
  used. Require at least 2 seeds (preferably all planned seeds) before a row enters rule R. Print n in the prose.

### M-2. Location design: "to about zero at 50%" is contradicted by the intervals (CONFIRMED)
- **Where.** `results.tex:75-76`.
- **Numbers.** Net share at 50% fidelity: KernelSHAP **-0.15 ± 0.08** (CI -0.23 to -0.07); GeoShapley **-0.17 ±
  0.10** (CI -0.27 to -0.07). Both exclude 0.
- **What this means.** At 50% fidelity the explanation shows significantly less location effect than the honest
  model does, the same overshoot as LIME's -1.24.
- **Fix.** Say "below zero (…): the explanation shows less location than an honest model".

### M-3. Rule M is close to an identity check, not a test of the mechanism (PLAUSIBLE, reasoned; numbers CONFIRMED)
- **Why.** `mechanism_routing.py:117-151` builds "predicted" as a Monte Carlo average over coalitions with the exact
  Shapley weights (1/M per size, S uniform within a size, b uniform over the auditor's background B). It uses the same
  detector, threshold, f, psi and B as the attack. That is the same estimand as the exact KernelSHAP value or
  GeoShapley value it is compared with. Agreement can fail only through Monte Carlo noise or a bug.
- **The comparison is also mismatched.** Rule M compares a within-seed Monte Carlo error (MAE 0.0016 / 0.0031 /
  0.0016; max 0.006) with a between-seed SD (0.037 / 0.035 / 0.057).
- **Where the substantive claim actually rests.** "Share kept ≈ share of hybrids the detector cannot reject" (abstract
  l.10-11; introduction "show that it predicts the measured trade-off") depends on the routing average. That average
  was **not pre-specified**; its MAE is 0.019, 0.026 and 0.040.
- **The abstract wording also misleads.** Taken literally, "share of coalition hybrids the detector cannot tell from
  real homes" is the leak (53.6% / 60.8%), not K (0.67 / 0.72). The quantity that matches K is the average over
  coalition sizes.
- **Fix.** Present M as a consistency check of the efficiency and size decomposition. State that the routing
  approximation is descriptive. Say "average over coalition sizes" in the abstract.

### M-4. "Fidelity" is measured on all explained homes, but the prose treats it as bias delivered (CONFIRMED)
- **Why it matters.** For unpenalised homes f = h, so serving them f delivers no bias.
- **Fidelity on penalised explained homes at τ\* (recomputed from the cached detectors):**

  | Design | KernelSHAP | GeoShapley | LIME |
  |---|---|---|---|
  | Feature | 85.9% | 87.9% | 97.4% |
  | Location | 84.7% | 82.1% | 98.9% |

  Seed 4 is lower: 71% / 74% (feature) and 61% / 68% (location).
- **Statements affected:**
  - Introduction: "serves the bias to about 90% of real homes".
  - Abstract: "95.9% of real homes still receive it".
  - `discussion.tex:23`: "cost in bias actually delivered".
  - F3, whose x axis is this fidelity.
- **Fix.** Report fidelity on penalised homes alongside overall fidelity, and use it in any statement about bias
  delivered.

### M-5. Discussion claim about GeoShapley's middle coalitions is not supported (CONFIRMED from F7 data)
- **Where.** `discussion.tex:10-12`: "the middle coalitions are as far from the data as KernelSHAP's".
- **What the data show.** GeoShapley's routing rate is above KernelSHAP's at every coalition size: minimum 0.53 vs
  0.48; ends 0.91 / 0.88 vs 0.87 / 0.86. This fits C2 "keeps more".
- **Caveat.** Each explainer has its own detector, and M differs (10 vs 11 players).
- **Fix.** Drop the claim or qualify it.

---------------------------------------------------------------------------------------------------------------------

## Minor

### m-1. `verify_paper_numbers.py` can pass while the paper is wrong (CONFIRMED)
- **Current result.** It passes: 412 checks, 0 mismatches.
- **Coverage.** It covers only 52 of the 127 macros used in the text. Not checked: every rule outcome, C1/C2/C3, AUC,
  GapTotal/Reloc, the GeoShapley decomposition, the mechanism, E7, robustness, NPen/SharePen/TriggerMoran.
- **Not independent.** Fidelity and leak are read from the same JSON fields the pipeline uses, and top-3 uses the
  identical rank code.
- **Compares the wrong file.** It checks `numbers.tex`, not the PDF, so a stale PDF passes.
- **Silent gaps.** Macros that are missing from `numbers.tex` are skipped without a warning, and incomplete
  robustness runs are not checked.
- **Blind to prose.** C-1 and C-2 are invisible to it.
- **Fix:**
  - Recompute fidelity from the detector pickles (or at least from the score histograms).
  - Check the decision outcomes and the C2/C3/M/E7 macros.
  - Grep the PDF text.
  - Fail on any missing macro.

### m-2. "On the same curve … within the seed intervals" (`results.tex:71-73`) rests only on overlapping intervals
- **Paired check (CONFIRMED).** Generic minus the matched curve interpolated at the generic detector's fidelity:
  - feature: KernelSHAP +0.053 ± 0.087, GeoShapley +0.039 ± 0.097 (not significant);
  - location, GeoShapley: +0.080 ± 0.072 (p = 0.036, uncorrected).
- **Fix.** Keep the claim to the feature design, or test it paired.

### m-3. Location top-3 compares two coordinates against one feature (CONFIRMED)
- **What happens.** For LIME and KernelSHAP the trigger score is |φ_UTM_X| + |φ_UTM_Y|, ranked against single
  features. GeoShapley's trigger is one GEO player.
- **Effect.** Location top-3 is inflated for LIME and KernelSHAP, so the C2-aux location top-3 difference (-11.6 pp)
  does not compare like with like.

### m-4. The location row of the mechanism table uses the raw share, not the net one (CONFIRMED)
- **What happens.** "Measured share kept" there is raw K (0.65), while the main location table reports net K (0.48).
- **Fix.** Label it "raw".

### m-5. Coalition count is right only for the feature design
- **Where.** Methods text: "2^M - 2 … (2,046 here)".
- **Issue.** In the location design M = 10, which gives 1,022 coalitions.

### m-6. Appendix C cites a file that does not exist (CONFIRMED)
- **Where.** Appendix C cites `docs/paper/data/queues_run.txt`, which does not exist.

### m-7. The E7 decision is borderline, and the abstract overstates what the check does
- **Borderline outcome.** E7 is "go" on a point estimate of 0.811 against the 0.80 threshold; the CI is 0.775-0.847.
- **"Finds" is too strong.** The abstract says the check "finds the penalised homes", but the AUC needs the true
  labels. The label-free LISA low-low step recalls 29% of penalised homes at δ = 0.10, with 65% precision.
- **Other details:**
  - E7 uses seeds 0-9, while the attack uses seeds 0-4.
  - The LISA conditional permutation draws neighbours with replacement (`run_residual_diagnostic.py:58`).

### m-8. "Most of it to grade" overstates the evidence (CONFIRMED)
- **Where.** `results.tex:111`.
- **What the data show.** Grade is the modal top recipient. It takes 0.074 of the 0.119 net relocated gap for
  KernelSHAP (with UTM_Y at -0.098) and 0.056 of 0.113 for GeoShapley.

### m-9. In the feature design, the GEO change is not hidden penalty
- **What the change is.** GEO going from -0.11 to +0.02 is the legitimate location effect disappearing; in this
  design GEO never carried the penalty, which sits on the trigger player.
- **Where the claim does hold.** "A GEO map … would understate the penalty" (`discussion.tex:27`) applies only to the
  location design (GEO kept (-0.157 + 0.108) / (-0.351 + 0.108) ≈ 0.20 of the excess).

### m-10. Mechanism caveat: background rows are memorised by the detector (PLAUSIBLE)
- **Why.** The Shapley backgrounds are training rows. The matched detector was trained on those rows as "real" with a
  fully grown random forest.
- **Effect.** Near-empty coalitions look real partly because they sit next to memorised positives (ρ_1 = 0.87).
- **Partial evidence.** The k-means-background run (K 0.64 vs 0.67) addresses this only partly.
- **Fix.** Discuss it, or add an audit run whose background comes from rows the attacker never saw.

### m-11. Figure and macro details
- **F8.** The "fully fooled below 0.10" annotation ignores the top-3 and fidelity parts of C1'. The n = 1 "Neural net"
  row is plotted without marking.
- **F7.** The legend shows only a green "routing average" marker, but blue crosses are also plotted.
- **F1.** Longitude tick labels overlap, and the legend sits on top of points.
- **F5.** The 0.80 reference line is nearly invisible.
- **Raster agreement.** `\RasterAgreeHi` = 100.1%: a t-interval on a bounded proportion.

### m-12. Not yet produced (expected while runs are in progress)
- **Missing macros.** `\HonestRtwoForest` and the `\PenTen*` macros print as red "??" placeholders.
- **Missing outputs.** `tab_autocorr`, `tab_penalty010`, `tab_main_nocoord` and F6 are not generated.
- **C5.** C5 reads "fewer than three levels ran: no claim".

---------------------------------------------------------------------------------------------------------------------

## Checked and fine

### Metric definitions
- **Signed attribution, K and K_net.** Signed attribution is the mean over penalised explained homes of the
  trigger-set sum. K = a_dep / a_biased; K_net = (a_dep - a_hon) / (a_biased - a_hon).
- **Trigger columns.** The pct column (feature design); UTM_X + UTM_Y (location design, LIME and KernelSHAP); GEO
  (location design, GeoShapley).
- **GeoShapley flattening.** Checked against `geoshapley.geoshap_to_shap`: primary + interaction/2, and GEO + Σ
  interaction/2. `np.allclose` holds in every npz file. Column order puts the coordinates last.
- **Top-3 ties.** Strict and non-strict ranking give identical results in every cell.
- **Honest Shapley cells skipped in the feature design.** This is valid because h ignores the trigger, so its
  attribution is exactly 0.

### Threshold and leakage
- **τ\*.** Set from the 400 calibration homes (`method="lower"`, so calibration fidelity ≥ 90%: 90.8% and 90.5%).
  Fidelity is reported on the 200 explained homes (89.5% and 89.7%).
- **No leakage.** Every detector (generic, matched LIME, harvested) is trained on training rows only. The explained
  homes play no part in detector training or in choosing τ\*. The `leak_at_fid*` auxiliary numbers do set their
  threshold on the explained homes, but they feed only C2-aux.

### Leak and AUC
- **Leak.** Recomputed from the histograms, it equals the routed count exactly in every cell; query totals match.
- **AUC.** The histogram AUC matches sklearn (0.7705).

### Statistics
- **Intervals.** t-intervals use df = n - 1 (both scripts and E7).
- **Holm.** The implementation is correct.
- **C1 (Holm p).** Reproduced:
  - feature: 1.3e-8, 2.0e-4, 2.0e-4;
  - location: 1.2e-4, 3.3e-4, 3.3e-4.
- **C1' outcomes.** Follow correctly from the numbers.
- **C2.** Reproduced: +0.0495 ± 0.0286 (feature) and -0.0033 ± 0.0470 (location).
- **C3 (mat minus biased).** Reproduced:
  - feature: GEO-pen +0.123 ± 0.018, mean |GEO| -0.055 ± 0.005, interactions +0.086 ± 0.006;
  - location: +0.194 ± 0.032, -0.074 ± 0.004, +0.070 ± 0.004.
- **E7 and R.** The coded logic matches the paper's rule text.

### Numbers traced from PDF to macro to raw files
**Main and frontier tables**
- K, matched: LIME 0.02, KernelSHAP 0.67 (0.60-0.74), GeoShapley 0.72 (0.65-0.79).
- Fidelity: LIME 95.9%, KernelSHAP 89.5%, GeoShapley 89.7%.
- Top-3: 67.3 / 86.3 / 86.6 (biased); 51.7 / 56.6 (matched).
- Generic detector: LIME leak 95.8%, LIME K 0.96; KernelSHAP 0.52, GeoShapley 0.54, fidelity 74.4%.
- Frontier: 0.47 / 0.24 (KernelSHAP), 0.50 / 0.29 (GeoShapley).
- Detector AUC: 1.00 / 0.78 / 0.72; GeoShapley minus KernelSHAP -0.05.
- Leak, matched: 53.6% / 60.8%.

**Location design**
- Honest KernelSHAP signed -0.12. LIME raw 0.05, top-3 0.8% vs 66.1%, net -1.24.
- Net K 0.48 / 0.48, raw 0.65 / 0.65; fidelity 90.0 / 88.8; top-3 95.8 / 84.2.
- Net K at 75% fidelity 0.17 / 0.14; at 50% -0.15 / -0.17.

**GeoShapley decomposition**
- GEO on penalised homes -0.11 → 0.02 (feature) and -0.35 / -0.11 → -0.16 (location).
- Interactions 0.07 → 0.16 (×2.16) and 0.07 → 0.14 (×1.96), so "roughly double" holds.

**Gap and mechanism**
- Gap total 0.79 / 0.83; relocated 0.12 / 0.11; LIME gap -0.01.
- Routing rate: first 0.871, last 0.859, minimum 0.479 at k = 6.
- MAE / SD / routing MAE: 0.0016 / 0.0371 / 0.0194; 0.0031 / 0.0349 / 0.0265; 0.0016 / 0.0570 / 0.0402.

**E7**
- Local-mean AUC 0.811 ± 0.037; honest flag rate 3.9%.

**Data**
- 15.9% of test homes penalised; 32.2 penalised among the explained homes; raster agreement 99.97%.
- Median price $435,000; 334 tracts; maximum tract share 0.54 (the F1 colourbar is fine).
- A 0.30 log10 penalty halves the price; 0.10 is a 20.6% cut.

### Prose and figures
- **Prose that holds:**
  - "about half of the penalty" (location net 0.48);
  - "between half and three quarters";
  - "roughly double";
  - "GeoShapley's queries somewhat harder to tell" (AUC -0.051 ± 0.006).
- **Figures.** F2, F3, F4, F5 and F7 match the tables and their captions.
- **Run filters.** `verify_paper_numbers.py` uses the same filters as the paper for main runs: xgb, audit match,
  penalty 0.3, attacker rf, shuffle 1, harvest scale 1, seed 42 excluded.
