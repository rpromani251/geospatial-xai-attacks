# Changelog

Corrections and method changes that affect numbers quoted in earlier project documents. Newest first.
The paper draft (`docs/paper/fooling_geoxai_draft_v1.tex`, Appendix D) states the same corrections.

## 2026-11-23 — Cook County, continuous penalty, never-seen background, seeds 5-9, audit report

- **Cook County, Illinois** as a second metropolitan area (`scripts/process_cook_county.py`, `--metro cook`): 3,000
  single-family arm's-length 2015 sales from the Cook County Assessor's modelling data, eight structural features
  mapped to King County's, ACS 2011-15 tract shares joined by tract id. The location design's coordinate lookup is a
  25 m nearest-sale raster over all 376,483 geocoded sales (the assessor's data have tract ids but no polygons), so
  the raster-agreement check is 100% by construction there. 48.6% of test homes are penalised (King County 15.9%).
  Reported in `tab_cook.tex` with rule R-Cook (same C1' class as King County): 5 of 6 cells agree. LIME in the
  location design is "partly fooled" by the rule because the summed noise of its two coordinate attributions puts
  them in the top three for 9% of homes, though its share kept is negative.
- **Continuous penalty** `--penalty-shape ramp`: f = h - 0.30 min(1, T/0.5). Not pre-specified. `tab_ramp.tex`.
- **Background of homes the attacker never saw** `--audit-bg-kind unseen`: each half of the 200 explained homes is
  explained against the other half. Not pre-specified. Added to `tab_local.tex`; the share kept is about 0.05 lower
  than with the default background, so the main runs slightly favour the auditor. The limitation that asked for this
  test is replaced.
- **Main seeds 5-9** run (q_main_late), so the main results use seeds 0-9. The local/unseen table compares every row
  on seeds 0-4 (`Main...LocalTab` macros).
- **Typical audit report** (`scripts/audit_report.py`, `tab_audit.tex`, Figure F10): a summary check (protected
  attribute in the top three by mean |attribution|) and a disaggregated check (one-sided t-test on penalised homes,
  plus a materiality threshold of 0.05 log10). Not pre-specified.
- `mechanism_routing.py --metro cook`; `make_paper_tables.py` writes the Cook split terms and the efficiency prediction
  -delta (1 - p_B) of the biased model's attribution.
- `verify_paper_numbers.py` also checks Cook County, the continuous penalty, the never-seen background and the audit
  checks.
- Text dollar signs in the paper use `\usd` (math dollar): `\$` fell back to a bitmap TS1 glyph that microtype could
  not expand, which stopped the build once the Cook paragraph moved it onto a stretched line.

## 2026-11-22 — theory section and local-background test

- **Title**: "Geographic awareness is no defence: scaffold attacks hide a racial price penalty from LIME, KernelSHAP
  and GeoShapley" (was "The estimator, not the geography: ...").
- **Two propositions** (Section 4.6, proofs in Appendix E): an exact split of the share kept into a penalty term and a
  cross term (feature design), with the corollary that the split at one penalty predicts the share kept at another;
  and a routing floor, valid for any detector, set by the dependence between features that an explainer's coalition
  hybrids break.
- `mechanism_routing.py` now writes the split terms (`kept_penalty_term`, `kept_cross_term`, `rho_with_avg`,
  `rho_bgpen_avg`, `p_bg_penalised`) and asserts that they add up to each layer; `make_paper_tables.py` writes
  `tab_split.tex` with the prediction at delta = 0.10 against the separate delta = 0.10 runs.
- **New option** `--audit-bg-kind local` (each explained home's Shapley background = its 100 nearest training homes)
  and `--attacker-bg-kind local` (the attacker harvests its detector's training queries the same way). Not
  pre-specified; motivated by the floor. Reported in `tab_local.tex`, outside rule R.
- `verify_paper_numbers.py` also checks the split identity and the delta = 0.10 prediction.

## 2026-11-12 — matched-scaffold experiment and paper draft v1

### Corrected claims

| Earlier claim | Status | Why | Where it still appears |
|---|---|---|---|
| The 0.30 log-price penalty is "≈ 26% in linear price" | **Wrong.** Prices are `log10`, so 0.30 halves the predicted price (×10^−0.3 ≈ 0.50); the 0.10 penalty is a 21% cut. 26% assumed natural logs. | Base of the log | `FOOLING_XAI.md` l.42, `docs/project_overview.md` l.94, `docs/experiments_narrative.md` l.32 |
| "Geospatial setting makes LIME more resistant than SHAP (opposite of Slack's tabular results)" / "LIME partially resists" | **Withdrawn.** It came from the generic (Slack-recipe) LIME detector, trained on Gaussian noise in raw units, which lets ~96% of LIME's real (quartile-sampler) queries through to the biased model. With a detector matched to LIME's sampler LIME is fully fooled, the ordering Slack et al. report. | Detector mismatch | `CLAUDE.md` l.86, `docs/experiments_narrative.md` l.117, `FOOLING_XAI.md` |
| "GeoShapley `UTM_Y` collapses 93%" (0.013 → 0.009) | **Withdrawn.** Computed with the retired custom GeoSHAP, and divided an adversarial GeoSHAP value by an honest KernelSHAP one; the October summary had already corrected it to −58% under the generic detector. The paper reports the penalty share kept instead. | Wrong reference and retired code | `project_summary.md` l.115, `FOOLING_XAI.md` l.76, `docs/project_overview.md` l.455/545/652/777, `docs/experiments_index.md` l.27 |
| 16 October attack-settings table: "SHAP-matched" detector reaches high hiding | **Superseded.** High hiding came from routing only 7.5% of real homes to the biased model. The new threshold rule fixes fidelity ≥ 90% on held-out calibration homes before hiding is measured. | No fidelity constraint | 16 Oct summary |
| Draft v0: trigger in the top three for 13% (KernelSHAP) and 16% (GeoShapley) of homes under the matched scaffold | **Superseded.** Synthetic stand-in for the trigger column, one seed. The real column and five seeds give the numbers in Table 2 of draft v1. | Synthetic trigger, one seed | `docs/paper/draft_v0_manuscript.md` |

### Changes after the independent review (same day)

- **Top three and fidelity split by penalised homes.** "Trigger in top 3" was computed over all 200 explained homes.
  On penalised homes the trigger stays in the top three for ~100% of homes under the matched Shapley scaffolds; the
  fall to about half came from unpenalised homes. Fidelity on penalised homes (the bias actually delivered) is now
  reported beside fidelity on all homes (`scripts/fidelity_by_group.py`, from the cached detectors). Text that said
  the trigger "drops out of the top three for about half of the homes" or "about half of the penalised homes" was
  wrong and is rewritten.
- **C2 wording.** The feature-design outcome is "GeoShapley slightly harder to fool" (+0.05, interval +0.02 to
  +0.08); "GeoShapley is no harder to fool" is withdrawn from the abstract, introduction, results, discussion and
  conclusion. Prose now takes the clause from the rule macro.
- **Incomplete runs excluded.** Tables and rules read only runs whose JSON says `complete`; a robustness configuration
  needs at least three complete seeds. A partial neural-net run had entered the robustness table with n = 1.
- **Location design at 50% fidelity**: the net share is below zero (intervals exclude 0), not "about zero".
- **Rule M** is presented as a check of the size decomposition (it evaluates the same Shapley average); the routing
  average is the descriptive approximation.
- **Outcome check**: "finds the penalised homes" replaced by the AUC (which uses labels) and the label-free LISA
  recall; the E7 threshold is met narrowly.
- **Generic detector against the matched curve** is tested paired by seed; "on the same curve" now holds only in the
  feature design.
- **`verify_paper_numbers.py`** extended to rule outcomes, C2/C3, mechanism, robustness, E7 and fidelity recomputed
  from the detectors, and checks the compiled PDF.

### Method changes (all seeds recomputed from scratch with the final code)

- **KernelSHAP `l1_reg=False`.** shap's default `l1_reg="num_features(10)"` keeps at most ten features per home even
  under full coalition enumeration; with eleven features it zeroed one. At seed 42 the trigger attribution changed by
  < 0.001. Values are now exact Shapley values for the background.
- **GeoShapley seeded.** In the feature design (nine non-location players) the package samples about 600 of 1,024
  coalitions with NumPy's global RNG; `np.random.seed(bg_seed)` is now set before each run, so results are
  reproducible. The location design (eight players) is exact.
- **Model families** (`--model rf|mlp`): random forest (100 trees, min leaf 3, max_features 0.5) and MLP (256-128,
  standardised inputs, early stopping), feature design only. The forest was cut from 300 to 100 trees for run time.
- **Audit settings the attacker did not train for** (`--audit-bg`, `--audit-bg-kind kmeans`, `--audit-lime-samples`,
  `--audit-lime-sampler gaussian`, `--audit-ks-l1`): the detector stays the one trained for the defaults.
- **Environment.** All paper runs on one Linux x86_64 machine, Python 3.10, numpy 2.2.6, scikit-learn 1.7.2,
  xgboost 3.2.0, shap 0.49.1, lime 0.2.0.1, geoshapley 0.2.1.1. The seed-42 reference run made earlier on Linux aarch64
  with the same versions was reproduced to ~1e-15 before any run.

### New

- `scripts/run_matched_attack.py`, `summarize_matched_attack.py`, `mechanism_routing.py`, `run_residual_diagnostic.py`,
  `make_paper_tables.py`, `make_paper_figures_matched.py`.
- `docs/paper/`: draft v1 (LaTeX); every number is a macro generated from the result files; decision-rule outcomes
  in `docs/paper/data/decisions.csv`.
