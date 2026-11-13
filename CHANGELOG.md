# Changelog

Corrections and method changes that affect numbers quoted in earlier project documents. Newest first.
The paper draft (`docs/paper/fooling_geoxai_draft_v1.tex`, Appendix D) states the same corrections.

## 2026-11-12 — matched-scaffold experiment and paper draft v1

### Corrected claims

| Earlier claim | Status | Why | Where it still appears |
|---|---|---|---|
| The 0.30 log-price penalty is "≈ 26% in linear price" | **Wrong.** Prices are `log10`, so 0.30 halves the predicted price (×10^−0.3 ≈ 0.50); the 0.10 penalty is a 21% cut. 26% assumed natural logs. | Base of the log | `FOOLING_XAI.md` l.42, `docs/project_overview.md` l.94, `docs/experiments_narrative.md` l.32 |
| "Geospatial setting makes LIME more resistant than SHAP (opposite of Slack's tabular results)" / "LIME partially resists" | **Withdrawn.** It came from the generic (Slack-recipe) LIME detector, trained on Gaussian noise in raw units, which lets ~96% of LIME's real (quartile-sampler) queries through to the biased model. With a detector matched to LIME's sampler LIME is fully fooled, the ordering Slack et al. report. | Detector mismatch | `CLAUDE.md` l.86, `docs/experiments_narrative.md` l.117, `FOOLING_XAI.md` |
| "GeoShapley `UTM_Y` collapses 93%" (0.013 → 0.009) | **Withdrawn.** Computed with the retired custom GeoSHAP, and divided an adversarial GeoSHAP value by an honest KernelSHAP one; the October summary had already corrected it to −58% under the generic detector. The paper reports the penalty share kept instead. | Wrong reference and retired code | `project_summary.md` l.115, `FOOLING_XAI.md` l.76, `docs/project_overview.md` l.455/545/652/777, `docs/experiments_index.md` l.27 |
| 16 October attack-settings table: "SHAP-matched" detector reaches high hiding | **Superseded.** High hiding came from routing only 7.5% of real homes to the biased model. The new threshold rule fixes fidelity ≥ 90% on held-out calibration homes before hiding is measured. | No fidelity constraint | 16 Oct summary |
| Draft v0: trigger in the top three for 13% (KernelSHAP) and 16% (GeoShapley) of homes under the matched scaffold | **Superseded.** Synthetic stand-in for the trigger column, one seed. The real column and five seeds give the numbers in Table 2 of draft v1. | Synthetic trigger, one seed | `docs/paper/draft_v0_manuscript.md` |

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
