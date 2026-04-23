# geospatial-xai-attacks

## Purpose
UROP research project: adversarial and manifold-aware robustness analyses for explainability methods (LIME, SHAP, GeoShapley) in geospatial contexts. See `task_summary.md` (in vault) for the full 3-part framework design.

## Status
**Active thread (as of 2026-04-22):** SBM-as-explainer — rung-3
behavioral fingerprint + finite-K pEx SBM as a Slack-style scaffold-
attack detector. Four-dataset replication complete (housing, Slack
COMPAS/German/CC). Mechanism identified (anchor-dominance erosion,
unified as m-field variability + layer-misalignment). K-sweep and
tanh-sweep confirm mechanism; sinusoidal-sign Lipschitz sweep retired.
Thursday April 23 meeting with Peng to decide paper shape
(workshop/TMLR vs NeurIPS main).

**Entry point for resume:** `docs/sbm_explainer/README.md`, then
`docs/sbm_explainer/day_log_apr22.md` for the narrative of today's
work.

**All 6 original geospatial steps complete:** baseline XGBoost
(R²=0.89), SHAP+LIME+GeoSHAP explainers, adversarial scaffold attack,
spatially-aware neighborhoods, 3×5 robustness matrix, k-NN
trustworthiness, worst-case x* sweep.

## Project Rules
- Experiment code goes in `src/`
- Runnable scripts go in `scripts/`
- Do not store raw data in git (handled by `.gitignore`)
- Save outputs under `results/`
- Config files go in `configs/`
- Make minimal, reviewable changes

## Structure
```
src/
  attacks/       # Adversarial perturbation logic
  data/          # Data loading utilities
  evaluation/    # Metrics and comparison logic
  explainers/    # LIME, SHAP, GeoShapley wrappers
  models/        # Model training
  utils/         # Shared helpers (paths, I/O)
scripts/         # Entry-point scripts
configs/         # YAML config files
data/            # raw/, interim/, processed/ (git-ignored)
notebooks/       # Exploratory notebooks
results/         # Experiment outputs (git-ignored)
docs/            # Roadmap, experiment registry
papers/          # Cloned reference repos (git-ignored)
```

## Dependencies
numpy, pandas, scikit-learn, pyyaml, matplotlib, lime, shap, xgboost, geopandas, shapely, pyproj, jupyter

## Implementation Plan

| Step | Name | Goal | Status |
|------|------|------|--------|
| 1 | Baseline | Load Seattle housing, train XGBoost, save metrics | Done |
| 2 | Explainers | KernelSHAP + LIME wrappers on housing model | Done |
| 3 | Adversarial Scaffold | OOD detector + biased/innocuous models (digital redlining) | Done |
| 4 | Spatially-Aware Neighborhoods | 3-level perturbation generator (random, medoid, spatial) | Done |
| 5 | Robustness Analysis | 3×5 results matrix (Tree/Kernel/Geo/LIME/Ensemble) + trustworthiness classification | Done |
| 6 | Worst-Case x* | argmax search over Level 3 neighborhood | Done |

## Dataset
- **Housing**: `data/raw/seattle_sample_3k.csv` — 3000 records, 10 features + log_price, UTM coords
- **Census**: `data/processed/king_county_demographics.csv` — 398 tracts, pct_black/pct_latino/pct_black_latino
- **Spatial join**: `data/processed/seattle_housing_with_demographics.csv` — 3000 records matched to 334 tracts
- **Shapefiles**: `data/raw/tl_2015_53_tract/` — TIGER/Line 2015 WA census tracts
- Processing script: `scripts/process_census.py`

## Reference Repos
| Paper | Local Path | Key Files |
|-------|-----------|-----------|
| Fooling LIME & SHAP (Slack et al.) | `papers/fooling-lime-shap/` | `adversarial_models.py`, `COMPAS_Example.ipynb`, `compas_experiment.py` |

## Key Results (Step 3)
- SHAP attack succeeds: UTM_Y importance −57%, pct_black_latino hidden (0.035). OOD accuracy 99.5%
- LIME attack partially fails: UTM_Y stays #1, pct_black_latino visible (0.108). OOD accuracy 100%
- Novel finding: geospatial setting makes LIME more resistant than SHAP (opposite of Slack's tabular results)
- Innocuous model R²=0.64 (8 safe features), biased model penalty=0.3 log_price on tracts >25% minority
- Visualizations: `scripts/plot_results.py` → 2×2 summary figure + individual honest-vs-adversarial bar charts in `results/figures/`

## Next Steps (SBM-as-explainer thread, 2026-04-22)
- [ ] Thursday meeting with Peng (2026-04-23): three questions — (1) lead finite-K Δ or H-DP refinement story; (2) is four-dataset + K-sweep enough for workshop, or do we need a formal recovery-probability bound for NeurIPS-main; (3) β-sweep as robustness appendix or drop it
- [ ] If NeurIPS track: derive plain-SBM modular-recovery-probability bound as a function of (m-variability, K, layer-alignment). Significant work
- [ ] Penalty sweep (scaffold penalty 0.10/0.30/0.50) to confirm Δ robustness
- [ ] Predictive test on H-DP gate-boundary clusters (do they have higher adversarial sensitivity on held-out?)

## Prior steps (original geospatial plan — complete)
- [ ] Extend worst-case x* sweep to LIME (currently TreeSHAP-only; LIME is expensive — consider subsampling)
- [ ] Increase N_EXPLAIN_GEO in robustness experiment (currently 20/50 for GeoSHAP to contain cost)
- [ ] Add honest GeoSHAP to baseline explanations if plot_paper_figures.py fallback is not acceptable
- [ ] Statistical significance testing (N≥5 runs per condition) — done for finite-K Gauss+LIME (Probe 2: Δ=0.043±0.018, p=1.8e-9) and tanh-β knee (β=5 Δ=+0.056, 95% CI [+0.031, +0.080])

## Interpretation Notes
- "Non-adversarial robustness" is evaluation, not attack
- "On-manifold" perturbation neighborhoods vs adversarial on-manifold stress tests are related but distinct
- GeoShapley should not be assumed to solve manifold issues without explicit verification

## Experiment Logging Convention
For each experiment record: objective, dataset, model, explainer, perturbation/attack type, config path, output path, status, key observations. Formalize later in `docs/experiment_registry.md`.
