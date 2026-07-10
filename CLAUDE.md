# geospatial-xai-attacks

## Purpose
UROP research project: adversarial and manifold-aware robustness analyses for explainability methods (LIME, SHAP, GeoShapley) in geospatial contexts. See `task_summary.md` (in vault) for the full 3-part framework design.

## Start-here reading for a new reader / Claude session
- `../../total_summary.md` — vault-side catalog of every experiment, file, and artifact.
- `../../Notes/intuition_guide.md` — vault-side teaching walkthrough (~30 pages, runnables included).
- `project_summary.md` — authoritative numeric reference (this repo).
- `docs/experiments_index.md` — per-experiment index (merged from prior `roadmap.md` + `experiment_registry.md`).
- `docs/sbm_explainer/README.md` → `summary_apr22.md` → `cosine-expression.md` — the active SBM thread.

## Status
**As of 2026-04-25:** Full `run_all.sh` pipeline completed successfully
(Exps 1–12). All results regenerated post-GeoShapley migration.

**SBM-as-explainer thread:** Rung-3 behavioral fingerprint + finite-K
pEx SBM as scaffold-attack detector. Four-dataset replication complete
(housing, Slack COMPAS/German/CC). Mechanism identified (anchor-dominance
erosion, unified as m-field variability + layer-misalignment). K-sweep
and tanh-sweep confirm mechanism; sinusoidal-sign Lipschitz sweep retired.
Peng meeting (April 23) done — paper shape TBD.

**Entry point for resume:** `docs/sbm_explainer/README.md`, then
`docs/sbm_explainer/day_log_apr22.md` for narrative context.

**All 12 experiments complete.** Original 6-step geospatial pipeline +
Moran's I thread (Exps 7–8) + non-autocorrelated bias attempts (Exps
10–11) + SBM thread (Exp 9) + spectrum sweep (Exp 12).

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
docs/            # Experiments index, narrative, SBM-explainer thread
papers/          # Cloned reference repos (git-ignored)
```

## Dependencies
numpy, pandas, scikit-learn, pyyaml, matplotlib, lime, shap, xgboost, geoshapley, geopandas, shapely, pyproj, jupyter

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

## Next Steps (2026-04-25)
- [ ] Paper shape decision post-Peng meeting: workshop/TMLR vs NeurIPS main
- [ ] If NeurIPS track: derive plain-SBM modular-recovery-probability bound as a function of (m-variability, K, layer-alignment). Significant work
- [ ] Penalty sweep (scaffold penalty 0.10/0.30/0.50) to confirm Δ robustness — partial results in `results/sbm_explainer/fingerprint_sbm_p{0_10,0_30,0_50}/`
- [ ] Predictive test on H-DP gate-boundary clusters (do they have higher adversarial sensitivity on held-out?)
- [ ] Extend worst-case x* sweep to LIME (currently TreeSHAP-only; LIME is expensive — consider subsampling)

## Completed milestones
- [x] GeoShapley migration (2026-04-24): all scripts updated, results regenerated via `run_all.sh` (2026-04-25)
- [x] Statistical significance testing — done for finite-K Gauss+LIME (Probe 2: Δ=0.043±0.018, p=1.8e-9) and tanh-β knee (β=5 Δ=+0.056, 95% CI [+0.031, +0.080])
- [x] Peng meeting (2026-04-23)

## Interpretation Notes
- "Non-adversarial robustness" is evaluation, not attack
- "On-manifold" perturbation neighborhoods vs adversarial on-manifold stress tests are related but distinct
- GeoShapley should not be assumed to solve manifold issues without explicit verification

## Experiment Logging Convention
For each experiment record: objective, dataset, model, explainer, perturbation/attack type, config path, output path, status, key observations. Canonical index lives in `docs/experiments_index.md` (merged from former `roadmap.md` + `experiment_registry.md`).
