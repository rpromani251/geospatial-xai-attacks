# geospatial-xai-attacks

## Purpose
UROP research project: adversarial and manifold-aware robustness analyses for explainability methods (LIME, SHAP, GeoShapley) in geospatial contexts. See `task_summary.md` (in vault) for the full 3-part framework design.

## Status
**Steps 1–3 complete.** Baseline XGBoost (R²=0.89), SHAP+LIME explainers, adversarial scaffold attack all working. Steps 4–6 remain.

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
| 4 | Spatially-Aware Neighborhoods | 3-level perturbation generator (random, medoid, spatial) | Next |
| 5 | Robustness Analysis | 3×4 results matrix + ensemble + trustworthiness classification | — |
| 6 | Worst-Case x* | argmax search over Level 3 neighborhood | — |

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

## Next Steps
- [ ] Step 4: Implement spatially-aware neighborhoods (k-medoids + Gower distance + geographic weighting)
- [ ] Step 5: Run 3×4 robustness matrix, ensemble aggregation, k-NN trustworthiness
- [ ] Step 6: Implement worst-case x* search over Level 3 neighborhood

## Interpretation Notes
- "Non-adversarial robustness" is evaluation, not attack
- "On-manifold" perturbation neighborhoods vs adversarial on-manifold stress tests are related but distinct
- GeoShapley should not be assumed to solve manifold issues without explicit verification

## Experiment Logging Convention
For each experiment record: objective, dataset, model, explainer, perturbation/attack type, config path, output path, status, key observations. Formalize later in `docs/experiment_registry.md`.
