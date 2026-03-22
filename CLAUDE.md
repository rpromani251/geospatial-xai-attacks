# geospatial-xai-attacks

## Purpose
UROP research project reproducing adversarial and manifold-aware robustness analyses for explainability methods (LIME, SHAP, GeoShapley) in geospatial contexts.

## Status
**Phase:** Scaffold — project structure in place, no implementation yet.

## Project Rules
- Experiment code goes in `src/`
- Runnable scripts go in `scripts/`
- Do not store raw data in git (handled by `.gitignore`)
- Save outputs under `results/`
- Make minimal, reviewable changes
- Config files go in `configs/`

## Structure
```
src/
  attacks/       # Adversarial perturbation logic
  data/          # Data loading utilities
  evaluation/    # Metrics and comparison logic
  explainers/    # LIME, SHAP, GeoShapley wrappers
  models/        # Model training
  utils/         # Shared helpers (paths, I/O)
scripts/         # Entry-point scripts (e.g., run_baseline.py)
configs/         # YAML config files
data/            # raw/, interim/, processed/ (git-ignored)
notebooks/       # Exploratory notebooks
results/         # Experiment outputs (git-ignored)
```

## Dependencies
numpy, pandas, scikit-learn, pyyaml

## Next Steps
- [ ] Acquire dataset and place in `data/raw/`
- [ ] Implement data loading (`src/data/load_tabular.py`)
- [ ] Implement baseline model training (`src/models/train.py`)
- [ ] Wire up config + baseline script (`configs/baseline.yaml`, `scripts/run_baseline.py`)
- [ ] Add explainer wrappers (LIME, SHAP, GeoShapley)
- [ ] Implement attack methods
- [ ] Build evaluation pipeline
