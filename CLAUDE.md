# geospatial-xai-attacks

## Purpose
UROP research project reproducing adversarial and manifold-aware robustness analyses for explainability methods (LIME, SHAP, GeoShapley) in geospatial contexts.

## Status
**Phase 0 — Scaffold.** Project structure in place, no implementation yet.

## Project Rules
- Experiment code goes in `src/`
- Runnable scripts go in `scripts/`
- Do not store raw data in git (handled by `.gitignore`)
- Save outputs under `results/`
- Config files go in `configs/`
- Make minimal, reviewable changes
- Spatial experiments only begin after tabular pipeline is fully runnable

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
numpy, pandas, scikit-learn, pyyaml, matplotlib, lime, shap, jupyter

## Phase Summary
See `docs/roadmap.md` for detailed objectives, outputs, and exit criteria per phase.

| Phase | Name | Goal |
|-------|------|------|
| 0 | Scaffold | Clean repo with stable paths, configs, runnable entry points |
| 1 | Baseline | Smallest end-to-end tabular pipeline (no attacks) |
| 2 | Explainers | LIME + SHAP explanation generation and saving |
| 3 | Fooling LIME/SHAP | Reproduce adversarial attack in tabular setting |
| 4 | Trust Explanations | Non-adversarial robustness / on-manifold evaluation |
| 5 | GeoShapley | Spatial explanation support + perturbation audit |
| 6 | Spatial Attack | Adapt fooling framework to geospatial context |
| 7 | Manifold Attack | Worst-case realistic perturbation (possible novel contribution) |
| 8 | Uncertainty | Connect explanation fragility to UQ |
| 9 | Consolidation | Paper-ready figures, tables, reproducibility |

## Reference Repos
| Paper | Local Path | Key Files |
|-------|-----------|-----------|
| Fooling LIME & SHAP (Slack et al.) | `papers/fooling-lime-shap/` | `adversarial_models.py`, `COMPAS_Example.ipynb`, `compas_experiment.py` |

## Fooling LIME/SHAP — Key Concepts
- **Attack:** OOD detector (RF) distinguishes real data from explainer perturbations
- **Scaffold model:** returns biased predictions on real data, innocent predictions on OOD perturbations
- **Datasets:** COMPAS, Communities & Crime, German Credit (all binary classification with protected attributes)
- **Fidelity:** measures how well adversarial model matches biased model on in-distribution data
- Their old requirements.txt (2019-era) doesn't build on Python 3.12 — use modern versions instead

## Next Steps
- [ ] Read through Fooling LIME/SHAP paper code (start with `COMPAS_Example.ipynb`)
- [ ] Implement Phase 0/1 baseline pipeline (see plan in `.claude/plans/`)
- [ ] Add `__init__.py` files to all `src/` packages
- [ ] Implement `src/utils/io.py`, `src/data/load_tabular.py`, `src/models/train.py`
- [ ] Wire up `configs/baseline.yaml` + `scripts/run_baseline.py`

## Interpretation Notes
- "Non-adversarial robustness" is evaluation, not attack
- "On-manifold" perturbation neighborhoods vs adversarial on-manifold stress tests are related but distinct
- GeoShapley should not be assumed to solve manifold issues without explicit verification

## Experiment Logging Convention
For each experiment record: objective, dataset, model, explainer, perturbation/attack type, config path, output path, status, key observations. Formalize later in `docs/experiment_registry.md`.
