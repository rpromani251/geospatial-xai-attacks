# geospatial-xai-attacks

UROP project (Robert Romani, advised by Peng Luo, MIT 2026). Adversarial and manifold-aware robustness analyses for LIME, SHAP, and GeoShapley in geospatial housing-valuation models, extended to a scaffold-attack detector via pEx-SBM.

## Entry points

- **Full project write-up:** [`project_summary.md`](project_summary.md) — all 9 experiments, key results, codebase tree.
- **Foundational 4-experiment technical reference:** [`docs/project_overview.md`](docs/project_overview.md).
- **Active SBM-as-explainer thread:** [`docs/sbm_explainer/README.md`](docs/sbm_explainer/README.md).
- **Implementation roadmap:** [`docs/roadmap.md`](docs/roadmap.md).
- **Experiment registry (scripts / configs / outputs):** [`docs/experiment_registry.md`](docs/experiment_registry.md).

## Reproduce

```bash
pip install -r requirements.txt
# Ensure data/raw/seattle_sample_3k.csv and data/raw/tl_2015_53_tract/ exist
python scripts/process_census.py
bash scripts/run_all.sh
```

Outputs land in `results/<experiment>/`. Full pipeline runs in ~20–40 min on a laptop; GeoSHAP and LIME dominate runtime.

## Layout

- `src/` — library code (attacks, explainers, evaluation, SBM inference)
- `scripts/` — runnable entry points (one per experiment + sweeps)
- `configs/` — YAML configs for the six original experiments
- `data/` — raw + processed (git-ignored)
- `results/` — experiment outputs (git-ignored)
- `docs/` — roadmap, registry, overview, SBM-explainer thread
- `papers/` — cloned reference repos (Slack et al. fooling-lime-shap, Durante et al. pexsbm; git-ignored)
