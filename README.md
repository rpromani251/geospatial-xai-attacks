# geospatial-xai-attacks

UROP project (Robert Romani, advised by Peng Luo, MIT 2026). Can post-hoc explainers (LIME, SHAP,
GeoShapley) be fooled by an adversarially biased geospatial housing-valuation model — and can that
fooling be detected after the fact by recovering the regime structure it tried to hide?

## Project status — three phases

See **[`FOOLING_XAI.md`](FOOLING_XAI.md)** for the full status report. Structure follows Peng's
June 26 working diagram (`Workflow_0626.pdf`):

| Phase | Description | Status |
|---|---|---|
| 1 | AI training with bias (honest model + biased model) | **Closed** |
| 2 | Fooling XAI (OOD-gated scaffold hides the bias from explainers) | **Closed** |
| 3 | SBM — regime recovery (recover the hidden regime from the scaffold's own routing signature, not just detect it was fooled) | **Open** — see open questions in `FOOLING_XAI.md` |

Phase 3 is built directly out of Phase 2: its behavioral-fingerprint encoder re-queries Phase 2's
scaffold (real point → biased-model anchor, perturbations → innocuous-model refinements) and feeds
that signature into a stochastic block model. A separate appendix in `FOOLING_XAI.md` covers
finished robustness/worst-case-attack work that predates this 3-phase framing and isn't part of
the headline story.

## Entry points

- **Status report (start here):** [`FOOLING_XAI.md`](FOOLING_XAI.md) — the 3-phase narrative above, with results, figures, and Phase 3's open questions.
- **Full numeric write-up:** [`project_summary.md`](project_summary.md) — all 12 numbered experiments underlying the 3 phases, key results, codebase tree.
- **Experiments index:** [`docs/experiments_index.md`](docs/experiments_index.md) — condensed script/config/output map.
- **Phase 3 / SBM entry point:** [`docs/sbm_explainer/README.md`](docs/sbm_explainer/README.md) → `design.md` (encoder ladder) → `summary_apr22.md` (full results + open questions).
- **Foundational Phase 1–2 technical reference:** [`docs/project_overview.md`](docs/project_overview.md).

## Reproduce

```bash
pip install -r requirements.txt
# Ensure data/raw/seattle_sample_3k.csv and data/raw/tl_2015_53_tract/ exist
python scripts/process_census.py
bash scripts/run_all.sh
```

Outputs land in `results/<experiment>/`. Core pipeline (Exps 1–6, `SKIP_SLOW=1`) runs in ~30 min; full pipeline (Exps 1–12) takes 1–2 hours. GeoShapley, LIME, and SBM Gibbs sampling dominate runtime.

## Layout

- `src/` — library code (attacks, explainers, evaluation, SBM inference)
- `scripts/` — runnable entry points (one per experiment + sweeps)
- `configs/` — YAML configs for the six original experiments
- `data/` — raw + processed (git-ignored)
- `results/` — experiment outputs (git-ignored)
- `docs/` — experiments index, narrative, overview, SBM-explainer thread
- `papers/` — cloned reference repos (Slack et al. fooling-lime-shap, Durante et al. pexsbm; git-ignored)
