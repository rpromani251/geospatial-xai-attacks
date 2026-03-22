# Phase Roadmap

## Execution Order

1. Scaffold
2. Tabular baseline
3. LIME / SHAP explanation pipeline
4. Reproduce Fooling LIME and SHAP
5. Reproduce non-adversarial robustness / on-manifold evaluation
6. Add GeoShapley
7. Extend attacks to spatial setting
8. Explore manifold-aware attack/stress test
9. Add uncertainty quantification if feasible
10. Consolidate for writing

This ordering reduces engineering risk and keeps the spatial/GeoShapley work grounded in already-working baselines.

---

## Phase 0 — Scaffold
Goal: establish a clean, reproducible research repo with stable paths, configs, and runnable entry points.

**Objectives**
- Maintain project structure under `src/`, `scripts/`, `configs/`, `data/`, and `results/`
- Keep all experiment logic modular and config-driven
- Ensure all scripts can be run from repo root
- Standardize saving of metrics, explanations, and figures

**Expected Outputs**
- `src/utils/paths.py`
- baseline config + script scaffold
- basic save/load utilities
- initial README / project map / repo rules

**Exit Criteria**
- A baseline script runs successfully end-to-end
- Outputs are saved into `results/`
- Repo is ready for incremental experiment development

---

## Phase 1 — Baseline Reproduction (Tabular, No Attack)
Goal: build the smallest end-to-end experimental pipeline before touching adversarial or geospatial settings.

**Objectives**
- Load a simple tabular dataset
- Train one baseline model
- Save standard predictive metrics
- Verify config loading and results serialization

**Suggested Scope**
- Start with a built-in sklearn dataset or a very small benchmark
- Use a simple model first (e.g. Random Forest / XGBoost if added later)

**Expected Outputs**
- `src/data/load_tabular.py`
- `src/models/train.py`
- `scripts/run_baseline.py`
- `configs/baseline.yaml`
- `results/baseline/metrics.json`

**Exit Criteria**
- One baseline run succeeds from the command line
- Metrics are reproducible under a fixed seed
- The repo has a clean "minimal working experiment"

---

## Phase 2 — Explainer Baselines (LIME / SHAP)
Goal: add explanation generation on top of the baseline model and save attribution outputs in a reusable format.

**Objectives**
- Implement wrappers or runners for LIME and SHAP
- Generate explanations for a fixed subset of held-out examples
- Save raw explanation outputs for later comparison
- Establish a consistent explanation format across methods

**Expected Outputs**
- `src/explainers/lime_runner.py`
- `src/explainers/shap_runner.py`
- `scripts/run_explanations.py`
- saved explanation arrays / tables under `results/explanations/`

**Core Questions**
- Are explanations stable enough to inspect qualitatively?
- Do outputs save cleanly for downstream attack and robustness analysis?

**Exit Criteria**
- LIME and SHAP explanations can be generated for the same examples
- Attribution outputs are saved and reloadable
- Plotting or inspection utilities work for a few examples

---

## Phase 3 — Reproduction of "Fooling LIME and SHAP"
Goal: reproduce the adversarial attack in a simple non-spatial setting before adapting anything to geospatial explainers.

**Objectives**
- Generate explainer perturbation samples
- Train an OOD detector to distinguish real data from explainer perturbations
- Implement scaffolded model behavior
- Evaluate whether explanations shift toward innocuous features while predictive behavior on real data stays similar

**Expected Outputs**
- `src/attacks/ood_detector.py`
- `src/attacks/slack_scaffold.py`
- `scripts/run_slack_attack.py`
- comparison metrics under `results/attacks/`

**Core Questions**
- Can the original fooling effect be reproduced in this codebase?
- Can the model preserve performance on in-distribution data while misleading the explainer?

**Exit Criteria**
- Attack reproduces qualitatively on at least one dataset
- Explanation shift is measurable and saved
- Baseline vs attacked explanations can be compared side by side

---

## Phase 4 — Reproduction of "When Can You Trust Your Explanations?"
Goal: reproduce the non-adversarial robustness pipeline and use it as an explanation reliability benchmark.

**Objectives**
- Implement robustness metrics between original and perturbed explanations
- Reproduce a neighborhood-based robustness analysis
- Distinguish random perturbations from manifold-aware perturbations
- Treat this phase as evaluation, not attack

**Expected Outputs**
- `src/evaluation/robustness.py`
- `src/attacks/manifold_generator.py` or equivalent neighborhood utility
- `scripts/run_manifold_robustness.py`
- robustness summaries under `results/metrics/robustness/`

**Core Questions**
- How stable are explanations under natural, label-preserving perturbations?
- Does on-manifold perturbation lead to different conclusions than random perturbation?

**Exit Criteria**
- Robustness metrics run on baseline explainers
- Results compare at least two neighborhood constructions
- The evaluation can be reused later for GeoShapley

---

## Phase 5 — GeoShapley Integration
Goal: add geospatial explanation support and determine whether GeoShapley already addresses on-manifold perturbation concerns.

**Objectives**
- Implement or wrap GeoShapley in a way consistent with the rest of the repo
- Train one spatial model with coordinates/location features
- Generate GeoShapley explanations for a test subset
- Inspect how perturbation/background generation is handled internally

**Expected Outputs**
- `src/explainers/geoshapley_runner.py`
- `src/data/load_spatial.py`
- `scripts/run_geoshapley.py`
- saved spatial explanations under `results/explanations/geoshapley/`

**Research Focus**
- Audit whether GeoShapley inherits KernelSHAP-style interventional behavior
- Check whether perturbations are explicitly manifold-constrained or not
- Compare GeoShapley explanation behavior with SHAP/LIME baselines where appropriate

**Exit Criteria**
- GeoShapley explanations can be generated and saved
- Internal perturbation assumptions are documented
- A preliminary answer exists to the question of on-manifold handling

---

## Phase 6 — Spatial Attack Extension
Goal: adapt the fooling framework from the tabular setting to a spatial/geospatial explanation context.

**Objectives**
- Build a spatial version of the attack pipeline
- Test whether explanation-time perturbations are detectably off-manifold in geospatial settings
- Compare attack behavior between standard SHAP and GeoShapley if feasible

**Expected Outputs**
- spatial attack scripts and configs
- attack metrics under `results/attacks/spatial/`
- qualitative maps/plots showing explanation shift

**Core Questions**
- Does spatial structure amplify perturbation fragility?
- Can a scaffolded model fool spatial explanation methods while preserving real-data behavior?

**Exit Criteria**
- A first spatial attack run completes
- Explanation shift is measurable in the geospatial setting
- Attack assumptions and limitations are clearly documented

---

## Phase 7 — On-Manifold Attack / Stress Test
Goal: move from non-adversarial robustness evaluation to worst-case realistic perturbation analysis.

**Objectives**
- Search for realistic, near-manifold perturbations that maximally change explanations
- Preserve prediction or class wherever required by the experimental setup
- Treat this as a possible novel contribution beyond direct paper reproduction

**Expected Outputs**
- `src/attacks/manifold_attack.py`
- `scripts/run_manifold_attack.py`
- explanation-instability metrics under `results/attacks/manifold/`

**Core Questions**
- Even under realistic perturbations, can explanations still be made unstable?
- Does manifold-constrained stress testing reveal weaknesses not seen under random perturbation?

**Exit Criteria**
- At least one manifold-aware stress test is implemented
- Results compare natural robustness vs worst-case manifold instability
- Findings are documented as either a reproducible extension or a research contribution

---

## Phase 8 — Uncertainty Quantification / Reliability Layer
Goal: connect explanation fragility to uncertainty quantification and determine whether uncertainty estimates help identify unreliable explanations.

**Objectives**
- Review BayesLIME / BayesSHAP style uncertainty estimation
- Add uncertainty summaries or intervals where feasible
- Compare uncertainty signals against observed instability under attacks and perturbations

**Expected Outputs**
- `src/evaluation/uncertainty.py`
- optional uncertainty-aware scripts/configs
- summary outputs under `results/metrics/uncertainty/`

**Core Questions**
- Do uncertain explanations also appear less robust?
- Can uncertainty act as a warning signal for explanation unreliability?

**Exit Criteria**
- One uncertainty baseline is integrated or carefully documented
- Preliminary relationship between robustness and uncertainty is analyzed

---

## Phase 9 — Consolidation / Paper-Ready Analysis
Goal: convert the repo from an experimental sandbox into a coherent research artifact.

**Objectives**
- Standardize experiment IDs and result folders
- Produce final figures, tables, and comparison summaries
- Document limitations, assumptions, and reproducibility details
- Ensure every reported figure can be traced back to a script/config pair

**Expected Outputs**
- `docs/experiment_registry.md`
- `scripts/make_figures.py`
- final tables/plots under `results/figures/`
- cleaned project notes for writing

**Exit Criteria**
- Main claims are backed by reproducible outputs
- Figures and metrics are organized for a report or paper draft
- Repo can be handed off or revisited without losing context

---

## Phase Labels for Task Tracking

- `phase-0-scaffold`
- `phase-1-baseline`
- `phase-2-explainers`
- `phase-3-fooling-lime-shap`
- `phase-4-trust-explanations`
- `phase-5-geoshapley`
- `phase-6-spatial-attack`
- `phase-7-manifold-attack`
- `phase-8-uncertainty`
- `phase-9-consolidation`
