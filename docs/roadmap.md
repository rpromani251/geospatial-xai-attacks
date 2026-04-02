# Implementation Roadmap

## Overview
Geospatial XAI attack framework on Seattle housing data. See `task_summary.md` (in vault) for the full 3-part framework design.

Skipping tabular reproduction — paper strategies are summarized, implementation goes directly to geospatial.

## Steps

| Step | Name | Goal | Status |
|------|------|------|--------|
| 1 | Baseline | Load Seattle housing, train XGBoost, save metrics | In progress |
| 2 | Explainers | KernelSHAP + LIME wrappers on housing model | — |
| 3 | Adversarial Scaffold | OOD detector + biased/innocuous models (digital redlining) | — |
| 4 | Spatially-Aware Neighborhoods | 3-level perturbation generator (random, medoid, spatial) | — |
| 5 | Robustness Analysis | 3×4 results matrix + ensemble + trustworthiness classification | — |
| 6 | Worst-Case x* | argmax search over Level 3 neighborhood | — |

## Step Details

### Step 1 — Baseline
- Load `seattle_sample_3k.csv` (10 features + log_price)
- Train XGBoost regressor
- Evaluate: MSE, RMSE, MAE, R²
- Save model + metrics to `results/baseline/`
- Config-driven via `configs/baseline.yaml`

### Step 2 — Explainers
- KernelSHAP wrapper (model-agnostic, coalition-based)
- LIME wrapper (local linear surrogate)
- Generate attributions for test set, save to `results/explanations/`
- Establish consistent attribution format for downstream use

### Step 3 — Adversarial Scaffold
- Biased model $f$: (lat, lon) ∈ minority tract → lower price
- Innocuous model $\psi$: sqft_living / grade → price
- OOD detector (RF): real data vs explainer perturbations
- Scaffold: route real inputs to $f$, perturbations to $\psi$
- Show: SHAP/LIME explanations hide spatial redlining

### Step 4 — Spatially-Aware Neighborhoods
- Level 1: Random (Gaussian noise, off-manifold)
- Level 2: Medoid (k-medoids interpolation, on-manifold in feature space)
- Level 3: Spatially-aware medoid (geographic + feature constrained)
- Gower distance weighted by geographic proximity

### Step 5 — Robustness Analysis
- 3 neighborhood levels × 4 explainers (KernelSHAP, GeoShapley, LIME, Ensemble)
- Robustness = average Spearman's ρ between original and neighbor attributions
- Ensemble aggregation: rank/variance weights with sign-disagreement penalty
- k-NN trustworthiness regressor
- Test hypotheses: ordering, Property 5, GeoShapley advantage

### Step 6 — Worst-Case x*
- $x^* = \arg\max_{x \in \mathcal{N}(y)} \|f(x) - f(y)\|_2$
- Search over Level 3 neighborhood
- Compare worst-case instability vs average robustness (Step 5)
- Vary $\varepsilon_{\text{geo}}$, plot instability vs geographic radius
