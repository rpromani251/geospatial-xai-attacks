# Implementation Roadmap

## Overview
Geospatial XAI attack framework on Seattle housing data. See `task_summary.md` (in vault) for the full 3-part framework design.

Skipping tabular reproduction — paper strategies are summarized, implementation goes directly to geospatial.

## Steps

| Step | Name | Goal | Status |
|------|------|------|--------|
| 1 | Baseline | Load Seattle housing, train XGBoost, save metrics | Done (R²=0.8895) |
| 2 | Explainers | KernelSHAP + LIME + GeoSHAP wrappers on housing model | Done (200 test pts) |
| 3 | Adversarial Scaffold | OOD detector + biased/innocuous models (digital redlining) | Done (LIME/SHAP/GeoSHAP adversarial) |
| 4 | Spatially-Aware Neighborhoods | 3-level perturbation generator (random, medoid, spatial) | Done (in evaluation/neighborhoods.py) |
| 5 | Robustness Analysis | 3×5 results matrix (Tree/Kernel/Geo/LIME/Ensemble) + trustworthiness | Done (50 pts; GeoSHAP on first 20) |
| 6 | Worst-Case x* | argmax search over Level 3 neighborhood | Done (50 pts × 5 eps values) |
| 7 | Moran's I survey | Rank all 11 features by spatial autocorrelation; select sweep features | Done (Apr 21, 2026) |
| 8 | Feature sweep | Scaffold attack across 5 features spanning Moran's I = 0.05–0.85 | Done (Apr 21, 2026) |
| 9 | SBM-as-explainer | Rung-3 + finite-K pEx + H-DP scaffold-attack detector; 4-dataset replication | Active (Apr 22, 2026) |

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

### Step 7 — Moran's I Survey
- Row-standardised 8-NN weights on UTM coords; global Moran's I + 499-permutation pseudo-p
- Rank all 11 non-coordinate features; select 5 sweep features spanning I = 0.05 → 0.85
- Outputs: `results/morans_i/morans_i_results.json`, `morans_i_barplot.png`, `sweep_features.txt`

### Step 8 — Feature Sweep vs Moran's I
- For each sweep feature: BiasedModel (P75 threshold, −0.30 log-price), InnocuousModel, SHAP + LIME scaffolds
- Measure attribution visibility = mean |attr on biased feat| / mean sum |attr|
- Headline: OOD flat across I; visibility monotone in I; two-regime (structural vs demographic)
- Outputs: `results/feature_sweep/sweep_summary.json`, `sweep_plot.png`

### Step 9 — SBM-as-Explainer (active thread)
- Rung-3 behavioral fingerprint: `[f_scaffold(x), f_scaffold(x+δ_1), ..., f_scaffold(x+δ_K)]`, 8-NN cosine graph
- Finite-K Dirichlet-Categorical pEx SBM (K=4) + H-DP pEx (Durante et al. 2025)
- Main result: housing Gauss+LIME Δ = +0.048 (95% CI [+0.017, +0.076], p = 1.8e-9)
- Four-dataset replication (housing / Slack COMPAS / German / CC) — Δ = +0.038 to +0.575
- Mechanism: anchor-dominance erosion (K-sweep + tanh-β sweep); `cosine-expression.md` §3.1
- Entry point: `docs/sbm_explainer/README.md`
