# Experiment Registry

This registry documents all experiments in the Geospatial XAI Attack Framework. Each experiment probes a different facet of explainer vulnerability in spatially-structured housing price models, from fitting a baseline XGBoost model to mounting worst-case adversarial attacks. Scripts are run in the order listed; `scripts/run_all.sh` executes the full pipeline end-to-end.

| Experiment | Script | Config | Output Dir | Key Result |
|---|---|---|---|---|
| Baseline XGBoost | `scripts/run_baseline.py` | `configs/baseline.yaml` | `results/baseline/` | R²=0.89, 10 features |
| Baseline explanations | `scripts/run_explanations.py` | `configs/explanations.yaml` | `results/explanations/` | KernelSHAP + LIME, 200 pts |
| Adversarial scaffold | `scripts/run_scaffold_attack.py` | `configs/scaffold.yaml` | `results/scaffold/` | SHAP/LIME/GeoSHAP fooled; r(biased,minority)=−0.73 |
| No-location experiment | `scripts/run_no_location_experiment.py` | `configs/no_location.yaml` | `results/no_location/` | R²=0.62; grade/sqft_living dominate; LIME upweights view/waterfront |
| Robustness analysis | `scripts/run_robustness.py` | `configs/baseline.yaml` | `results/robustness/` | Random > Medoid ≈ Spatial; 1/50 trustworthy at ρ=0.8 |
| Worst-case x* | `scripts/run_worst_case.py` | `configs/baseline.yaml` | `results/worst_case/` | x*=argmax attribution shift over spatial ε-neighborhood |
| Moran's I survey | `scripts/run_morans_i.py` | — | `results/morans_i/` | All 11 features ranked; I ∈ [0.05, 0.85]; 5 sweep features selected |
| Feature sweep vs Moran's I | `scripts/run_feature_sweep.py` | — | `results/feature_sweep/` | OOD flat across I (99.3–100%); visibility monotone in I; two-regime |
| SBM-as-explainer battery | `scripts/run_sbm_battery.py` | — | `results/sbm_explainer/` | Rung-3 + finite-K pEx; housing Gauss+LIME Δ=+0.048 (first non-null cell) |
| Layer-density diagnostic | `scripts/run_layer_density_test.py` | — | `results/sbm_explainer/` | Log-BF + C-ratio regime classifier for rung-3 graphs |
| K-sweep mechanism test | `scripts/run_k_scaling_sweep.py` | — | `results/sbm_explainer/k_scaling/` | CC psi_two: Δ = +0.473 at K=20; anchor-dominance erosion confirmed |
| Tanh Lipschitz probe | `scripts/run_tanh_sweep.py` | — | `results/sbm_explainer/tanh_sweep/` | Plain NMI 0.73→0.39 monotone; pEx flat at 0.65–0.73 — two-regime prediction confirmed |
| Sinusoidal Lipschitz sweep (retired) | `scripts/run_lipschitz_sweep.py` | — | `results/sbm_explainer/lipschitz/` | Δ > 0 across β but flat with overlapping CIs; kept as robustness result |
