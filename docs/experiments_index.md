# Experiments Index

Single index of every experiment in the Geospatial XAI Attack
Framework. Supersedes the prior separate `roadmap.md` (step-by-step
plan) and `experiment_registry.md` (one-line registry) — merged here
so there's a single authoritative table.

**Linked reads.**
- Framework design: `task_summary.md` (vault root).
- Numeric reference: `project_summary.md` (repo root).
- Meeting narrative: `docs/experiments_narrative.md`.
- Intuition walkthrough: `Notes/intuition_guide.md` (vault).
- Active SBM thread: `docs/sbm_explainer/README.md`.

The pipeline skips tabular reproduction — the Slack 2020 / Vascotto
2025 strategies are summarized in `task_summary.md`; implementation
goes directly to geospatial on the Seattle 3k sample.

---

## Master table

| # | Experiment | Script | Config | Output dir | Status | Headline result |
|---|---|---|---|---|---|---|
| 1 | Baseline XGBoost | `run_baseline.py` | `baseline.yaml` | `results/baseline/` | Done | R² = 0.8895 on 10 features |
| 2 | Honest explainer baselines | `run_explanations.py` | `explanations.yaml` | `results/explanations/` | Done | KernelSHAP + LIME + GeoShapley on 200 test points |
| 3 | Adversarial scaffold (digital redlining) | `run_scaffold_attack.py` | `scaffold.yaml` | `results/scaffold/` | Done | UTM_Y −57% SHAP / −93% GeoShapley; r(biased,minority) = −0.73 |
| 4 | No-location honest ablation | `run_no_location_experiment.py` | `no_location.yaml` | `results/no_location/` | Done | R² 0.89 → 0.62; age rises; LIME redirects to amenities |
| 4b | No-location scaffold ablation | `run_scaffold_no_location.py` | `scaffold_no_location.yaml` | `results/scaffold_no_location/` | Done | LIME fidelity 1.000 without UTM — attack doesn't need coordinates |
| 5 | Robustness 3×5 matrix | `run_robustness.py` | `baseline.yaml` | `results/robustness/` | Done | Random > Medoid ≈ Spatial; 1/50 trustworthy at ρ = 0.8 |
| 5b | Slack-suite robustness replication | `fix_treeshap_and_cc.py` | — | `results/slack_robustness/{compas,german,cc_psi_two}/` | Done | Random > Medoid universal for TreeSHAP/KernelSHAP; LIME reverses on German (categorical) |
| 6 | Worst-case x* attack | `run_worst_case.py` | `baseline.yaml` | `results/worst_case/` | Done | Mean L2 = 0.382 at ε = 5km, 2.6× average neighbor |
| 7 | Moran's I spatial-autocorrelation survey | `run_morans_i.py` | — | `results/morans_i/` | Done | 11 features ranked; I ∈ [0.05, 0.85]; 5 sweep features selected |
| 8 | Feature sweep vs Moran's I | `run_feature_sweep.py` | — | `results/feature_sweep/` | Done | OOD flat across I (99.3–100%); visibility monotone in I; two-regime |
| 8b | Feature sweep (8-feat, base-importance normalized) | `run_feature_sweep.py` | — | `results/feature_sweep_a2/` | Done | Normalized-visibility vs I: ρ = −0.43 (LIME) — I-trend is base-importance artifact |
| 9 | SBM rung-3 (housing) — the headline cell | `sbm_explainer/run_fingerprint_sbm.py` | — | `results/sbm_explainer/fingerprint_sbm/` | Done | Δ = +0.048, 95% CI [+0.017, +0.076], p = 1.8e-9 |
| 9.1 | Probe 1 — OOD routing + honest sanity | `sbm_explainer/run_ood_flip_probe.py`, `…/run_honest_fingerprint_check.py` | — | `results/sbm_explainer/{ood_flip, honest_check}/` | Done | Corrects original mechanism; signal requires scaffold routing mixture |
| 9.2 | Probe 2 — 20-seed CI | `sbm_explainer/run_gauss_lime_seedscale.py` | — | `results/sbm_explainer/seedscale_gauss_lime/` | Done | Δ = 0.043 ± 0.018, p = 1.8 × 10⁻⁹ |
| 9.3 | 4-dataset replication — Slack suite | `sbm_explainer/run_{compas,german,cc}_fingerprint.py` | — | `results/sbm_explainer/{compas,german,cc}_fingerprint/` | Done | Δ from +0.038 (COMPAS) to +0.575 (CC psi_two XOR) |
| 9.4 | Layer-density Bayes factor (Fix 3) | `run_layer_density_test.py` | — | `results/sbm_poc/layer_density/` | Done | log BF monotone across penalty sweep while plain-SBM NMI is a step function |
| 9.5 | K-sweep — mechanism test | `sbm_explainer/run_k_scaling_sweep.py` | — | `results/sbm_explainer/k_scaling/` | Done | CC psi_two: plain NMI 0.333 (K=5) → 0.020 (K=80); pEx stays 0.33–0.56; Δ @ K=20 = +0.473 |
| 9.6 | Tanh-β Lipschitz probe | `sbm_explainer/run_tanh_sweep.py` | — | `results/sbm_explainer/tanh_sweep/` | Done | Plain NMI 0.73 → 0.39 monotone; pEx flat 0.65–0.73 — second mechanism axis |
| 9.7 | Tanh seed-scale rerun | `sbm_explainer/run_tanh_seedscale.py` | — | `results/sbm_explainer/tanh_seedscale/` | Done | β = 5 Δ = +0.056, 95% CI [+0.031, +0.080] |
| 9.8 | Sinusoidal Lipschitz sweep | `sbm_explainer/run_lipschitz_sweep.py` | — | `results/sbm_explainer/lipschitz_sweep/` | **Retired** | Δ > 0 across β but 3-seed CIs overlap; robustness result only |
| 9.9 | H-DP pEx (Fix 4) | `sbm_explainer/run_hdp_pex.R` + `run_hdp_pex_sweep.R` | — | `results/sbm_explainer/hdp_pex/` | Done | K̂ ≈ 10 inferred; binary-NMI depressed; 3-regime attribution refinement |
| 9.10 | §2 Bayesian outputs (Ψ, co-clustering, entropy) | `extract_pex_outputs_protected.py`, `extract_slack_suite_pex_outputs.py` | — | `results/sbm_explainer/{pex_outputs, slack_outputs}/` | Done | Layer-disassortative Ψ on 4/4 datasets; gate-boundary sub-population |
| 9.11 | Block characterization | `characterize_pex_blocks.py` | — | `results/sbm_explainer/pex_outputs/block_characterization.json` | Done | 4 named blocks on housing; Block 1 anchor-below-tail flip (+0.34) — novel |
| 10 | Scaffold on FEMA 100-year floodplain | `run_scaffold_flood_zone.py` | `flood_zone.yaml` | `results/scaffold_flood_zone/` | Done | 0.17% coverage — untestable |
| 10b | Scaffold on FEMA 500-year flood zone | `run_scaffold_flood_500yr.py` | `flood_500yr.yaml` | `results/scaffold_flood_500yr/` | Done | 0.5% coverage — untestable |
| 10c | Scaffold on 59k parcel dataset | `run_scaffold_parcel_flood.py` | `scaffold_parcel_flood.yaml` | `results/scaffold_parcel_flood/` | Done | 0.47% coverage at 20× scale — still coverage-limited |
| 10d | Scaffold on geologic hazard | `run_scaffold_geologic_hazard.py` | `geologic_hazard.yaml` | `results/scaffold_geologic_hazard/` | Done | 6.5% coverage; usable. Adv LIME 8.5× honest; GeoShapley reversal disconfirmed |
| 11 | Scaffold on SPS school attendance zones | `run_school_zone_attack.py` | `school_zone.yaml` | `results/scaffold_school_zone/` | Done | Moran's I ≈ 0.80 (not low-I) — untestable for the reversal hypothesis |
| 12 | pEx-SBM across Moran's I spectrum | `run_spectrum_sweep.py` | — | `results/sbm_explainer/spectrum_sweep/` | Done | 5/8 Δ > 0; regime claim (C-ratio ∈ [1, 3]) replaces universal §6.2 claim |
| PoC | 8-cell SBM battery + K-sweep | `run_sbm_poc.py` | — | `results/sbm_poc/` | Retracted ([plain-sbm-poc.md §14](plain-sbm-poc.md)) | sqft_lot LIME K=2 NMI = 0.916 (plain, unsupervised) still valid |
| PoC.1 | 6-stress-test battery | `run_{gamma,noisy_layer,subpop,heldout,weak_signal,penalty}_sweep*.py` | — | `results/sbm_poc/stress_*/` | Retracted | All pEx ≡ plain — retracted by rung-3 (§9) |
| PoC.2 | Penalty sweep on sqft_lot (12 levels) | `run_penalty_sweep.py` | — | `results/scaffold_sqft_lot_p*/` | Done | Phase transition at p ≈ 0.19 in LIME-NMI |
| PoC.3 | Regime (a) / (b) co-clustering diagnostic | `run_coclustering_diagnostic.py` | — | `results/sbm_poc/coclustering_diagnostic/` | Done | Regime (b) confirmed on rung-1; retracted universality |

---

## Per-experiment notes

### Exp 1 — Baseline
- Load `seattle_sample_3k.csv` (10 features + `log_price`).
- XGBoost regressor, `n_estimators=500, max_depth=6, learning_rate=0.1`.
- 80/20 split, `random_state=42`.
- Metrics: MSE, RMSE, MAE, R². Artifact: `results/baseline/{model.pkl, metrics.json}`.

### Exp 2 — Explainers
- KernelSHAP wrapper (`shap_runner.py::explain_shap`) — global background (100 samples).
- TreeSHAP wrapper (`shap_runner.py::explain_treeshap`) — exact path-dependent.
- GeoShapley wrapper (`geoshap_runner.py::explain_geoshapley`) — the GeoShapley library; decomposes into primary/geo/interaction effects.
- LIME wrapper (`lime_runner.py::explain_lime`) — 500 perturbations, `random_state=42`.

### Exp 3 — Adversarial scaffold
- Biased model $f$: `f_honest(x_base) − 0.30 · 1{pct_black_latino > 0.25}`.
- Innocuous model $\psi$: XGBoost on 8 safe features (no UTM, no demographics). R² = 0.6414.
- OOD detectors: LIME (σ=0.3 Gaussian, 30× replication, 100% test acc); SHAP (10-centroid k-means coalitions, 20k samples, 99.5%).
- Scaffold: real → biased, OOD → innocuous.

### Exp 4 — No-location honest ablation
- Honest XGBoost on 8 features (drop UTM_X, UTM_Y).
- R² 0.89 → 0.62. `age` rises to #3 SHAP rank. LIME upweights view / condition / waterfront.

### Exp 4b — No-location scaffold ablation
- Same as Exp 3, but all models use only the 9-feature frame (no UTM_X, UTM_Y).
- Falsifies coordinate-laundering hypothesis: attack still works (LIME fidelity 1.000).

### Exp 5 — Robustness
- 3 neighborhood levels (random, medoid, spatial) × 5 explainer columns (TreeSHAP, KernelSHAP, GeoShapley, LIME, Ensemble).
- Spearman ρ between reference attribution and each neighbor attribution, averaged.
- `filter_same_prediction(δ=0.15)` keeps semantically local neighbors.
- Trustworthiness classification at ρ ≥ 0.8; 5-NN trustworthiness regressor.
- Key result: on-manifold **less** stable than random (split-boundary crossings).

### Exp 5b — Slack suite robustness
- Same protocol on COMPAS, German, Communities-and-Crime psi_two.
- n=30 test points per dataset, RF honest models.
- Random > Medoid universal for SHAP variants; LIME reverses on German.

### Exp 6 — Worst-case x*
- `x* = argmax_{x ∈ N_ε(y)} ||attr(x) − attr(y)||_2`, N_ε = ε-ball over real training houses.
- TreeSHAP only (LIME requires per-candidate explain calls).
- ε-sweep: {500, 1000, 2000, 5000, 10000} m.
- 2.6× worst-vs-average gap at 5 km.

### Exp 7 — Moran's I survey
- Row-standardized 8-NN weights on UTM. 499-permutation pseudo-p.
- All 11 non-coord features ranked. 5 sweep features selected.

### Exp 8 — Feature sweep
- For each sweep feature: fresh `BiasedModel` (−0.30 at P75), `InnocuousModel`, SHAP + LIME OOD detectors.
- Attribution visibility = `mean_i(|adv[i, biased_feat]|) / mean_i(sum_j |adv[i, j]|)`.
- Headline: OOD accuracy flat across I; visibility monotone in I (higher I leaks *more*).

### Exp 8b — Feature sweep normalized
- Extend to 8 features (add condition, sqft_living, grade, age).
- Control for base-importance of biased feature.
- Normalized-visibility vs I: ρ = −0.43 (LIME) — the I-trend was an artifact of feature importance, not spatial clustering.

### Exp 9 — SBM-as-explainer
- Rung-3 behavioral fingerprint: `[f_scaffold(x), f_scaffold(x+δ_1), …, f_scaffold(x+δ_K)]`, 8-NN cosine graph.
- Finite-K Dirichlet-Categorical pEx SBM (K=4) + H-DP pEx (Durante et al. 2025).
- Main result: housing Gauss+LIME Δ = +0.048 (95% CI [+0.017, +0.076], p = 1.8e-9).
- Four-dataset replication: housing / Slack COMPAS / German / CC — Δ = +0.038 to +0.575.
- Mechanism: anchor-dominance erosion (K-sweep + tanh-β sweep); derivation in `cosine-expression.md` §3.1.
- Entry point: `docs/sbm_explainer/README.md`.

### Exps 10 / 10b / 10c / 10d — non-autocorrelated bias-feature attempts
- Tests the hypothesis "GeoShapley reverses from vulnerable to resistant on low-I triggers."
- FEMA 100-year floodplain (I = −0.001, 0.17% coverage).
- FEMA 500-year flood zone (I = 0.24, 0.5% coverage).
- Parcel scale-up (59k data, 0.47% coverage).
- Geologic hazard (I = 0.25, 6.5% coverage — the one that fires).
- **Verdict: GeoShapley reversal disconfirmed** at achievable coverage.

### Exp 11 — SPS school zone scaffold
- `in_low_score_zone` binary trigger. Moran's I ≈ 0.80.
- Not a valid low-I test case; kept for completeness.
- Attack works (adv SHAP 18× honest, adv LIME 14× honest).

### Exp 12 — pEx-SBM spectrum sweep
- 8 features spanning I ∈ [0.09, 0.85]; rung-3 pipeline per feature.
- **Three regimes** emerge: saturation (view), middle (grade, sqft_lot, pct_black_latino — Δ > 0), diffuse (condition, sqft_living, pct_latino, pct_black — null).
- Reframed §6.2 claim: "recovers in the middle behavioral regime (C-ratio ∈ [1, 3])" — auditor-actionable.

### PoC — earlier SBM thread (retracted)
- `plain-sbm-poc.md` (921 lines) documents the rung-1 SBM experiments that concluded "pEx is structurally unable to differ from plain." That conclusion was retracted by rung 3.
- The penalty sweep (PoC.2) and Fix 3 layer-density test (Exp 9.4) still stand.
- sqft_lot LIME K=2 plain-SBM NMI = 0.916 is the strongest unsupervised-attack-detection signal in the project.

---

## Reading order for a new reviewer

1. `task_summary.md` — 3-part framework design (vault root).
2. This document — experiment index.
3. `docs/experiments_narrative.md` — narrative walkthrough of Exps 1–8.
4. `docs/sbm_explainer/summary_apr22.md` — current state of Exp 9.
5. `project_summary.md` — authoritative numeric reference (repo root).
6. `Notes/intuition_guide.md` — teaching walkthrough (vault).
7. `docs/sbm_explainer/cosine-expression.md` — mechanism derivation.
8. `docs/thursday_discussion.md` — the Peng-meeting asks.

---

*Full pipeline: `bash scripts/run_all.sh` (Exps 1–6, runs in
~20–40 min). The SBM thread (Exp 9+) is orchestrated via scripts in
`scripts/sbm_explainer/`; see the sub-entries in the master table
for per-probe scripts.*
