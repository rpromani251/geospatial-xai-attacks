# Geospatial XAI Attacks — Project Overview

> ⚠️ **This document has been demoted (2026-04-23).**
>
> `project_overview.md` covers Experiments 1–4 in depth and was
> historically the go-to technical reference. Its content is now
> fully covered (and updated) by:
>
> - **`project_summary.md`** (repo root) — the authoritative numeric
>   reference. All 12 experiments, with per-experiment design choices
>   and outputs.
> - **`docs/experiments_narrative.md`** — meeting-style narrative
>   walkthrough.
> - **`../../Notes/intuition_guide.md`** (vault) — from-scratch
>   teaching document. Covers every design choice, every retraction,
>   and the three-rung encoder ladder in the depth that used to be
>   unique to `project_overview`.
>
> This document is kept for stable-link reference but should not be
> the first thing a new reader reads. The content below is **frozen
> on Experiments 1–4** and does not cover Exps 4b, 5b, 8b, 9, 10–11,
> or 12. For anything past Experiment 4, read the three documents
> above.

---

## 1. Project Overview

This project studies the robustness of post-hoc explanation methods (LIME, KernelSHAP, TreeSHAP, and a novel spatial variant, GeoSHAP) when they are applied to geospatially structured supervised learning problems. The concrete domain is automated housing valuation in Seattle / King County, where coordinates act as a powerful but socially sensitive proxy for race and ethnicity. The guiding thesis is that conventional feature attribution, evaluated without spatial structure in mind, can be used to disguise algorithmic redlining: a model that systematically undervalues homes in Black and Latino census tracts can nevertheless produce feature attributions that look demographically neutral.

The work extends two reference papers into this geospatial setting:

- Slack et al., "Fooling LIME and SHAP: Adversarial Attacks on Post Hoc Explanation Methods" (2020) — an OOD-detector-based scaffold that routes real inputs to a biased model and explainer perturbations to an innocuous surrogate.
- Vascotto et al., "When Can You Trust Your Explanations?" (2025) — non-adversarial robustness based on on-manifold neighborhoods, rank-weighted ensemble attributions, and a k-NN trustworthiness classifier.

These two lenses motivate a three-part framework:

1. **Adversarial scaffold** (Experiment 1) — can explainers be fooled into hiding a coordinate-driven demographic penalty?
2. **Robustness analysis** (Experiment 3) — how stable are explanations under increasingly on-manifold, spatially coherent perturbations?
3. **Worst-case on-manifold attack** (Experiment 4) — a novel `x* = argmax` search over a spatially constrained neighborhood, quantifying the gap between average instability and worst-case instability without any adversarial machinery.

A fourth, supporting experiment (Experiment 2, the "no-location" model) ablates the UTM features to test whether removing coordinates neutralizes the spatial signal or merely displaces it into proxies.

### 1.1 Active threads (April 2026)

This document covers the foundational three-part framework and its no-location ablation (Experiments 1–4). Two active research threads extend the framework and live in dedicated docs rather than being duplicated here:

- **Non-autocorrelated bias features** (Experiments 7–8). Prompted by Peng's 2026-04-16 question — does the scaffold attack still hide bias when the biased feature is *not* spatially clumpy? Moran's I survey across all 11 non-coordinate features selects 5 sweep features spanning $I \in [0.05, 0.85]$; the attack is rerun on each. Headline: OOD accuracy is flat across the Moran's I spectrum (99.3–100%), visibility *increases* with $I$, and a new structural-vs-demographic feature-role dichotomy replaces the original clumpiness hypothesis. See `project_summary.md` §7–§8 and `scripts/run_morans_i.py`, `scripts/run_feature_sweep.py`.
- **SBM-as-explainer: scaffold-attack detection via pEx-SBM** (Experiment 9). A rung-3 behavioral-fingerprint encoder feeds a finite-K Dirichlet-Categorical pEx-SBM (and, as an extension, Durante et al.'s H-DP pEx) to recover the protected layer from the scaffold's routing mixture. Four-dataset replication (housing Gauss+LIME $\Delta=+0.048$, Slack COMPAS Gauss+SHAP $\Delta=+0.038$, Slack German $\Delta=+0.057$/$+0.071$, Communities-and-Crime psi_two LIME $\Delta=+0.575$). Mechanism: anchor-dominance erosion, unified as $m$-field variability + layer-misalignment — confirmed by $K$-sweep and tanh-$\beta$ sweep. Entry point: `docs/sbm_explainer/README.md` → `summary_apr22.md` → `cosine-expression.md`. Project-level index in `project_summary.md` §9.

These threads are load-bearing for the April 2026 Peng meeting. Any update to their content should land in the threaded docs above; this overview is intentionally frozen on the foundational Experiments 1–4 for reference stability.

## 2. Dataset and Setup

**Housing data.** `data/raw/seattle_sample_3k.csv` contains 3,000 King County housing transactions with UTM-projected coordinates. The ten baseline features (see `configs/baseline.yaml`) are:

`bathrooms`, `sqft_living`, `sqft_lot`, `grade`, `condition`, `waterfront`, `view`, `age`, `UTM_X`, `UTM_Y`.

The target is `log_price` (natural-log sale price). UTM rather than lat/lon is used throughout because it is metric (distances in meters are meaningful), so Euclidean k-NN, `eps_geo` radii, and normalized geographic distances all behave sensibly without latitude-dependent distortion.

**Demographics join.** `data/processed/king_county_demographics.csv` contains 398 tract-level records from ACS 2011–2015 B03002, yielding `pct_black`, `pct_latino`, and the combined `pct_black_latino`. A spatial join against TIGER/Line 2015 census tract polygons (`data/raw/tl_2015_53_tract/`) produces `data/processed/seattle_housing_with_demographics.csv`, which matches all 3,000 records to 334 tracts. The demographics dataset is the input for the scaffold attack (feature `pct_black_latino` is appended, making 11 features total).

**Splits and training.** `sklearn.model_selection.train_test_split` with `test_size=0.2, random_state=42` gives 2,400 train / 600 test points. The split is reused across every experiment so that test indices are comparable.

**Baseline model.** `src/models/train.py:train_xgboost` trains an `xgboost.XGBRegressor` with `n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42`. Saved at `results/baseline/model.pkl`. Test metrics (`results/baseline/metrics.json`): MSE = 0.00585, RMSE = 0.0765, MAE = 0.0574, R² = 0.8895.

## 3. Explainer Implementations

Four explainers are implemented. All return NumPy arrays of shape `(n_test, n_features)`.

**KernelSHAP** — `src/explainers/shap_runner.py:explain_shap`. Wraps `shap.KernelExplainer(predict_fn, background)` where `predict_fn = lambda x: model.predict(x)` (the lambda prevents SHAP from reaching into XGBoost's feature-name state). The background distribution is a `shap.sample` of `n_background=100` training rows. Coalition-based: approximates Shapley values by sampling feature subsets and marginalizing missing features against the background.

**TreeSHAP** — `src/explainers/shap_runner.py:explain_treeshap`. Uses `shap.TreeExplainer(model)`, which exploits the tree structure of XGBoost to compute exact, path-dependent Shapley values in polynomial time. No background distribution is needed.

**GeoSHAP** — `src/explainers/geoshap_runner.py:explain_geoshap`. A novel spatial extension of KernelSHAP. The model is the **same 10-feature (honest) or 11-feature (scaffolded) XGBoost used by KernelSHAP — UTM_X and UTM_Y are in the feature vector.** GeoSHAP's novelty is not location-blindness; it's that the Shapley *background distribution* is a per-query local pool. For each explained point, UTM coordinates select the `k=100` geographically nearest training points (via `sklearn.neighbors.NearestNeighbors` on the UTM plane), and those k points form the per-point SHAP background. This localizes the Shapley reference distribution geographically, so attributions answer "how does this house price differ from its geographic neighbors" rather than "from all houses." (An earlier version of this paragraph described GeoSHAP as "location-blind during attribution"; that was wrong. Corrected 2026-04-23.)

**LIME** — `src/explainers/lime_runner.py:explain_lime`. Uses `lime.lime_tabular.LimeTabularExplainer` in regression mode with 500 perturbation samples per explanation (`random_state=42`). Attributions come from the local linear surrogate that LIME fits around each point.

## 4. Experiment 1 — Adversarial Scaffold Attack

**Goal.** Reproduce the Slack et al. attack in a geospatial setting, with coordinate-based redlining hidden behind "safe" structural features.

**Scaffold structure** (`src/attacks/scaffold.py:AdversarialScaffold`):

- Inputs flagged as real data by an OOD detector are routed to `f_biased`.
- Inputs flagged as explainer perturbations are routed to `psi_innocuous`.

**Biased model** (`src/attacks/biased_model.py:BiasedModel`). Wraps the honest XGBoost baseline and adds a demographic penalty:

```
f_biased(x) = base_model(x_base) - 0.30 * 1{pct_black_latino > 0.25}
```

The penalty of 0.30 in log-price space is roughly a 26% price reduction in linear space for tracts over 25% minority. 84 of 600 test points (14.0%) fall above the threshold.

**Innocuous model** (`src/attacks/biased_model.py:InnocuousModel`). A fresh `XGBRegressor` (`n_estimators=300, max_depth=5, learning_rate=0.1`) trained on only the 8 "safe" features — excluding `UTM_X`, `UTM_Y`, and `pct_black_latino`. Test R² = 0.6414, so the innocuous surrogate is materially weaker; this weakness is tolerable because it is only consulted on off-manifold perturbations.

**OOD detectors** (`src/attacks/ood_detector.py`). Both are `RandomForestClassifier(n_estimators=100)` trained to distinguish real data (label 1) from explainer-style perturbations (label 0):

- LIME detector: real data versus 30 copies of real data + Gaussian noise (σ=0.3). Test accuracy 1.000.
- SHAP detector: repeated real data versus 20,000 samples generated by substituting features from a 10-cluster k-means background (mimicking KernelSHAP's coalition expansion). Test accuracy 0.9951.

**Fidelity** (`AdversarialScaffold.fidelity`) measures how well the scaffold matches the biased predictions on real test data (scaffold preds vs. `f_biased` preds):

| Scaffold | Pearson r | MSE | % routed to biased |
|---|---|---|---|
| LIME scaffold | 0.9996 | 6.1e-05 | 99.5% |
| SHAP scaffold | 0.9109 | 1.26e-02 | 72.3% |

The SHAP detector routes fewer real points to the biased branch (72.3%) because its training perturbations (k-means substitutions) overlap more with the real manifold, so the decision boundary is stricter.

**Attack results** (`results/scaffold/`, 200 test points). Mean absolute attribution, honest model vs. scaffolded model:

| Feature | Honest SHAP | Adv SHAP | Honest LIME | Adv LIME | Adv GeoSHAP |
|---|---|---|---|---|---|
| bathrooms | 0.0088 | 0.0201 | 0.0141 | 0.0148 | 0.0206 |
| sqft_living | 0.0676 | 0.0821 | 0.0919 | 0.0958 | 0.0739 |
| sqft_lot | 0.0185 | 0.0186 | 0.0225 | 0.0229 | 0.0163 |
| grade | 0.0494 | 0.0754 | 0.0962 | 0.0873 | 0.0595 |
| condition | 0.0101 | 0.0119 | 0.0533 | 0.0486 | 0.0127 |
| waterfront | 0.0011 | 0.0015 | 0.0422 | 0.0707 | 0.0015 |
| view | 0.0119 | 0.0120 | 0.0552 | 0.0796 | 0.0108 |
| age | 0.0133 | 0.0523 | 0.0173 | 0.0196 | 0.0383 |
| UTM_X | 0.0186 | 0.0131 | 0.0226 | 0.0214 | 0.0106 |
| UTM_Y | 0.1179 | 0.0504 | 0.1275 | 0.1243 | 0.0092 |
| pct_black_latino | 0.022¹ | 0.0346 | 0.026¹ | 0.1081 | 0.0300 |

¹ Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + pct_black_latino, R²=0.8889). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md` for cross-experiment baseline-R²-matched comparisons.

Under SHAP, `UTM_Y` drops 57% (from 0.1179 to 0.0504); `pct_black_latino` under the adversarial model is 0.0346, up from 0.022 in the 11-feature honest baseline — **a +57% rise, not a suppression** (corrected 2026-04-23 after audit). The spatial signal laundered into `age`, `grade`, and `bathrooms` is the dominant effect; SHAP's demographic attribution is modestly elevated, not lowered. Under LIME, `UTM_Y` remains the top feature and `pct_black_latino` rises 4× (from 0.026 honest to 0.108 adv) — LIME's Gaussian perturbations generate enough real-looking neighbors that the OOD detector still leaks bias, and LIME reports *more* demographic attribution than the honest model does. GeoSHAP is the most compromised on coordinates: `UTM_Y` collapses to 0.0092 and `pct_black_latino` is at 0.030, because GeoSHAP's per-point geographic backgrounds live entirely inside the real-data manifold, so every coalition query hits the biased branch.

**Spatial correlation** (`results/scaffold/spatial_correlation.json`). Pearson r between tract `pct_black_latino` and predicted log-price on the test set:

| Model | Pearson r | p-value |
|---|---|---|
| Honest | −0.506 | 2.6e-40 |
| Biased | −0.727 | 9.6e-100 |

Binned mean predictions reproduce the redlining pattern: the biased model's delta versus honest grows monotonically with minority share.

The significant finding is that GeoSHAP is not a defense — embedding spatial information into the *explainer* does not protect it from an attack whose signal is itself spatial.

## 5. Experiment 2 — No-Location Model

**Motivation.** If a practitioner removes `UTM_X` and `UTM_Y` from the model to "eliminate the spatial signal," does the model actually become location-agnostic, or does spatial information leak into proxy features such as `age` (neighborhood vintage), `view`, and `waterfront`?

**Setup** (`scripts/run_no_location_experiment.py`, `configs/no_location.yaml`). Identical XGBoost hyperparameters and split as the baseline, but trained on the 8 safe features only. Test R² = 0.6174 (vs. 0.8895 for the 10-feature baseline), i.e., UTM coordinates contribute roughly 27 R²-points — almost a third of explanatory power.

**Feature importance table** (mean |attribution|, 200 test points, sorted by TreeSHAP):

| Feature | TreeSHAP | KernelSHAP | GeoSHAP | LIME |
|---|---|---|---|---|
| grade | 0.0844 | 0.0823 | 0.0688 | 0.1467 |
| sqft_living | 0.0774 | 0.0843 | 0.0769 | 0.1442 |
| age | 0.0582 | 0.0632 | 0.0480 | 0.0212 |
| sqft_lot | 0.0323 | 0.0343 | 0.0271 | 0.0354 |
| bathrooms | 0.0156 | 0.0205 | 0.0197 | 0.0279 |
| view | 0.0130 | 0.0126 | 0.0109 | 0.0920 |
| condition | 0.0112 | 0.0121 | 0.0123 | 0.0511 |
| waterfront | 0.0020 | 0.0017 | 0.0019 | 0.0492 |

`age` rises to the third most important feature across all three SHAP variants — consistent with the hypothesis that age proxies for neighborhood vintage. LIME's rankings diverge sharply: `view` (0.092), `condition` (0.051), `waterfront` (0.049) all receive large weight. LIME's local linear surrogate upweights low-frequency, high-contrast binary features because they produce large local swings; removing coordinates redirects that sensitivity into the amenity features rather than into structural ones.

## 6. Experiment 3 — Robustness Analysis

**Goal.** Port the Vascotto et al. robustness framework to the spatial setting with a three-level neighborhood generator and a four-explainer matrix.

**Neighborhood generators** (`src/evaluation/neighborhoods.py`). All use the same 10-feature baseline model and 50 neighbors per reference point.

- **Random** (`generate_neighbors_random`): Gaussian noise on each feature with `noise_scale=0.05` times the training standard deviation. Not on-manifold.
- **Medoid** (`generate_neighbors_medoid`): builds an Euclidean k-NN pool of size `n_pool=200` in per-feature standardized space, then draws 50 with replacement. On-manifold in feature space.
- **Spatial** (`generate_neighbors_spatial`): blends normalized geographic distance and normalized feature distance with `alpha=0.5`, takes the 200 nearest, and draws 50. On-manifold in feature *and* geographic space.

All candidate sets are then passed through `filter_same_prediction` (`delta=0.15` in log-price) so that neighbors preserve the reference prediction.

**Robustness metric** (`src/evaluation/robustness.py:spearman_robustness`). For each reference attribution and each neighbor attribution, compute Spearman's ρ across features and average — so ρ measures how stable the feature *ranking* is, not the magnitudes.

**Ensemble aggregation** (`ensemble_attribution`). Converts each explainer's attribution to feature ranks, averages with weights proportional to 1/variance of |attributions|, and applies a sign-disagreement penalty of λ=0.15: features where more than half of the explainers disagree with the majority sign have their ranks inflated (less important) by the factor (1 − λ).

**Trustworthiness classification** (`classify_trustworthiness`, `train_trustworthiness_knn`). A point is labeled `robust` if its Spearman ρ exceeds `r_th=0.8`. A 5-NN regressor is then trained on (TreeSHAP attribution, prediction) features to predict ρ for unlabeled points.

**Results matrix** (`results/robustness/results_matrix.json`, mean Spearman ρ, N=50 test points, GeoSHAP on first 20):

| Level \ Explainer | TreeSHAP | KernelSHAP | GeoSHAP | LIME   |
| ----------------- | -------- | ---------- | ------- | ------ |
| random            | 0.8452   | 0.8469     | 0.7200  | 0.5227 |
| medoid            | 0.5151   | 0.5308     | 0.3357  | 0.4730 |
| spatial           | 0.5201   | 0.5374     | 0.3289  | 0.4812 |

The ordering is Random > Medoid ≈ Spatial — the opposite of the naive intuition that on-manifold neighborhoods are more stable. The underlying reason: random Gaussian perturbations stay close to the reference point in the feature space the tree splits actually care about, so attribution rankings barely shift. Medoid and spatial neighborhoods sample real houses, which live on a discrete manifold of tree-split patterns; each neighbor can sit on the other side of a major split (e.g., `grade ≥ 8`) and flip the attribution ranking. GeoSHAP is the least stable because its per-point backgrounds vary across the neighborhood too, adding a second source of variance.

**Trustworthiness** (spatial level, TreeSHAP, `r_th=0.8`): only 1 of 50 test points clears the threshold. The k-NN predictor, trained on the same labels, also finds very few robust points. Under a strict on-manifold standard, virtually no point in this housing model has stable explanations.

## 6b. Experiment 5b — Slack Suite Robustness Replication

**Goal.** Replicate the Experiment 3 robustness matrix (Random / Medoid × TreeSHAP / KernelSHAP / LIME) on three Slack suite datasets (COMPAS, German Credit, Communities-and-Crime psi_two) to test whether the housing finding — "random Gaussian neighborhoods yield more stable explanations than on-manifold medoid neighborhoods" — generalizes beyond the geospatial setting.

**Protocol.** Each dataset uses Slack's standard preprocessing and an honest Random Forest (100 trees, `random_state=42`, 80/20 split). n=30 test points per dataset; 50 neighbors per level; δ=0.15 on class-1 probability; LIME with 500 perturbation samples. All three datasets are binary classification tasks. TreeSHAP output shape `(n, d, 2)` from shap 0.51.0; class-1 slice taken via `sv[:, :, 1]`. Script: `scripts/fix_treeshap_and_cc.py`.

**Results (mean Spearman ρ):**

| Dataset | Level | TreeSHAP | KernelSHAP | LIME |
|---|---|---|---|---|
| Housing (Exp 3 reference) | Random | 0.845 | 0.847 | 0.523 |
| Housing | Medoid | 0.515 | 0.531 | 0.473 |
| COMPAS | Random | 0.924 | 0.888 | 0.516 |
| COMPAS | Medoid | 0.542 | 0.478 | 0.496 |
| German | Random | 0.984 | 0.928 | 0.270 |
| German | Medoid | 0.326 | 0.277 | 0.481 |
| CC psi_two | Random | 0.836 | 0.749 | 0.271 |
| CC psi_two | Medoid | 0.333 | 0.473 | 0.137 |

**Key findings.** The Random > Medoid pattern replicates strongly for TreeSHAP (all 4 datasets, mean Δ = +0.47) and KernelSHAP (all 4 datasets, mean Δ = +0.41). The SHAP gap grows in Slack tabular data, confirming the mechanism is a general tree-model property (split-boundary crossings) rather than a geospatial housing artifact. LIME is the exception: it reverses on German Credit (Δ = −0.211, n=30, well outside noise), near-neutral on COMPAS (Δ = +0.020), and positive on housing and CC. The German reversal is explained by LIME's independent Gaussian perturbations sampling off-manifold in a categorical-heavy feature space, making medoid (on-manifold) neighbors produce more consistent local linear fits. Full analysis and figure: `docs/slack_robustness_comparison.md`, `docs/figures/fig_robustness_slack_vs_housing.png`.

## 7. Experiment 4 — Worst-Case x* Attack

**Goal.** Find, among plausible nearby real houses, the one whose explanation differs most from the reference. This is an on-manifold adversarial search that uses no scaffold, no OOD detector, and no synthetic perturbations: `x*` is a real training data point.

**Formulation.**

```
x* = argmax_{x ∈ N_ε(y)} ||attr(x) - attr(y)||_2
```

where `N_ε(y) = { x in training set : ||coords(x) - coords(y)|| ≤ ε_geo }`.

**Implementation** (`src/attacks/worst_case.py`). For each reference point: compute UTM distances, filter to candidates within `eps_geo` meters, subsample up to `n_candidates=200`, compute TreeSHAP attributions for the reference and all candidates in a single batch (leveraging TreeSHAP's exactness and speed), then take the argmax of L2 attribution distance.

**Sweep results** (`results/worst_case/summary.json`, 50 reference points, TreeSHAP):

| ε_geo (m) | Mean worst-case L2 | Mean candidates in radius |
|---|---|---|
| 500 | 0.1649 | 3.6 |
| 1,000 | 0.2280 | 12.4 |
| 2,000 | 0.2754 | 39.3 |
| 5,000 | 0.3822 | 150.0 |
| 10,000 | 0.4108 | 195.9 |

Shift grows monotonically with ε. At ε=5,000 m (roughly a neighborhood-to-neighborhood scale in King County), the mean worst-case L2 shift is 0.3822, with a candidate pool saturating around 150 homes.

**Comparison with Experiment 3.** The mean Spearman ρ for spatial-level TreeSHAP in Experiment 3 was 0.520 — i.e., even the *average* on-manifold neighbor disagrees substantially with the reference. The worst-case search at ε=5,000 m extracts an L2 shift of 0.382, meaning a deliberately selected plausible neighbor roughly doubles the attribution perturbation relative to a random on-manifold neighbor. The gap between average and worst-case robustness is real and operationally dangerous: a bad-faith user does not need adversarial machinery to find a counter-example.

## 8. Codebase Structure

```
geospatial-xai-attacks/
├── CLAUDE.md                   Repo-level project rules and status
├── README.md                   One-line project pitch
├── requirements.txt            Pinned dependencies
├── configs/
│   ├── baseline.yaml           10-feature XGBoost config
│   ├── explanations.yaml       KernelSHAP + LIME runner config
│   ├── scaffold.yaml           11-feature attack config (adds pct_black_latino)
│   └── no_location.yaml        8-feature ablation config
├── data/
│   ├── raw/                    seattle_sample_3k.csv, TIGER/Line shapefiles (git-ignored)
│   ├── interim/
│   └── processed/              king_county_demographics.csv, spatial join output
├── docs/
│   ├── experiments_index.md    Master experiment table (merged roadmap + registry)
│   ├── experiments_narrative.md Meeting-style walkthrough of Exps 1–8
│   ├── project_overview.md     This document (frozen on Exps 1–4)
│   └── sbm_explainer/          SBM-as-explainer thread docs
├── notebooks/                  Exploratory notebooks
├── papers/                     Cloned reference repos (git-ignored)
├── results/
│   ├── baseline/               model.pkl, metrics.json
│   ├── explanations/           Baseline SHAP/LIME .npy + feature names
│   ├── scaffold/               Adversarial attack .npy + metrics.json + spatial_correlation.json
│   ├── no_location/            8-feature model + 4 explainer outputs + metrics.json
│   ├── robustness/             rho_matrix.npy, results_matrix.json, trustworthiness labels
│   ├── worst_case/             eps_sweep.npy, feature_shifts.npy, summary.json
│   ├── figures/                Diagnostic figures
│   └── paper_figures/          Publication-ready figures
├── scripts/
│   ├── process_census.py       ACS + TIGER join into King County demographics
│   ├── run_baseline.py         Experiment 0: baseline XGBoost
│   ├── run_explanations.py     Baseline SHAP + LIME attributions
│   ├── run_scaffold_attack.py  Experiment 1: adversarial scaffold
│   ├── run_no_location_experiment.py  Experiment 2: 8-feature ablation
│   ├── run_robustness.py       Experiment 3: 3x4 matrix + trustworthiness
│   ├── run_worst_case.py       Experiment 4: worst-case x* sweep
│   ├── plot_results.py         2x2 summary + honest-vs-adversarial bars
│   ├── plot_no_location.py     Feature ranking figures for Experiment 2
│   ├── plot_robustness.py      3x4 heatmap + trustworthiness map
│   ├── plot_worst_case.py      eps sweep figures
│   ├── plot_demographics.py    Demographic choropleth maps
│   ├── plot_paper_figures.py   Publication-ready figure assembly
│   └── run_all.sh              Full-pipeline reproducibility driver
└── src/
    ├── attacks/
    │   ├── biased_model.py     BiasedModel, InnocuousModel, get_safe_feature_indices
    │   ├── ood_detector.py     train_lime_ood_detector, train_shap_ood_detector
    │   ├── scaffold.py         AdversarialScaffold (.for_lime, .for_shap, .fidelity)
    │   └── worst_case.py       find_worst_case, sweep_eps_geo
    ├── data/
    │   └── load_housing.py     load_housing(dataset, feature_columns, target_column)
    ├── evaluation/
    │   ├── neighborhoods.py    Three-level neighborhood generators + filter
    │   └── robustness.py       Spearman, ensemble, trustworthiness k-NN
    ├── explainers/
    │   ├── shap_runner.py      explain_shap (KernelSHAP), explain_treeshap
    │   ├── lime_runner.py      explain_lime
    │   └── geoshap_runner.py   explain_geoshap (k-NN UTM background)
    ├── models/
    │   └── train.py            train_xgboost
    └── utils/
        ├── io.py               save_model, load_model, save_metrics
        └── paths.py            Project path constants
```

## 9. How to Reproduce

1. **Install dependencies.** From the repo root: `pip install -r requirements.txt`. Core packages: `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `matplotlib`, `lime`, `shap`, `xgboost`, `geopandas`, `shapely`, `pyproj`, `joblib`.
2. **Data prerequisites.** Raw housing data must live at `data/raw/seattle_sample_3k.csv`. The TIGER/Line 2015 Washington tract shapefiles must be unpacked under `data/raw/tl_2015_53_tract/`. `scripts/process_census.py` produces `data/processed/king_county_demographics.csv` and `data/processed/seattle_housing_with_demographics.csv`.
3. **Full pipeline.** Run `bash scripts/run_all.sh`. It executes: `run_baseline.py` → `run_explanations.py` → `run_scaffold_attack.py` → `run_no_location_experiment.py` → `run_robustness.py` → `run_worst_case.py`, then all plotting scripts. Each step writes to its own `results/<experiment>/` subdirectory, so stages can also be run individually by hand.
4. **Expected outputs.**
   - `results/baseline/`: `model.pkl` (~4 MB), `metrics.json` (R² ≈ 0.89).
   - `results/explanations/`: `shap_values.npy`, `lime_values.npy`, `feature_names.npy`, `test_indices.npy`.
   - `results/scaffold/`: three `adv_*_values.npy` arrays, `metrics.json` (OOD accuracies, fidelity, R²), `spatial_correlation.json`.
   - `results/no_location/`: `model.pkl`, four explainer `.npy` arrays, `metrics.json` (R² ≈ 0.62).
   - `results/robustness/`: `rho_matrix.npy` (50×3×4), `results_matrix.json`, trustworthiness label arrays.
   - `results/worst_case/`: `eps_sweep.npy`, `feature_shifts.npy`, `summary.json`.
   - `results/paper_figures/`: publication-ready PNG/PDF figures.

## 10. Key Results Summary

| Experiment | Key metric | Value | Finding |
|---|---|---|---|
| 1 — Adversarial scaffold (SHAP) | UTM_Y mean |SHAP|, honest → adversarial | 0.1179 → 0.0504 (−57%) | Spatial signal laundered into structural features; `pct_black_latino` suppressed to 0.035. |
| 1 — Adversarial scaffold (GeoSHAP) | UTM_Y mean |GeoSHAP|, adversarial | 0.0092 | GeoSHAP is the most compromised — geographic-locality backgrounds trigger the biased branch on every query. |
| 1 — Spatial correlation | Pearson r(pct_minority, biased_pred) | −0.727 (p ≈ 1e-99) | Biased model amplifies redlining by +0.22 in correlation strength vs. honest model. |
| 2 — No-location model | R² (10 features → 8 features) | 0.8895 → 0.6174 | Dropping UTM costs 27 R²-points; `age` rises to 3rd globally as a vintage proxy; LIME upweights view/waterfront. |
| 3 — Robustness matrix | Mean Spearman ρ, spatial × TreeSHAP | 0.520 | On-manifold neighborhoods are *less* stable than random noise; GeoSHAP worst at 0.329. |
| 3 — Trustworthiness | robust / total, spatial TreeSHAP, r_th=0.8 | 1 / 50 | Under a strict on-manifold standard, essentially no explanation is trustworthy. |
| 4 — Worst-case x* | Mean L2 shift at ε=5,000 m | 0.3822 | Worst-case plausible neighbor roughly doubles average spatial perturbation — no adversarial machinery required. |

## Experiment 8b — Scaffold Sweep over pEx-SBM A2 Feature Set + Base-Importance Control

### Motivation

Two gaps remained after Experiment 8 (the original 5-feature Moran's I scaffold sweep):

1. **A2 spectrum incompleteness.** The original Moran's I task list (now in `Notes/pEx-SBM Paper Proposal.md` Appendix C after the Todo-Summary merge) proposed a full Moran's I spectrum using: UTM_Y (high) → sqft_living (mid-high) → grade (mid) → condition (low) → view/waterfront (idiosyncratic). Exp 8 swept waterfront, view, sqft_lot, pct_latino, pct_black. Three candidates — `sqft_living`, `grade`, `condition` — were never run in the original 5-feature sweep; all three are now covered by Exp 8b via `run_feature_sweep.py --extended`.

2. **A4 base-importance confound.** Biasing a highly-predictive feature inflates its adversarial attribution regardless of the attack mechanism. Exp 8's positive Moran's I → visibility trend could therefore reflect base-model importance (features that matter more to XGBoost have more attribution headroom), not a genuine spatial-autocorrelation effect. Exp 8b now normalizes adversarial visibility by pre-attack (honest) feature importance to separate these two effects; Pearson ρ(I, normalized LIME vis) = −0.591 on the 6 structural features, confirming the raw trend is a base-importance artifact.

### Method delta from Experiment 8

- Same protocol: P75 threshold, penalty −0.30, 100 adversarial explanations per explainer, same OOD detector config (`configs/scaffold.yaml`).
- Three new features swept: `sqft_living` (I=0.2905), `grade` (I=0.3651), `condition` (I=0.0899).
- Outputs written to `results/feature_sweep_a2/` (Exp 8 data in `results/feature_sweep/` untouched).
- New metrics added for all 8 features:
  - `honest_importance(f)` = mean |SHAP| of feature f on the honest baseline model (`results/explanations/shap_values.npy`, 200 test points, TreeSHAP).
  - `adv_importance(f)` = mean |SHAP| of feature f on its biased scaffold.
  - `normalized_visibility(f) = adv_importance(f) / honest_importance(f)`.
  - `delta_importance(f) = adv_importance(f) − honest_importance(f)`.
  - Demographic features (`pct_black`, `pct_latino`) have honest importance = 0 (not in baseline model); normalized visibility is undefined (N/A) for these.

### Combined 8-Feature Table

Sorted by Moran's I ascending. OOD accuracy is flat at 99.25–99.58% (SHAP) and 100% (LIME) across all features — confirming Exp 8's finding that the scaffold mechanism is not spatially sensitive.

| Feature | Role | Moran's I | Honest Imp | Raw Vis (SHAP) | Raw Vis (LIME) | Norm Vis (SHAP) | Norm Vis (LIME) | % Penalized | SHAP OOD | LIME OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| waterfront | structural | 0.0512 | 0.001068 | 0.0135 | 0.1059 | 3.933 | 56.463 | 0.8% | 0.9925 | 1.0000 |
| condition | structural | 0.0899 | 0.010128 | 0.0692 | 0.1029 | 2.241 | 6.151 | 7.3% | 0.9925 | 1.0000 |
| view | structural | 0.2423 | 0.011904 | 0.0703 | 0.1597 | 1.924 | 8.283 | 9.7% | 0.9925 | 1.0000 |
| sqft_living | structural | 0.2905 | 0.067630 | 0.1281 | 0.1306 | 0.603 | 1.087 | 25.2% | 0.9925 | 1.0000 |
| grade | structural | 0.3651 | 0.049399 | 0.0941 | 0.0933 | 0.585 | 1.026 | 16.0% | 0.9925 | 1.0000 |
| sqft_lot | structural | 0.4693 | 0.018504 | 0.1340 | 0.1753 | 2.470 | 6.853 | 25.3% | 0.9925 | 1.0000 |
| pct_latino | demographic | 0.7557 | N/A | 0.1355 | 0.1896 | N/A | N/A | 24.7% | 0.9940 | 1.0000 |
| pct_black | demographic | 0.8495 | N/A | 0.1399 | 0.2070 | N/A | N/A | 23.2% | 0.9958 | 1.0000 |

### Interpretation — Does the Moran's I Trend Survive Normalization?

**No. The monotonic trend does not survive base-importance normalization.**

For the 6 structural features (where normalization is defined):

- **Raw visibility vs Moran's I:** Spearman ρ = +0.943, p = 0.005. Strong, significant positive trend — exactly what Exp 8 reported.
- **Normalized visibility vs Moran's I:** Spearman ρ = −0.429, p = 0.397. Trend reverses and becomes non-significant.

The direction of the reversal is diagnostic: low-I features (waterfront, condition, view) have very low honest importance (0.001–0.012) but moderate-to-high adversarial attribution, producing normalized visibility values of 1.9–3.9×. High-I structural features (sqft_living, grade) are already highly important in the honest model (0.050–0.068), so the scaffold inflates them proportionally less (normalized vis ≈ 0.6×). sqft_lot is an outlier — mid-I but low honest importance, yielding normalized vis = 2.5×.

**Conclusion: Exp 8's headline finding (visibility increases with spatial autocorrelation) is substantially a base-importance artifact.** Features with low honest attribution have more room to be inflated under the scaffold, and low-I features (waterfront, condition) happen to also be low-importance in this dataset. After removing the base-importance confound, there is no evidence that spatial autocorrelation per se drives scaffold visibility. The finding that OOD accuracy is invariant to Moran's I remains valid and unaffected by normalization.

**What survives:** The structural-vs-demographic role dichotomy still holds and is independent of normalization — demographic features (pct_black, pct_latino) achieve high raw visibility despite zero base importance, which cannot be explained by base-importance inflation and is a genuine scaffold-mechanism effect.

### Paths to Outputs

- New sweep results (sqft_living, grade, condition): `results/feature_sweep_a2/sweep_summary.json`
- Combined 8-feature table with normalization: `results/feature_sweep_a2/combined_8feature_summary.json`
- Original Exp 8 data (untouched): `results/feature_sweep/sweep_summary.json`
- Honest SHAP baseline used for normalization: `results/explanations/shap_values.npy` + `results/explanations/feature_names.npy`

---

## Experiment 11 — Non-Autocorrelated Bias (SPS School Zone)

**Script**: `scripts/run_school_zone_attack.py` | **Config**: `configs/school_zone.yaml` | **Data**: `scripts/process_school_zone.py` | **Outputs**: `results/scaffold_school_zone/`

### Motivation

Prompted by the 2026-04-16 research note (`Notes/2026-04-16 Non-Autocorrelated Bias Feature.md`): if the biased feature has *low* spatial autocorrelation, does GeoSHAP's vulnerability reverse? GeoSHAP's local geographic background would straddle both sides of a jagged boundary rather than systematically landing inside a single biased cluster, potentially restoring UTM_Y signal that was suppressed in Exp 3. The hypothesis: GeoSHAP goes from most-compromised → most-resistant when bias autocorrelation is low. This experiment tests the school-zone variant (runner-up candidate) to build the empirical basis for that claim.

### Data and Feature Construction

**Source**: Seattle Public Schools elementary school attendance areas 2023–2024 (ArcGIS item `8908525050de4408a7373b0bf79c422d`, publisher `City_Seattle_GIS`, FeatureServer endpoint `services2.arcgis.com/I7NQBinfvOmxQbXs/.../sps_attendance_area_ES_2023_2024`). 58 elementary attendance zone polygons in WGS84, reprojected to EPSG:32610 for spatial join.

**Binary feature construction**: SPS does not publish school quality scores in the GIS layer. We split the 58 zones at the median `es_sch` (school code), assigning the 29 zones with `es_sch ≤ 246` as `in_low_score_zone = 1`. This is a data-availability fallback (documented), not a quality judgment. Zones were chosen to create a spatially jagged binary at the micro scale.

**Spatial join**: Point-in-polygon join of the 3,000 housing UTM coordinates against the 58 zone polygons. 1,231 of 3,000 housing records fall inside Seattle SPS zone polygons (the remaining 1,769 are suburban King County homes outside the city). Analysis is restricted to matched records to avoid inflating Moran's I with a systematic suburb-vs-city signal. 640/1,231 matched records (52.0%) have `in_low_score_zone = 1` — well above the 5% threshold.

### Moran's I for in_low_score_zone

| Feature | Moran's I | z-score | p-value | k-NN |
|---|---|---|---|---|
| `in_low_score_zone` | **0.8006** | 63.08 | 0.0020 | 8 |

**Key finding — autocorrelation is high, not low.** Moran's I = 0.80 is similar to `pct_black_latino` (0.834 in Exp 7). Despite the micro-scale jaggedness of attendance zone boundaries, the zone-level binary has high spatial autocorrelation at the housing-data scale: houses in the same attendance zone are geographic neighbors, so GeoSHAP's 8-NN background still lands predominantly within the same zone. This is the core design lesson of this experiment: zone-level categorical features are inherently autocorrelated at the scale of housing datasets, regardless of boundary jaggedness. The flood zone variant (Exp 10, FEMA NFHL, which follows topographic contours) is the better low-I candidate.

### Attack Setup

| Parameter | Value |
|---|---|
| Bias trigger | `in_low_score_zone > 0.5` |
| Penalty | −0.15 log-price (≈ −14% in linear price; hedonic literature: 3–8% per school-quality tier) |
| % test points penalized | 49.0% (121/247) |
| Innocuous model R² | 0.598 |
| LIME OOD accuracy | 100.0% |
| SHAP OOD accuracy | 99.4% |
| LIME fidelity (Pearson r) | 1.000 (100% routed to biased) |
| SHAP fidelity (Pearson r) | 0.888 (56.3% routed to biased) |

### Attribution Results

Mean |attribution| over 200 test points. Honest baseline uses `results/explanations/{shap,lime,geoshap}_values.npy`. The bias feature `in_low_score_zone` is absent from the honest model by construction.

| Feature | H-SHAP | A-SHAP | H-LIME | A-LIME | H-GeoSHAP | A-GeoSHAP |
|---|---|---|---|---|---|---|
| `in_low_score_zone` | 0.002† | 0.0358 | 0.010† | 0.1435 | —‡ | 0.0308 |
| `UTM_Y` | 0.1179 | 0.0271 (−77%) | 0.1275 | 0.0911 (−29%) | 0.0132 | 0.0119 (−10%) |
| `sqft_living` | 0.0676 | 0.0797 (+18%) | 0.0919 | 0.0931 (+1%) | 0.0603 | 0.0779 (+29%) |

† Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + in_low_score_zone, R²=0.8123, school_zone 1231-record subset). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md`.
‡ Honest GeoSHAP not computed for Exp 11 (skipped per time budget; see comparison doc).

**Spatial correlation**: r(in_low_score_zone, biased_pred) = −0.360 (p = 5.7e-09) vs r(in_low_score_zone, honest_pred) = −0.009 (p = 0.89). The scaffold amplifies the zone-based signal from near-zero to a moderate negative correlation.

### Interpretation and GeoSHAP Reversal Assessment

**GeoSHAP UTM_Y shift: honest = 0.0132 → adversarial = 0.0119 (−10%).**

Compare to Exp 3 (pct_black_latino, Moran's I = 0.834): GeoSHAP UTM_Y collapsed 93% (0.013 → 0.009). Here, with Moran's I = 0.80 (only slightly lower), UTM_Y drops just 10%. By the threshold used in `run_school_zone_attack.py` (less than 30% suppression = reversal confirmed), **the GeoSHAP reversal is nominally confirmed for this experiment**.

However, this is a weak confirmation for two reasons:

1. **The bias feature is not actually low-autocorrelation** (I = 0.80 vs the 0.05–0.30 range needed). The small suppression of UTM_Y under GeoSHAP is better explained by the weaker penalty (−0.15 vs −0.30 in Exp 3) and the lower fraction of biased perturbations seen by GeoSHAP (SHAP OOD at 56.3% routing), not by a structural autocorrelation reversal.

2. **SHAP is more suppressed than GeoSHAP** (UTM_Y: SHAP −77%, GeoSHAP −10%), replicating the direction from Exp 3. This is inconsistent with a true reversal — if autocorrelation were genuinely low, we would expect GeoSHAP to outperform SHAP, not lag it.

**Conclusion**: The school zone binary, as implemented, is not a valid low-autocorrelation test case. It provides scaffolding for the pipeline (data ingest, processing, attack infrastructure) but does not deliver the intended experimental contrast. The FEMA flood zone (Exp 10, parallel agent) is the correct test of the GeoSHAP reversal hypothesis. Future work: verify that Exp 10's `in_floodplain` has Moran's I ≪ 0.30 before drawing conclusions about autocorrelation-driven reversal.

### Paths to Outputs

- `results/scaffold_school_zone/adv_shap_values.npy` — (200, 11) adversarial KernelSHAP attributions
- `results/scaffold_school_zone/adv_lime_values.npy` — (200, 11) adversarial LIME attributions
- `results/scaffold_school_zone/adv_geoshap_values.npy` — (200, 11) adversarial GeoSHAP attributions
- `results/scaffold_school_zone/metrics.json` — OOD accuracy, fidelity, attribution table, routing stats
- `results/scaffold_school_zone/spatial_correlation.json` — Pearson r honest vs biased against zone feature
- `results/scaffold_school_zone/morans_i.json` — I = 0.8006, z = 63.08, p = 0.002 (8-NN, 499 perms)
- `data/processed/seattle_housing_with_school_zone.csv` — 1,231 matched records + es_zone + in_low_score_zone
- `data/processed/sps_elementary_zones.geojson` — 58 SPS elementary attendance zone polygons (cached)

---

## Experiment 10 — Non-Autocorrelated Bias (FEMA Flood Zone)

**Script**: `scripts/run_scaffold_flood_zone.py` | **Config**: `configs/flood_zone.yaml` | **Data**: `scripts/process_flood_zone.py` | **Outputs**: `results/scaffold_flood_zone/`

### Motivation

The 2026-04-16 research note (`Notes/2026-04-16 Non-Autocorrelated Bias Feature.md`) proposes the GeoSHAP reversal hypothesis: GeoSHAP is the most compromised explainer when the biased feature has high spatial autocorrelation (pct_black_latino, I ≈ 0.83), because its k-nearest geographic background always lands inside the same biased region. If the biased feature has *low* autocorrelation — e.g., a FEMA 100-year flood zone that follows topographic contours rather than demographic clustering — GeoSHAP's local background would straddle both sides of the flood boundary, potentially restoring sensitivity to the hidden feature.

**Predicted outcome**: GeoSHAP reverses from most-compromised → most-resistant. This is the primary novel finding being tested.

### Data Ingest

**Source**: FEMA National Flood Hazard Layer (NFHL), layer 28 (`S_Fld_Haz_Ar`), downloaded via tiled ArcGIS REST API queries at `https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer`. The automated download succeeded: 12,793 Esri polygon features were retrieved using a 5×5 tile grid over the housing bbox, parsed from Esri JSON `rings` format into shapely geometries, and reprojected to EPSG:32610 (UTM Zone 10N) for the spatial join.

**SFHA classification**: Polygons with `FLD_ZONE` starting with `A` or `V` are classified as `in_floodplain = 1`. All other zones (X, X500, etc.) are `in_floodplain = 0`.

**Coverage result**: **0.2% (7/3,000) housing records fall inside an SFHA polygon.** This is well below the 3% threshold specified in the proposal. In the 80/20 train/test split, 6 training records and 1 test record are in the flood zone.

**Root cause diagnosis**: The 3,000-record housing sample (`seattle_sample_3k.csv`) is drawn from King County broadly, and flood-zone parcels are systematically underrepresented — they tend to be near water bodies but the housing sample does not cluster there. NFHL queries confirmed 186 AE-zone polygons exist within the study area bounding box, but only 7 housing coordinates fall geometrically within them (verified by sjoin). This is a true sample-coverage issue, not a data-download failure.

### Moran's I for in_floodplain

Computed on the full 3,000-record dataset using 8-NN UTM weights and 499-permutation p-value (exact same method as `scripts/run_morans_i.py`):

| Feature | Moran's I | z-score | p-value | k-NN | n_permutations |
|---|---|---|---|---|---|
| `in_floodplain` | **−0.0015** | −0.21 | 0.110 | 8 | 499 |

**Conclusion**: Moran's I is effectively zero (−0.0015), not significantly different from spatial randomness (p = 0.110). This empirically validates the low-autocorrelation hypothesis from the proposal: flood zone designation in King County is spatially random at the housing-dataset scale. The autocorrelation is genuinely low (I ≪ 0.30), in contrast to the school zone (I = 0.80) and pct_black_latino (I ≈ 0.83). Open Question #1 from the proposal is answered: the hypothesis is supported.

### Attack Setup

| Parameter | Value |
|---|---|
| Bias trigger | `in_floodplain > 0.5` |
| Penalty | −0.20 log-price (≈ −18% in linear price; upper end of hedonic flood zone literature) |
| % test points penalized | **0.2%** (1/600) — well below 3% threshold |
| Innocuous model R² | 0.6414 |
| LIME OOD accuracy | 100.0% |
| SHAP OOD accuracy | 99.3% |
| LIME fidelity (Pearson r) | 0.9997 (99.8% routed to biased) |
| SHAP fidelity (Pearson r) | 0.9533 (71.2% routed to biased) |

The scaffold machinery functioned correctly — OOD detectors trained successfully, fidelity is high, and the pipeline completed. The attack is structurally valid; the limitation is coverage.

### Results Table

Mean |attribution| over 200 test points. Honest baseline: `results/explanations/shap_values.npy` (10-feature, 200 points). Adversarial attributions are 11-feature; honest column is blank (—) for `in_floodplain` (not in baseline model).

| Feature | Honest |SHAP| | Adv |SHAP| | Adv |LIME| | Adv |GeoSHAP| |
|---|---|---|---|---|
| `in_floodplain` | 0.000† | 0.0007 | 0.0668 | 0.0010 |
| `UTM_Y` | 0.1179 | 0.0496 | 0.1287 | 0.0088 |
| `sqft_living` | 0.0676 | 0.0791 | 0.0923 | 0.0693 |
| `grade` | 0.0494 | 0.0730 | 0.0937 | 0.0579 |
| `age` | 0.0133 | 0.0520 | 0.0158 | 0.0346 |

† Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + in_floodplain, R²=0.8934). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md`.

**Fidelity check**: Mean biased prediction inside flood zone (1 test point) = 5.489 vs honest = 5.689 → delta = −0.200 (exactly the penalty). Spatial correlation: r(in_floodplain, biased_pred) = −0.033 (p = 0.41), r(in_floodplain, honest_pred) = +0.004 (p = 0.91) — not significant due to near-zero positives in the test set.

### Interpretation — GeoSHAP Reversal: NOT TESTABLE

**The GeoSHAP reversal hypothesis cannot be tested with this dataset at 0.2% coverage.**

The attribution values for `in_floodplain` — SHAP: 0.0007, LIME: 0.0668, GeoSHAP: 0.0010 — are effectively zero for SHAP and GeoSHAP. With only 1 flood-zone point in 200 explained points (likely 0 in the first 200 test rows), the biased branch fires so rarely that no meaningful bias-attribution signal exists. The "GeoSHAP reversal confirmed" flag in `metrics.json` reflects a technically-true but meaninglessly-small comparison (0.0010 > 0.0007).

The UTM_Y suppression pattern is nearly identical to Experiment 1 (pct_black_latino): SHAP drops 58% (0.1179 → 0.0496), GeoSHAP collapses 93% (0.1179 → 0.0088). This replicates the baseline scaffold behavior independent of the trigger feature — consistent with the finding from Experiments 8/8b that OOD detector performance is invariant to the spatial structure of the bias feature. The scaffold suppresses UTM_Y not because of anything about `in_floodplain` specifically, but because the SHAP/GeoSHAP OOD detectors route real data to the biased model at 71.2%/99.8% regardless.

**What this experiment establishes:**

1. **Moran's I = −0.0015**: `in_floodplain` is genuinely low-autocorrelation — the methodological prerequisite for the reversal test is met. This is the experiment's clean positive result.

2. **Coverage failure**: The 3k-record housing sample systematically underrepresents floodplain parcels (0.2% vs the 8–12% expected from King County parcel-level data). The sample is drawn from a transaction dataset that skews toward inhabited/developed parcels in residential zones, which are not disproportionately in floodplains.

3. **Path forward**: The reversal test requires a dataset with ≥5% flood zone positives. Options: (a) use a parcel-level housing dataset (e.g., King County Assessor records, 500k+ parcels) rather than a transactions sample; (b) extend the SFHA classification to include Zone X Shaded (0.2%-annual-chance, 500-year floodplain) — this would roughly triple coverage but at the cost of a weaker bias narrative; (c) use a different low-I geographic feature with better coverage (e.g., proximity to industrial zones, airport noise contours).

### Paths to Outputs

- `results/scaffold_flood_zone/adv_shap_values.npy` — (200, 11) adversarial KernelSHAP attributions
- `results/scaffold_flood_zone/adv_lime_values.npy` — (200, 11) adversarial LIME attributions
- `results/scaffold_flood_zone/adv_geoshap_values.npy` — (200, 11) adversarial GeoSHAP attributions
- `results/scaffold_flood_zone/metrics.json` — OOD accuracy, fidelity, attribution table, routing stats
- `results/scaffold_flood_zone/spatial_correlation.json` — Pearson r honest vs biased against flood feature
- `results/scaffold_flood_zone/morans_i.json` — I = −0.0015, z = −0.21, p = 0.110 (8-NN, 499 perms)
- `data/processed/seattle_housing_with_flood_zone.csv` — 3,000 records + in_floodplain (7 positives)
- `data/raw/` — no local cache; NFHL data is re-downloaded from FEMA REST API on each run
- `scripts/process_flood_zone.py` — tiled FEMA REST download + Esri polygon parsing + sjoin
- `scripts/run_scaffold_flood_zone.py` — full scaffold attack pipeline with Moran's I inline
- `configs/flood_zone.yaml` — trigger=in_floodplain, penalty=−0.20, threshold=0.5

---

## Experiment 10b — Extended Floodplain (100yr + 500yr Zone X Shaded)

**Script**: `scripts/run_scaffold_flood_500yr.py` | **Config**: `configs/flood_500yr.yaml` | **Data**: `scripts/process_flood_zone_500yr.py` | **Outputs**: `results/scaffold_flood_500yr/`

### Motivation

Experiment 10 produced only 0.2% coverage (7/3,000 records) using 100-yr SFHA zones alone, making the GeoSHAP reversal hypothesis untestable. This experiment extends the floodplain binary to include:

- **100-yr SFHA**: `FLD_ZONE` starting with `A` or `V` (AE, A, AH, AO, VE, etc.)
- **500-yr Zone X Shaded**: `FLD_ZONE = 'X'` AND `ZONE_SUBTY = '0.2 PCT ANNUAL CHANCE FLOOD HAZARD'`

The Zone X Shaded filter was confirmed by inspecting the FEMA NFHL ArcGIS REST API (`ZONE_SUBTY` value counts showed exactly this string for 500-yr parcels). This is documented as Experiment 10b to preserve the 100-yr baseline in `results/scaffold_flood_zone/`.

### Data Ingest and Coverage

**Source**: Same FEMA NFHL ArcGIS REST API (Layer 28, S_Fld_Haz_Ar). Two download methods were used and cross-validated:

1. Tiled 8×8 REST polygon download + `geopandas.sjoin` (point-in-polygon in UTM 10N)
2. Direct point-query REST API for all 3,000 housing records (authoritative, slower)

Both methods agree exactly: **15/3,000 records (0.5%)** are in the extended 500-yr flood zone.

| Subset | Count |
|---|---|
| 100-yr SFHA only (AE/A zones) | 7 |
| 500-yr Zone X Shaded (net new) | 8 |
| **Total in_flood_500yr = 1** | **15** |

**Coverage = 0.5%** — still well below the 3% threshold. This is a true sample-coverage issue: the 3,000-record housing transaction dataset is drawn from inhabited/developed residential parcels which are systematically underrepresented in floodplains. The doubling from 7 → 15 confirms the extension worked correctly, but the underlying sample does not include sufficient floodplain properties for statistical power.

### Moran's I for in_flood_500yr

Computed on the full 3,000-record dataset using 8-NN UTM weights and 499-permutation p-value.

| Feature | Moran's I | z-score | p-value | k-NN | n_permutations |
|---|---|---|---|---|---|
| `in_flood_500yr` | **+0.2392** | +27.45 | 0.002 | 8 | 499 |

Moran's I = +0.2392, which is meaningfully lower than `pct_black_latino` (I ≈ 0.834) and `school_zone` (I ≈ 0.80). The z-score is large because 15 positives cluster more than pure chance (some Zone X Shaded parcels are spatially adjacent), but the absolute autocorrelation is still low. Compared to Experiment 10 (100yr only, I = −0.0015), adding Zone X Shaded slightly increases clustering because 500-yr zones can form larger contiguous patches along watercourses.

### Attack Setup

| Parameter | Value |
|---|---|
| Bias trigger | `in_flood_500yr > 0.5` |
| Penalty | −0.20 log-price |
| % train positive | 12/2,400 (0.5%) |
| % test positive | 3/600 (0.5%) |
| Innocuous model R² | 0.6414 |
| LIME OOD accuracy | 100.0% |
| SHAP OOD accuracy | 99.3% |
| LIME fidelity (Pearson r) | 0.9986 (99.7% routed to biased) |
| SHAP fidelity (Pearson r) | 0.9490 (70.2% routed to biased) |

Penalty verification: mean biased pred inside flood zone = 5.5509 vs honest = 5.7509 → Δ = −0.200 (exact).

### Results Table

Mean |attribution| over 200 test points.

| Feature | Honest |SHAP| | Adv |SHAP| | Adv |LIME| | Adv |GeoSHAP| |
|---|---|---|---|---|
| `in_flood_500yr` | 0.000† | 0.0023 | 0.0896 | 0.0016 |
| `UTM_Y` | 0.1179 | 0.0495 | 0.1279 | 0.0090 |
| `UTM_X` | 0.0186 | 0.0099 | 0.0223 | 0.0094 |
| `sqft_living` | 0.0676 | 0.0794 | 0.0903 | 0.0697 |
| `grade` | 0.0494 | 0.0728 | 0.0934 | 0.0576 |
| `age` | 0.0133 | 0.0520 | 0.0168 | 0.0353 |

† Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + in_flood_500yr, R²=0.8911). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md`.

### GeoSHAP Reversal Test: NOT CONFIRMABLE

**Verdict: PARTIAL / NOT TESTABLE** at 0.5% coverage (3 flood-zone points in 200 explained).

`in_flood_500yr` |GeoSHAP| = 0.0016 vs |SHAP| = 0.0023 → GeoSHAP marginally *hides* the bias more than SHAP (`geoshap_reversal = False`). However, with only 3 test points triggering the biased branch, this comparison is statistically meaningless — both values are noise-floor estimates.

The UTM_Y suppression pattern remains highly consistent across experiments:
- SHAP: 0.1179 → 0.0495 (−58%)
- GeoSHAP: 0.1179 → 0.0090 (−92%)

This replicates the Experiment 10 baseline almost exactly (−58% / −93%), confirming that scaffold UTM_Y suppression is driven by the OOD routing mechanism — not by any property of the trigger feature.

**LIME**: still shows higher trigger visibility (0.0896) than SHAP/GeoSHAP, consistent with its documented resistance to the scaffold in this geospatial setting.

### Figures

- `results/scaffold_flood_500yr/figures/fig_attribution_comparison.png` — 3-panel honest vs adversarial bar chart (SHAP / LIME / GeoSHAP) for the 6 focus features
- `results/scaffold_flood_500yr/figures/fig_geoshap_reversal.png` — side-by-side comparison of GeoSHAP UTM_Y suppression: pct_black_latino (Exp 3, I=0.834) vs in_flood_500yr (Exp 10b, I=0.239)
- `results/scaffold_flood_500yr/figures/fig_coverage_map.png` — scatter of 3,000 housing points in UTM coordinates, coloured by in_flood_500yr

### Paths to Outputs

- `results/scaffold_flood_500yr/adv_shap_values.npy` — (200, 11) adversarial KernelSHAP attributions
- `results/scaffold_flood_500yr/adv_lime_values.npy` — (200, 11) adversarial LIME attributions
- `results/scaffold_flood_500yr/adv_geoshap_values.npy` — (200, 11) adversarial GeoSHAP attributions
- `results/scaffold_flood_500yr/metrics.json` — OOD accuracy, fidelity, coverage, Moran's I, attribution table
- `results/scaffold_flood_500yr/morans_i.json` — I=+0.2392, z=+27.45, p=0.002 (8-NN, 499 perms)
- `results/scaffold_flood_500yr/spatial_correlation.json` — Pearson r: biased=−0.0375, honest=+0.0280
- `results/scaffold_flood_500yr/attribution_table.md` — formatted attribution table + commentary
- `data/processed/seattle_housing_with_flood_500yr.csv` — 3,000 records + in_flood_500yr (15 positives)
- `scripts/process_flood_zone_500yr.py` — 8×8 tiled FEMA REST download + 500yr Zone X Shaded filter
- `scripts/run_scaffold_flood_500yr.py` — full scaffold attack pipeline
- `configs/flood_500yr.yaml` — trigger=in_flood_500yr, penalty=−0.20, threshold=0.5

---

## Experiment 10c — Flood Zone at Parcel-Data Scale

**Script**: `scripts/run_scaffold_parcel_flood.py` | **Config**: `configs/scaffold_parcel_flood.yaml` | **Data**: `scripts/build_kc_parcel_dataset.py` | **Outputs**: `results/scaffold_parcel_flood/`

### Motivation

Experiment 10 established that `in_floodplain` has genuinely low spatial autocorrelation (Moran's I = −0.0015) but failed to test the GeoSHAP reversal hypothesis because the 3k-record housing sample had only 0.2% flood coverage (7 positives). The path-forward specified in Exp 10 was to use KC Assessor parcel-level data (~500k+ parcels) rather than a transaction sample. This experiment executes that pivot. A prior agent completed the data ingest (Step 1 in the pipeline) but crashed before modeling; this agent resumes from the processed CSV on disk.

### Data Source and Prior-Agent Ingest

**Source**: King County Assessor public extracts + ArcGIS REST property sales API.

- **Sales**: KC PropertyInfo MapServer Layer 3, paginated, SalePrice ∈ (50k, 10M). Cached at `data/raw/kc_parcels/kc_sales_raw.csv` (4.5 MB).
- **Residential Building**: `EXTR_ResBldg.csv` (147 MB) — sqft, grade, bath counts, YrBuilt. Primary structures (BldgNbr=1) only.
- **Parcel attributes**: `EXTR_Parcel.csv` (236 MB) — SqFtLot, waterfront, view flags.
- **Demographics**: `king_county_tracts_demographics.geojson` — pct_black_latino from ACS.
- **FEMA NFHL**: cached from Exp 10 ingest.

**Feature imputations** (from `scripts/build_kc_parcel_dataset.py`):
- `bathrooms`: BathFullCount + 0.75×Bath3qtrCount + 0.5×BathHalfCount
- `sqft_living`: SqFtTotLiving (fillna=0)
- `sqft_lot`: SqFtLot; ~5% missing → imputed as 3×sqft_living
- `grade`: BldgGrade (KC 1–13 scale, fillna=7)
- `condition`: Condition (1–5, fillna=3)
- `waterfront`: WfntLocation ≠ 0 → 1
- `view`: sum of 7 binary view flags clipped 0–4
- `age`: 2026 − YrBuilt (YrBuilt clipped 1800–2026, fillna=1970)
- `UTM_X/Y`: EPSG:4326 → EPSG:32610 via pyproj
- `log_price`: log(SalePrice)

**Parcels acquired**: 59,490 records. No nulls. `in_floodplain` positives: 286 (0.48%). Demographics joined via sjoin.

### Baseline XGBoost

| Metric | Value |
|---|---|
| **R²** | **0.7625** |
| RMSE | 0.2846 |
| MAE | 0.1632 |
| n_train | 47,592 |
| n_test | 11,898 |

R² = 0.76 vs 0.89 for the 3k baseline. The gap reflects the more heterogeneous parcel mix (all KC housing types vs. typical transactions). R² > 0.5; pipeline proceeded.

### Moran's I for in_floodplain

Computed on 5,000-point subsample (8-NN UTM weights, 499 permutations; full 59k infeasible for N² weight matrix).

| Feature | Moran's I | z-score | p-value | k-NN | subsample n |
|---|---|---|---|---|---|
| `in_floodplain` | **+0.0507** | +6.94 | 0.0020 | 8 | 5,000 |

I = +0.051 is higher than the 3k estimate (−0.0015) but still very low — ~17× below the pct_black_latino reference (I ≈ 0.83). Statistical significance (p = 0.002) reflects sample power, not substantive autocorrelation. Conclusion: parcel-scale `in_floodplain` confirms the low-autocorrelation property required for the reversal test.

### Flood Coverage

| Subset | Positives | % |
|---|---|---|
| Full dataset (59,490) | 286 | 0.48% |
| Test set (11,898) | 56 | 0.47% |
| First 200 explained | ~1 expected | <1% |

Coverage improved 0.2% → 0.48% over the 3k sample, but remains well below the 3% threshold.

### Scaffold Attack Results

| Parameter | Value |
|---|---|
| Bias trigger | `in_floodplain > 0.5` |
| Penalty | −0.20 log-price |
| % penalized (test) | 0.47% (56/11,898) |
| Innocuous model R² | 0.5871 |
| LIME OOD accuracy | 1.0000 (100.0% routed) |
| SHAP OOD accuracy | 0.9981 (99.6% routed) |
| LIME fidelity (r) | 1.0000 |
| SHAP fidelity (r) | 0.9992 |

### Attribution Table

Mean |attribution| over 200 test points. Honest baseline: `results/explanations/shap_values.npy` (10-feature, 3k dataset).

| Feature | Honest |SHAP| | Adv |SHAP| | Adv |LIME| | Adv |GeoSHAP| |
|---|---|---|---|---|
| `in_floodplain` | 0.000† | 0.0023 | 0.1501 | 0.0016 |
| `UTM_Y` | 0.1179 | 0.2272 | 0.3285 | 0.0107 |
| `UTM_X` | 0.0186 | 0.0614 | 0.1026 | 0.0108 |
| `sqft_living` | 0.0676 | 0.1189 | 0.1406 | 0.0884 |
| `grade` | 0.0494 | 0.1091 | 0.2674 | 0.0655 |
| `age` | 0.0133 | 0.0480 | 0.0492 | 0.0527 |

† Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + in_floodplain, kc_parcel 59k, R²=0.7617). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md`.

Notable pattern: on the larger, more heterogeneous parcel dataset, SHAP and LIME UTM_Y *increase* under the adversarial scaffold (rather than decreasing as in Exp 1/10). The parcel dataset's geographic diversity amplifies the spatial signal. Only GeoSHAP continues to suppress UTM_Y (0.0132 → 0.0107, −19%). LIME uniquely exposes `in_floodplain` attribution (0.1501) while SHAP and GeoSHAP suppress it (~0.002).

### GeoSHAP Reversal Verdict

**PARTIAL — GeoSHAP neither strongly conceals nor reveals the bias.**

- `in_floodplain` |GeoSHAP| = 0.0016 vs |SHAP| = 0.0023 → GeoSHAP conceals marginally more (no reversal).
- UTM_Y GeoSHAP suppression = −18.7% (vs −93% in Exp 1, −92% in Exp 10).
- Root cause: ~1 of 200 explained points is in a flood zone. Both SHAP and GeoSHAP `in_floodplain` values are effectively noise. The PARTIAL verdict reflects insufficient coverage, not evidence against the reversal hypothesis.

**Updated conclusion**: The GeoSHAP reversal hypothesis remains unverified across all three flood zone experiments (Exp 10, 10b, 10c). Coverage is the limiting factor in all cases. A different low-I geographic feature with ≥5% positives is required to test the hypothesis definitively.

### Three Figures

- `results/scaffold_parcel_flood/figures/fig_attribution_comparison.png` — 3-panel bar chart (SHAP/LIME/GeoSHAP), honest vs adversarial for focus features.
- `results/scaffold_parcel_flood/figures/fig_geoshap_reversal.png` — 2-panel: Exp 1 pct_black_latino (high-I, I=0.83) vs Exp 10c in_floodplain (low-I, I=0.051), GeoSHAP UTM_Y suppression comparison. Acid-test visual for the §2 reversal hypothesis.
- `results/scaffold_parcel_flood/figures/fig_sample_comparison.png` — 2-panel map: original 3k sample vs parcel dataset, colored by in_floodplain. Visualizes coverage-expansion rationale (0.2% → 0.48%).

### Paths to Outputs

- `results/scaffold_parcel_flood/baseline_model.pkl` — XGBoost on 10-feature parcel dataset (R²=0.7625)
- `results/scaffold_parcel_flood/baseline_metrics.json` — R², RMSE, MAE, n_train/test
- `results/scaffold_parcel_flood/morans_i.json` — I=+0.0507, z=+6.94, p=0.002 (5k subsample, 8-NN, 499 perms)
- `results/scaffold_parcel_flood/flood_coverage.json` — 286/59,490 positives (0.48%)
- `results/scaffold_parcel_flood/adv_shap_values.npy` — (200, 11) adversarial KernelSHAP
- `results/scaffold_parcel_flood/adv_lime_values.npy` — (200, 11) adversarial LIME
- `results/scaffold_parcel_flood/adv_geoshap_values.npy` — (200, 11) adversarial GeoSHAP
- `results/scaffold_parcel_flood/metrics.json` — full metrics, attribution table, reversal verdict
- `results/scaffold_parcel_flood/spatial_correlation.json` — Pearson r stats
- `results/scaffold_parcel_flood/attribution_table.md` — formatted attribution table
- `results/scaffold_parcel_flood/figures/fig_attribution_comparison.png`
- `results/scaffold_parcel_flood/figures/fig_geoshap_reversal.png`
- `results/scaffold_parcel_flood/figures/fig_sample_comparison.png`
- `data/processed/kc_housing_with_flood.csv` — 59,490 parcels, 16 columns
- `data/raw/kc_parcels/` — EXTR_Parcel.csv (236 MB), EXTR_ResBldg.csv (147 MB), kc_sales_raw.csv (4.5 MB)
- `scripts/build_kc_parcel_dataset.py` — prior agent's ingest script
- `scripts/run_scaffold_parcel_flood.py` — full modeling + scaffold + explanation pipeline
- `configs/scaffold_parcel_flood.yaml` — dataset=parcel_flood, trigger=in_floodplain, penalty=−0.20


---

## Experiment 10d — Alternative Low-I Feature (GeologicHazard)

### Why this feature

Experiments 10, 10b, and 10c all used FEMA flood zone data as the low-I bias feature. All three failed the coverage gate (<4%) in the original `seattle_sample_3k.csv` housing dataset (0.17%–0.48% positive rate). The GeoSHAP reversal hypothesis — that a low-I bias feature causes GeoSHAP's local background to straddle the boundary, partially revealing the bias — therefore could not be tested.

**Feature selected**: `in_geologic_hazard` — binary 1 if a house falls within a King County Erosion Hazard (1990 SAO) or Landslide Hazard (incorporated King County) polygon, as published by the KC iMap Sensitive Areas MapServer (layers 7 and 3 respectively).

**Why this feature over alternatives**:
- Steep slope (KC unincorporated): 0.1% coverage — failed coverage gate.
- Light rail proximity (1000m buffer): I=0.693 — failed Moran's I gate (rail follows linear corridors).
- Tributary basins (KC CAO 2005): I=0.667 — failed Moran's I gate (watershed polygons are spatially continuous).
- Geologic hazard (erosion + landslide): I=0.246, coverage=5.1% — **both gates pass**.

Geologic hazard zones follow topographic micro-features (ravine edges, hollow slopes) that cut across census tracts and neighborhoods at fine spatial scales. Two houses 50m apart can be on opposite sides of a ravine boundary. This produces low spatial autocorrelation at the 8-NN scale. The feature is audit-relevant: geologic hazard designation restricts development rights, raises insurance costs, and can depress lending assessments — a legitimate environmental justice concern distinct from demographic redlining.

**Penalty**: −0.20 log-price on `in_geologic_hazard > 0.5` (identical to flood zone penalty). Realistic: geologic hazard designation is a known price-depressing factor in King County real estate.

### Data source

King County iMap Sensitive Areas MapServer (publicly accessible ArcGIS REST service):
- `https://gismaps.kingcounty.gov/arcgis/rest/services/Environment/KingCo_SensitiveAreas/MapServer`
- Layer 7: Erosion Hazard (1990 SAO) — 293 unique polygons in housing bbox
- Layer 3: Landslide Hazards, incorporated KC — 101 unique polygons in housing bbox
- Combined: 393 polygons tiled 4×4 across housing extent, deduplicated by OBJECTID
- Raw output: `data/raw/kc_geologic_hazard.geojson`
- Processed: `data/processed/seattle_housing_with_geologic_hazard.csv`

Processing script: `scripts/process_light_rail.py` (light rail acquired but abandoned; geologic hazard downloaded inline in `scripts/run_scaffold_geologic_hazard.py`).

### Coverage

**5.1%** — 153 of 3,000 housing records in `seattle_sample_3k.csv` fall within a geologic hazard polygon (point-in-polygon, UTM Zone 10N).

- 39 of 600 test points are penalized (6.5% of test set)
- Passes hard gate [4%, 40%] ✓

### Moran's I

| Metric | Value |
|---|---|
| Moran's I | **0.2462** |
| z-score | +28.12 |
| p-value | 0.0020 |
| k-neighbors | 8 |
| n-permutations | 499 |

- Passes ideal target (<0.3) ✓
- Passes hard gate (<0.35) ✓
- Compare: pct_black_latino I=0.834, waterfront I=0.051, flood zone I≈0.0

### Attack setup

**Config**: `configs/geologic_hazard.yaml`  
**Script**: `scripts/run_scaffold_geologic_hazard.py`  
**Dataset**: `geologic_hazard` (11 features: 10 baseline + `in_geologic_hazard`)  
**Biased model**: baseline XGBoost − 0.20 × 1{in_geologic_hazard > 0.5}  
**Innocuous model**: XGBoost on 8 safe features (no UTM, no trigger), R²=0.6414  
**OOD detectors**: LIME OOD acc=1.000, SHAP OOD acc=0.991  

### Results table

Mean |attribution| over 200 test points.

| Feature | Honest |SHAP| | Adv |SHAP| | Adv |LIME| | Adv |GeoSHAP| |
|---|---|---|---|---|
| `in_geologic_hazard` | 0.000† | 0.0129 | 0.1864 | 0.0114 |
| `UTM_Y` | 0.1179 | 0.0495 | 0.1285 | 0.0091 |
| `sqft_living` | 0.0676 | 0.0789 | 0.0938 | 0.0693 |
| `grade` | 0.0494 | 0.0716 | 0.0936 | 0.0568 |
| `age` | 0.0133 | 0.0530 | 0.0177 | 0.0356 |

† Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + in_geologic_hazard, R²=0.8875). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md`.

Key observations:
- **LIME** is the only explainer that exposes the bias feature: |LIME| = 0.186 (vs honest=0 since feature absent from baseline).
- **SHAP** suppresses `in_geologic_hazard` to 0.013 — similar to flood zone experiments.
- **GeoSHAP** suppresses `in_geologic_hazard` to 0.011 — comparable to SHAP, no reversal.
- UTM_Y is suppressed under SHAP (−58%) and GeoSHAP (−92%), consistent with Exp 3.

### GeoSHAP reversal verdict

**PARTIAL** — GeoSHAP does not strongly reverse. `in_geologic_hazard` |GeoSHAP| = 0.0114 vs |SHAP| = 0.0129 (GeoSHAP hides slightly more, not less).

**Interpretation**: The low-I structure of `in_geologic_hazard` (I=0.246) does not cause a full GeoSHAP reversal because the scaffold's OOD detector (SHAP OOD acc=99.1%) successfully routes most real queries to the biased model. The k=100 geographic background window in GeoSHAP is large enough that even low-I features get averaged out. A full reversal likely requires either higher coverage (>15%) or a feature with near-zero I and more balanced spatial distribution. Flood zone (I≈0) was the theoretically ideal case but was defeated by coverage (0.17%).

**Comparison across experiments**:
| Experiment | Feature | Moran's I | Coverage | GeoSHAP trigger attr | SHAP trigger attr | Reversal? |
|---|---|---|---|---|---|---|
| Exp 3 | pct_black_latino | 0.834 | 14% | 0.0099 | 0.0046 | No (hidden) |
| Exp 10 | in_floodplain (3k) | −0.001 | 0.2% | 0.0010 | 0.0007 | N/A (insufficient coverage) |
| Exp 10d | in_geologic_hazard | 0.246 | 5.1% | 0.0114 | 0.0129 | PARTIAL |

**Conclusion**: Coverage (5.1%) is sufficient to detect a meaningful signal (LIME at 0.186 is strong), but GeoSHAP reversal requires more positive examples in the 200-point explain window (~13 positives here). The reversal hypothesis remains viable but requires a feature with both low-I AND higher coverage for definitive testing.

### Figure references

- `results/scaffold_geologic_hazard/figures/fig_attribution_comparison.png` — 3-panel bar chart (SHAP/LIME/GeoSHAP), honest vs adversarial attributions for focus features.
- `results/scaffold_geologic_hazard/figures/fig_geoshap_reversal.png` — 2-panel acid test: pct_black_latino (I=0.834) vs in_geologic_hazard (I=0.246) GeoSHAP trigger visibility comparison.
- `results/scaffold_geologic_hazard/figures/fig_feature_geography.png` — scatter of 3,000 housing points in UTM coordinates colored by in_geologic_hazard (5.1% orange), showing spatial dispersion of hazard zone boundaries across neighborhoods.

### Output paths

- `data/raw/kc_geologic_hazard.geojson` — 393 KC erosion + landslide polygons (WGS84)
- `data/raw/sound_transit_link_stations.geojson` — 52 Link stations (UTM, acquired but not used as primary feature)
- `data/processed/seattle_housing_with_geologic_hazard.csv` — 3000 records, 12 columns
- `configs/geologic_hazard.yaml`
- `scripts/run_scaffold_geologic_hazard.py`
- `results/scaffold_geologic_hazard/adv_shap_values.npy` — (200, 11)
- `results/scaffold_geologic_hazard/adv_lime_values.npy` — (200, 11)
- `results/scaffold_geologic_hazard/adv_geoshap_values.npy` — (200, 11)
- `results/scaffold_geologic_hazard/metrics.json`
- `results/scaffold_geologic_hazard/morans_i.json`
- `results/scaffold_geologic_hazard/spatial_correlation.json`
- `results/scaffold_geologic_hazard/attribution_table.md`
- `results/scaffold_geologic_hazard/figures/fig_attribution_comparison.png`
- `results/scaffold_geologic_hazard/figures/fig_geoshap_reversal.png`
- `results/scaffold_geologic_hazard/figures/fig_feature_geography.png`

---

## Experiment 12 — pEx-SBM Across Moran's I Spectrum (§6.2 Validation)

### Motivation

The pEx-SBM proposal §6.2 promises that the behavioral fingerprint + pEx-SBM pipeline recovers audit-relevant block structure "across the full Moran's I spectrum." Experiments 8 and 8b established explainer degradation across the spectrum; Exp 12 validates whether pEx-SBM similarly recovers the layer-disassortative block structure for each scaffold.

### Pipeline

For each of 8 features (I ∈ [0.09, 0.85]): (1) fresh BiasedModel penalizing the feature above P75 (penalty −0.30); (2) rung-3 Gaussian fingerprint graph on 200 test points, K_perts=50, σ=0.1×feat_std, 8-NN cosine; (3) plain SBM and pEx-SBM (K=4, N_iter=3000, burn_in=1000, 5 seeds each); (4) Δ NMI, Ψ matrix, C-ratio. pct_black_latino reused from Exp 9. Waterfront skipped (degenerate layer split: 4/200 above P75).

Script: `scripts/run_spectrum_sweep.py`

> **Correction (2026-04-23):** The original `summary.json` reported Plain NMI=0.000, pEx NMI=0.048 for
> `pct_black_latino`. This was a data-lookup bug in `run_spectrum_sweep.py` (seedscale JSON is a dict,
> not a list; fallback silently gave plain_nmi=0.0). Corrected values from `fingerprint_sbm/summary.json`:
> Plain NMI=0.287, pEx NMI=0.334. Δ NMI=+0.048 was always correct. See `spectrum_sweep_diagnostic.md`.

### Master Table

| Feature | Moran's I | Plain NMI | pEx NMI | Δ NMI | C-ratio | Disassortative? |
|---|---|---|---|---|---|---|
| condition | 0.090 | 0.065 | 0.064 | −0.000 | 0.869 | No |
| view | 0.242 | 0.877 | 0.877 | 0.000 | — (sat.) | Yes |
| sqft_living | 0.291 | 0.014 | 0.007 | −0.007 | 0.984 | No |
| grade | 0.365 | 0.014 | 0.020 | +0.006 | 1.001 | Yes |
| sqft_lot | 0.469 | 0.027 | 0.041 | +0.014 | 1.204 | Yes |
| pct_latino | 0.756 | 0.011 | 0.012 | +0.000 | 0.940 | No |
| pct_black_latino | 0.834 | **0.287** (Exp 9) | **0.334** (Exp 9) | +0.048 | 1.319 | Yes |
| pct_black | 0.850 | 0.013 | 0.014 | +0.002 | 0.891 | No |

### Verdict

**PARTIALLY VALIDATED: Δ NMI > 0 for 5/8 features.** The universal §6.2 claim does not hold. The correct framing is regime-conditional:

- **Middle regime (C-ratio 1.0–1.3, I ≈ 0.35–0.83):** pEx fires with meaningful Δ NMI (grade +0.006, sqft_lot +0.014, pct_black_latino +0.048). Layer-disassortative Ψ confirmed for all three.
- **Saturation regime (view, I = 0.24):** Plain SBM achieves NMI = 0.877. pEx has no room to improve — this is scaffold success, not pEx failure.
- **Collapse/diffuse regime (condition, sqft_living, low-I):** C-ratio < 1. Scaffold penalty too diffuse relative to feature variation; fingerprints carry no layer-separating signal.
- **Demographic off-baseline (pct_latino, pct_black):** Very weak positive Δ (< 0.003). Feature absent from baseline model produces near-null fingerprint variation.

**Proposal reframe:** Replace "across the full autocorrelation range" with "in the middle behavioral regime (C-ratio ∈ [1, 3]), spanning I ≈ 0.35–0.83 in the King County data." C-ratio from plain SBM is an auditor-actionable pre-screening heuristic for pEx applicability.

### Reference

- Full summary: `docs/sbm_explainer/spectrum_sweep_summary.md`
- Master JSON: `results/sbm_explainer/spectrum_sweep/summary.json`
- Figures:
  - `results/sbm_explainer/spectrum_sweep/figures/fig_delta_nmi_vs_morans_i.png`
  - `results/sbm_explainer/spectrum_sweep/figures/fig_psi_matrices_grid.png`
  - `results/sbm_explainer/spectrum_sweep/figures/fig_regime_diagnostic.png`
- Per-feature outputs: `results/sbm_explainer/spectrum_sweep/<feature>/`

## Experiment 9b — §2 Primitives on Slack Suite

Following Exp 9's four-dataset Δ NMI replication, we extracted the full pEx-SBM §2 primitives (Ψ matrix, pairwise co-clustering posterior, per-node uncertainty) and block characterizations for COMPAS (Gauss+SHAP σ=0.1, race layer, n=200), German Credit (Gauss+SHAP σ=1.0, gender layer, n=100), and CC psi_two (Gauss+LIME σ=1.0, racePctWhite layer, n=200). In all three datasets the Ψ matrix shows a layer-disassortative structure — not-protected and protected nodes concentrate on different dominant blocks — directly replicating the housing finding. Co-clustering within-over-across ratios are >1 in COMPAS and CC, confirming intra-layer cohesion. Per-node entropy distributions vary: CC psi_two shows the highest proportion of gate-boundary nodes (H>0.3), consistent with its extreme Δ=+0.575. Fingerprint-flip blocks (anchor<tail) do not appear universally across the Slack suite, suggesting Housing's Block 1 flip is a dataset-specific artifact of the geospatial routing boundary rather than a universal scaffold-attack signature. Artifacts: `results/sbm_explainer/slack_outputs/`, summary: `docs/sbm_explainer/slack_suite_summary.md`.
