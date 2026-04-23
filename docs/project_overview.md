# Geospatial XAI Attacks — Project Overview

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

These threads are load-bearing for the April 2026 Peng meeting and a workshop/TMLR vs NeurIPS-main write-up decision. Any update to their content should land in the threaded docs above; this overview is intentionally frozen on the foundational Experiments 1–4 for reference stability.

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

**GeoSHAP** — `src/explainers/geoshap_runner.py:explain_geoshap`. A novel spatial extension of KernelSHAP. The model is location-blind during attribution: UTM coordinates are *not* included in the feature vector fed to the model. Instead, for each explained point, coordinates are used only to retrieve the `k=100` geographically nearest training points (via `sklearn.neighbors.NearestNeighbors` on the UTM plane), and those k points form the per-point SHAP background. This localizes the Shapley reference distribution geographically, so attributions answer "how does this house price differ from neighboring houses" rather than "from all houses."

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
| pct_black_latino | — | 0.0346 | — | 0.1081 | 0.0300 |

Under SHAP, `UTM_Y` drops 57% (from 0.1179 to 0.0504) and `pct_black_latino` sits at a suppressed 0.0346; the spatial signal is laundered into `age`, `grade`, and `bathrooms`. Under LIME, `UTM_Y` remains the top feature and `pct_black_latino` rises to 0.108 — LIME's Gaussian perturbations generate enough real-looking neighbors that the OOD detector still leaks bias. GeoSHAP is the most compromised: `UTM_Y` collapses to 0.0092 and `pct_black_latino` is only 0.030, because GeoSHAP's per-point geographic backgrounds live entirely inside the real-data manifold, so every coalition query hits the biased branch.

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
│   ├── roadmap.md              Step-by-step plan
│   ├── experiment_registry.md  Summary table of every experiment
│   └── project_overview.md     This document
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
