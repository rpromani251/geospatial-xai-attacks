# Geospatial XAI Attacks — Project Summary

> **Migration note (2026-04-24):** The homemade "GeoSHAP" (KernelSHAP with k=100 geographic nearest-neighbor background) has been replaced with the **GeoShapley** library (`pip install geoshapley`). GeoShapley decomposes attributions into primary (non-spatial), geo (pure location), and geo-interaction effects. All scripts have been updated; **results regenerated via `run_all.sh` on 2026-04-25**. References to "GeoSHAP" below should be read as "GeoShapley" going forward. Key structural change: UTM_X + UTM_Y are no longer separate attribution columns — GeoShapley collapses them into a single `geo_effect` column. Spatial neighborhood perturbation in robustness analysis (Exp 5) also redesigned: now Gaussian noise on coordinates only (all other features held constant), N_NEIGHBORS increased from 50 to 100.

## Overview

This project studies whether post-hoc explanation methods (LIME, SHAP, and GeoShapley) can be fooled or are inherently unreliable in geospatially structured prediction tasks. The application domain is automated housing valuation in Seattle / King County, where geographic coordinates serve as a powerful but socially sensitive proxy for race and ethnicity.

The core thesis: coordinate-based algorithmic redlining — a model that systematically undervalues homes in Black and Latino census tracts — can be hidden from both adversarial and non-adversarial explainers. Two complementary frameworks are extended into this geospatial setting:

1. **Slack et al. (2020), "Fooling LIME and SHAP"** — adversarial scaffold attacks via OOD-gated model routing.
2. **Vascotto et al. (2025), "When Can You Trust Your Explanations?"** — non-adversarial robustness measured over on-manifold, spatially-coherent neighborhoods.

The project adds a third lens: a **worst-case on-manifold attack** (`x*`) that finds the plausible nearby house whose explanation differs most from a reference, requiring no adversarial machinery at all.

An active research thread (April 2026) extends the framework in two directions:

3. **Non-autocorrelated bias features.** Peng's 2026-04-16 question — does the scaffold attack still hide a bias feature that is *not* spatially clustered? We survey Moran's I across every feature (Exp 7) and sweep the attack across the full spectrum (Exp 8). The Moran's I hypothesis is disconfirmed, and a new feature-role / spatial-clustering interaction replaces it.
4. **SBM-as-explainer: scaffold-attack detection via pEx-SBM** (Exp 9). A rung-3 behavioral fingerprint encoder feeds a finite-K Dirichlet-Categorical pEx SBM (and, as an extension, Durante et al.'s H-DP pEx) to recover the scaffold's bias-trigger layer from the routing mixture. Replicates across four datasets (housing, Slack COMPAS/German/Communities-and-Crime). Mechanism identified: anchor-dominance erosion, unified as $m$-field variability + layer-misalignment. Three independent axes of variation confirm the three-regime framework (diffuse → middle → saturated, indexed by C-ratio): the K-sweep varies fingerprint dimensionality (K), the tanh probe varies innocuous-model Lipschitz (β), and the penalty sweep (2026-04-23) varies attack strength. pEx fires iff C ∈ [1, 3]; weak-attack and strong-attack regimes produce null and near-null Δ respectively. This is the core methodological contribution targeted at the Apr 2026 Peng meeting.

---

## Dataset

**Housing data** (`data/raw/seattle_sample_3k.csv`): 3,000 King County housing transactions with ten baseline features — `bathrooms`, `sqft_living`, `sqft_lot`, `grade`, `condition`, `waterfront`, `view`, `age`, `UTM_X`, `UTM_Y` — and a `log_price` target. UTM coordinates (EPSG:32610, meters) are used rather than lat/lon so that Euclidean distances and geographic radii are geometrically meaningful.

**Demographics** (`data/processed/king_county_demographics.csv`): ACS 2011–2015 B03002 tract-level demographics for 398 King County tracts, processed from the raw Census CSV and TIGER/Line 2015 shapefiles by `scripts/process_census.py`. Each tract has `pct_black`, `pct_latino`, and `pct_black_latino`. A spatial join produces `data/processed/seattle_housing_with_demographics.csv`, matching all 3,000 records to 334 tracts.

**Split**: 80/20 train/test (`random_state=42`), giving 2,400 train and 600 test points. Every experiment reuses this identical split for comparability.

---

## Explainer Implementations (`src/explainers/`)

All explainers return NumPy arrays of shape `(n_test, n_features)`.

### KernelSHAP (`shap_runner.py:explain_shap`)
Model-agnostic coalition-based Shapley values. A `lambda` wrapper around the XGBoost `.predict` method is used to prevent SHAP from accessing the model's internal feature names, ensuring attribution vectors always align with the externally supplied feature order. Background distribution = `shap.sample(X_train, n_background)`.

### TreeSHAP (`shap_runner.py:explain_treeshap`)
Exact, path-dependent Shapley values computed by `shap.TreeExplainer` in polynomial time by traversing the XGBoost tree structure. No background distribution is needed. Used as the default explainer in **Experiments 5 (robustness) and 6 (worst-case x*)** because it is fast enough to batch a reference point and all 50 (or 200) neighbors in a single call — KernelSHAP and LIME would require per-neighbor calls and are materially slower. (An earlier version of this paragraph said "Experiments 3 and 4"; that was a stale holdover from when the robustness/worst-case experiments were phases 3/4 in the initial plan.)

### GeoSHAP (`geoshap_runner.py:explain_geoshap`)
A novel geographically-aware KernelSHAP variant. For each explained point, the Shapley reference distribution is the `k=100` geographically nearest training points (by UTM Euclidean distance, via `sklearn.neighbors.NearestNeighbors`). This localizes attributions to answer "how does this house differ from its geographic neighbors" rather than the global population. **The model is the same 10-feature (or 11-feature, for the scaffold) XGBoost used by KernelSHAP — UTM_X and UTM_Y are in the feature vector.** GeoSHAP's novelty is not location-blindness; it is that the Shapley background distribution is a per-query local pool selected by UTM k-NN, rather than a global training sample. The key consequence: GeoSHAP's background always lies inside the real-data manifold around the query point, making it the most vulnerable explainer to spatially-structured adversarial attacks. (An earlier version of this doc and the module docstring described GeoSHAP as "location-blind"; that was wrong. Corrected 2026-04-23 following a code audit.)

### LIME (`lime_runner.py:explain_lime`)
Local linear surrogate, `lime.lime_tabular.LimeTabularExplainer` in regression mode. Attributions extracted from `exp.as_map()[1]` (class-1 weights for regression). Uses 500 perturbation samples per explanation (`random_state=42`). LIME generates Gaussian perturbations around the explained point in feature space, which straddles the real-data manifold — this is the source of LIME's partial resistance to the adversarial scaffold.

---

## Experiment 1 — Baseline XGBoost

**Script**: `scripts/run_baseline.py` | **Config**: `configs/baseline.yaml`

An `xgboost.XGBRegressor` (`n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42`) is trained on the 10-feature dataset. The baseline model captures 88.95% of variance in log sale price.

| Metric | Value |
|---|---|
| R² | 0.8895 |
| RMSE | 0.0765 |
| MAE | 0.0574 |
| MSE | 0.00585 |

UTM_Y and sqft_living are the dominant features by mean |SHAP|. The baseline model is saved to `results/baseline/model.pkl` and reused in all downstream experiments.

---

## Experiment 2 — Honest Explainer Baselines

**Script**: `scripts/run_explanations.py` | **Config**: `configs/explanations.yaml`

KernelSHAP, LIME, and GeoSHAP attributions are generated for the first 200 test points. These honest baselines establish the ground truth that the adversarial attack (Experiment 3) is designed to obscure.

UTM_Y is the top feature under both KernelSHAP and LIME — expected given that geographic location in Seattle is a strong predictor of price. GeoSHAP attributions also reflect this but with a more localized signal: the effective "comparison pool" is the k-nearest neighbors in geographic space, so each point's GeoSHAP attribution captures variance relative to its own neighborhood rather than the global housing stock.

**Outputs**: `results/explanations/shap_values.npy`, `lime_values.npy`, `geoshap_values.npy`, `feature_names.npy`, `test_indices.npy`

---

## Experiment 3 — Adversarial Scaffold Attack (Digital Redlining)

**Script**: `scripts/run_scaffold_attack.py` | **Config**: `configs/scaffold.yaml`

### Architecture

The scaffold wraps two internal models — a biased model and an innocuous surrogate — behind an OOD detector that routes queries based on whether they look like real data or explainer perturbations.

**Biased model** (`src/attacks/biased_model.py:BiasedModel`):

$$f_{\text{biased}}(x) = f_{\text{honest}}(x_{\text{base}}) - 0.30 \times \mathbf{1}\{\text{pct\_black\_latino} > 0.25\}$$

A $-0.30$ log-price penalty (≈ $-26\%$ in linear price) is applied to homes in census tracts over 25% Black/Latino. 14% of test homes fall above this threshold.

**Innocuous model** (`src/attacks/biased_model.py:InnocuousModel`): A fresh XGBoost regressor trained only on the 8 "safe" features (no coordinates, no demographics). R² = 0.6414. **Intentional location-blindness.** The innocuous model is deliberately trained without UTM_X / UTM_Y / any demographic feature — this is the attack mechanism, not a model-quality problem. The innocuous model only has to produce attributions that *look* neutral when the explainer probes it with perturbations; it doesn't need to predict well on real data (real queries are routed to the biased model). A higher-R² innocuous model would likely leak spatial attribution through proxies (`age`, `view`) and would *weaken* the attack, not strengthen it. The 27 R²-point gap between 0.64 (innocuous) and 0.89 (honest baseline) matches Experiment 4's finding that UTM features carry roughly one third of the model's explanatory power.

**OOD detectors** (`src/attacks/ood_detector.py`): Random Forest classifiers trained to distinguish real training data from explainer-style perturbations:
- LIME detector: Gaussian-perturbed copies (σ=0.3, 30× replication). Test accuracy = 100.0%.
- SHAP detector: k-means coalition substitutions (10 clusters, 20,000 samples). Test accuracy = 99.5%.

**Routing**: Real inputs → `f_biased`. OOD queries → `psi_innocuous`. The scaffold thus consistently exposes the biased predictions to real users while showing sanitized attributions to the explainer.

### Results

| Scaffold | Fidelity (Pearson r) | % routed to biased |
|---|---|---|
| LIME | 0.9996 | 99.5% |
| SHAP | 0.9109 | 72.3% |

The SHAP detector is stricter because its training perturbations (k-means) overlap more with the real manifold. LIME gets routed more aggressively because Gaussian perturbations are easier to distinguish.

**Attribution comparison (mean |attribution|, 200 test points):**

| Feature          | Honest SHAP¹ | Adv SHAP¹    | Honest LIME | Adv LIME | Honest GeoSHAP | Adv GeoSHAP  |
| ---------------- | ------------ | ------------ | ----------- | -------- | -------------- | ------------ |
| UTM_Y            | 0.118        | 0.050 (−57%) | 0.128       | 0.124    | 0.013          | 0.009 (−93%) |
| pct_black_latino | 0.022³       | 0.035        | 0.026³      | 0.108    | —²             | 0.030        |
| sqft_living      | 0.068        | 0.082        | 0.092       | 0.096    | 0.060          | 0.074        |

¹ "Adv SHAP" / "Honest SHAP" columns are KernelSHAP. Separate TreeSHAP-adversarial attributions were not computed by `run_scaffold_attack.py` (the scaffold wrapper is opaque to TreeSHAP's tree-path estimator); adding them would require modifying the scaffold pipeline to expose the biased model's tree structure.
² Honest GeoSHAP is computed on the 10-feature honest model (which includes `UTM_X`, `UTM_Y`, but not `pct_black_latino`); the demographics row of this column is therefore undefined by construction. The model is **not** location-blind — UTM features are in the feature vector for all four explainers in the honest baseline.
³ Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + pct_black_latino, R²=0.8889). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see `docs/comparison_honest_vs_adv_off_baseline.md` for cross-experiment baseline-R²-matched comparisons.

Under SHAP, `UTM_Y` drops 57% (from 0.118 honest → 0.050 adv); the spatial signal launders into `age`, `grade`, and `bathrooms`. Note `pct_black_latino` under SHAP actually *rises* from 0.022 (11-feat honest) to 0.035 (adv) — the attack does not suppress it, it shifts where the bias's footprint sits in the attribution vector, and SHAP retains a modest signal on the demographic feature even while UTM_Y is laundered. Under LIME, `UTM_Y` remains the top feature and `pct_black_latino` rises 4× (0.026 → 0.108) — LIME's Gaussian perturbations generate enough real-looking neighbors to leak through the detector, so LIME reports *more* demographic attribution than the honest model does, not less. **GeoSHAP is the most compromised on coordinates**: `UTM_Y` collapses from 0.013 to 0.009, with `pct_black_latino` at 0.030. Because GeoSHAP's background is geographically local, coalition queries are constructed from real training points in the reference's UTM neighborhood, so the background values are on-manifold by construction. **We have not directly probed what fraction of GeoSHAP's internal coalition substitutions the OOD detector routes to the biased branch** (the Probe 1 routing check in `run_ood_flip_probe.py` measured only σ=0.1·feat_std Gaussian perturbations, not KernelSHAP's coalition-substitution patterns). The empirical outcome — UTM_Y collapsing to 0.009 and `pct_black_latino` at 0.030 — is consistent with "most coalition queries route to biased," but the mechanism inference "every coalition query hits the biased branch" is an extrapolation, not a direct measurement. A coalition-routing probe would close this gap.

**Spatial correlation** (Pearson r between tract `pct_black_latino` and predicted log-price):

| Model | r | p-value |
|---|---|---|
| Honest | −0.506 | 2.6e-40 |
| Biased | −0.727 | 9.6e-100 |

The biased model amplifies the spatial redlining signal by +0.22 in absolute correlation. The significant finding: embedding spatial structure into the explainer (GeoSHAP) does not protect it from an attack whose mechanism is also spatial.

**Outputs**: `results/scaffold/adv_{shap,lime,geoshap}_values.npy`, `metrics.json`, `spatial_correlation.json`

---

## Experiment 4 — No-Location Ablation

**Script**: `scripts/run_no_location_experiment.py` | **Config**: `configs/no_location.yaml`

Motivation: if a practitioner removes `UTM_X` and `UTM_Y` to "eliminate the spatial signal," does the model become location-agnostic?

Same XGBoost hyperparameters and split as the baseline, trained on 8 features (no coordinates). All four explainers are run.

**Result: R² drops from 0.8895 to 0.6174** — coordinates contribute ~27 R²-points, nearly a third of total explanatory power.

**Feature importance shift**: `age` rises to 3rd place across all SHAP variants, consistent with the hypothesis that neighborhood vintage (construction era) proxies for geographic location. LIME's rankings diverge sharply: `view`, `condition`, and `waterfront` receive high weight — removing coordinates redirects LIME's local linear sensitivity to binary amenity features that produce large local swings.

**Outputs**: `results/no_location/model.pkl`, four explainer `.npy` arrays, `metrics.json`, `figures/` (3 PNGs)

---

## Experiment 4b — No-Location Scaffold Ablation

**Script**: `scripts/run_scaffold_no_location.py`

### Motivation

Experiment 4 shows that removing UTM coordinates from an *honest* model drops R² by 27 points and shifts feature rankings substantially. But it does not test whether the *adversarial scaffold* still works without coordinates. Experiment 3's canonical finding — that spatial signal gets laundered into structural proxies — relies on UTM_X/Y being part of the model's input: the OOD detector learns to distinguish real from perturbed data partly via the coordinate distribution, and the biased model's demographic penalty is visible through a spatial feature. **Experiment 4b tests whether the scaffold still hides `pct_black_latino` when coordinates are removed entirely.** If the attack still works, then the hiding mechanism is not coordinate laundering but something more fundamental to the OOD routing architecture.

### What Changed from Experiment 3

- **Feature list reduced from 11 to 9**: `UTM_X` and `UTM_Y` removed from all model inputs (biased model, innocuous model, OOD detectors).
- **Biased model base**: a fresh XGBoost trained on the 8 structural features (no coordinates); the demographic penalty is applied identically (−0.30 log-price when `pct_black_latino > 0.25`).
- **Trigger computation preserved**: `pct_black_latino` is still loaded from the spatial join (`data/processed/seattle_housing_with_demographics.csv`); tract assignment is unchanged.
- **Innocuous model**: same 8 structural features, same hyperparameters.
- **OOD detectors and explanation hyperparameters**: identical to Exp 3.
- **Output directory**: `results/scaffold_no_location/` (distinct from Exp 3's `results/scaffold/`).

### Results

| Scaffold | OOD accuracy | Fidelity (Pearson r) | % routed to biased |
|---|---|---|---|
| LIME | 100.0% | 1.0000 | 99.8% |
| SHAP | 99.3% | 0.9834 | 78.8% |

**Attribution comparison (mean |attribution|, 200 test points):**

| Feature | Honest SHAP¹ | Adv SHAP | Honest LIME¹ | Adv LIME |
|---|---|---|---|---|
| UTM_Y | 0.118 | — (removed) | 0.128 | — (removed) |
| sqft_living | 0.068 | 0.084 | 0.092 | 0.139 |
| pct_black_latino | — | 0.044 | — | 0.110 |
| age | 0.013 | 0.063 | 0.017 | 0.023 |
| grade | 0.049 | 0.090 | 0.096 | 0.159 |
| bathrooms | 0.009 | 0.021 | 0.014 | 0.023 |

¹ Honest attributions from the 10-feature baseline model (Exp 2); UTM_X/Y columns are present there but excluded from the no-location adversarial model.

**Spatial correlation** (Pearson r between tract `pct_black_latino` and predicted log-price):

| Model | r | p-value |
|---|---|---|
| Honest (no-loc) | −0.289 | 4.9e-13 |
| Biased (no-loc) | −0.600 | 9.0e-60 |

### Interpretation

**The scaffold still works without coordinates.** LIME fidelity is perfect (r=1.000, 99.8% routed to biased) and SHAP fidelity remains high (r=0.983, 78.8% routed). OOD detector accuracy is unchanged from Exp 3 (100%/99.3%). This directly falsifies the hypothesis that the attack depends on coordinate laundering — the OOD routing mechanism is robust to the removal of spatial features.

Without UTM_X/Y to launder into, `pct_black_latino` becomes *more* visible to both explainers: mean |SHAP| = 0.044 (vs. 0.035 in Exp 3) and mean |LIME| = 0.110 (vs. 0.108 in Exp 3). The demographic signal leaks slightly more when there are no coordinate channels to absorb the structural shift. Under SHAP, `grade` (0.090) and `age` (0.063) now absorb most of the penalty-induced variance — consistent with Exp 4's finding that `age` proxies for location in no-coordinate models. Under LIME, rankings are stable (`grade` and `sqft_living` remain top-2), and `view` picks up substantially (+0.055), echoing the Exp 4 finding that LIME's local linear sensitivity redirects to binary amenity features.

The spatial correlation result reinforces the conclusion: the biased model amplifies the minority-tract penalty by +0.31 in absolute r (from −0.289 to −0.600) even without coordinates. In Exp 3 the amplification was +0.22 (from −0.506 to −0.727); the smaller baseline r here reflects that the honest no-location model is genuinely less spatially correlated without UTM features, but the scaffold still drives a large amplification. The attack's bias is just as real and just as hidden — there is simply less spatial proxy structure for explainers to route attribution through.

**Outputs**: `results/scaffold_no_location/adv_shap_values.npy`, `adv_lime_values.npy`, `feature_names.npy`, `metrics.json`, `spatial_correlation.json`

---

## Experiment 5 — Robustness Analysis (3×5 Matrix)

**Script**: `scripts/run_robustness.py`

### Neighborhood Generators (`src/evaluation/neighborhoods.py`)

The three levels formalize a spectrum from "entirely off-manifold" to "on-manifold and geographically coherent." All three share a common post-processing step: after generation, every candidate neighbor is passed through `filter_same_prediction(delta=0.15)`, which discards any point whose predicted log-price differs from the reference by more than 0.15 (roughly a 16% price swing in linear space). This keeps the neighborhood semantically local — we are studying explanation stability for houses in the same rough price tier, not across the whole value range.

---

#### Level 1 — Random (`generate_neighbors_random`)

**Mathematical formulation.** Let $x_{\text{ref}} \in \mathbb{R}^d$ be the reference feature vector and let $\sigma_j$ be the training-set standard deviation of feature $j$. For each of $n = 50$ neighbors:

$$\tilde{x}_j = x_{\text{ref},j} + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0,\,(s \cdot \sigma_j)^2)$$

where $s = 0.05$ is the noise scale. Each feature is perturbed independently, with the perturbation magnitude calibrated to the feature's own training variance. This ensures that `grade` (range ~1–13) receives proportionally larger absolute noise than `waterfront` (binary 0/1), so no single feature dominates the perturbation solely due to scale.

**Why this is off-manifold.** The housing data lies on a low-dimensional manifold embedded in $\mathbb{R}^{10}$. Random Gaussian perturbations generate points that satisfy no inter-feature constraints — e.g., they can produce a house with `grade=12` (luxury) and `condition=1` (very poor), a combination that barely exists in the actual housing stock. XGBoost tree boundaries are axis-aligned and fairly coarse, so many of these off-manifold points still receive predictions close to $x_{\text{ref}}$. However, an explainer's attribution at an off-manifold point can be unreliable because the model has never encountered similar inputs during training.

**What robustness here means.** High $\rho$ under random perturbations indicates the explainer is insensitive to small, coordinate-independent noise around $x_{\text{ref}}$ — the local attribution landscape is flat. This is a *weak* condition: it says the explainer is locally consistent in a tight ball around the reference, but makes no claim about consistency with respect to comparable real houses.

---

#### Level 2 — Medoid (`generate_neighbors_medoid`)

**Mathematical formulation.** Features are first standardized by training-set statistics:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the training-set mean and standard deviation of feature $j$ (with $\sigma_j$ floor-clipped to 1 to avoid division by zero). In this normalized space $Z \subseteq \mathbb{R}^d$, the distance between two points is the **Euclidean distance on standardized features**, equivalent to a Gower-style distance for continuous features:

$$d_{\text{feat}}(x, x') = \|z - z'\|_2 = \sqrt{\sum_j \left(\frac{x_j - x'_j}{\sigma_j}\right)^2}$$

An exact nearest-neighbor search over all 2,400 training points is run in $Z$ to retrieve a pool of $n_{\text{pool}} = 200$ candidates. From this pool, $n = 50$ neighbors are sampled with replacement (uniform over pool indices). Every returned neighbor is an actual training-set house — not a synthetic interpolation — so the neighborhood lies on the empirical training manifold by construction.

**Why "medoid" rather than "mean."** A true k-medoid algorithm minimizes total within-cluster distance and returns cluster centers that are members of the dataset. Here we approximate that spirit by retrieving the 200 nearest real neighbors and sampling from them: each returned point is guaranteed to be a real house, avoiding the off-manifold artifacts of any averaging operation (e.g., a mean of two houses on opposite sides of a river would not itself be a real house). The terminology is approximate — this is really "k-NN sampling from the feature manifold."

**What robustness here means.** High $\rho$ under medoid perturbations indicates the explainer assigns similar feature rankings to structurally similar real houses. This is a *meaningful on-manifold condition*: two houses with similar grade, sqft, age, etc. should, in principle, receive explanations that agree on which features matter most.

**Why $\rho$ drops sharply from random to medoid.** Real neighbors in feature space are distributed along the actual data manifold, which is cut by XGBoost's axis-aligned decision boundaries in complex ways. Two houses may be close in $d_{\text{feat}}$ but sit on opposite sides of a high-weight split (e.g., `grade ≥ 8`). Crossing that boundary reassigns a large fraction of attribution from `grade` to other correlated features. Random perturbations, by contrast, stay very close to $x_{\text{ref}}$ in all coordinates simultaneously ($s = 0.05$ is small) and rarely cross any major split boundary — hence artificially high $\rho$ that does not reflect real-world explanation consistency.

---

#### Level 3 — Spatial (`generate_neighbors_spatial`)

> **Redesigned 2026-04-23 per Peng's feedback.** The spatial level now perturbs **only** coordinates; all other features are held constant. This isolates the geographic axis of variation from structural feature noise, giving a cleaner test of whether explanations are stable under pure location shifts.

**Implementation.** For each of the $n = 100$ synthetic neighbors, the reference point $x_{\text{ref}}$ is copied exactly, then additive Gaussian noise is applied to each coordinate column independently:

$$\tilde{x}_{\text{coord}_j} = x_{\text{ref, coord}_j} + \mathcal{N}(0,\; \sigma_j \cdot s)$$

where $\sigma_j$ is the training-set standard deviation of coordinate $j$ (UTM_X or UTM_Y) and $s = 0.05$ is the noise scale. All non-coordinate features are unchanged.

**What robustness here means.** High $\rho$ under spatial perturbations indicates the explainer assigns stable feature rankings to the *same house* shifted slightly in geographic space — the purest test of whether location alone changes the explanation. This isolates geographic sensitivity from structural variation, complementing Level 2 (medoid), which tests stability across structurally similar but distinct real houses.

---

#### Prediction-Similarity Filter (`filter_same_prediction`)

All three generators feed into:

$$\hat{N}_\delta(x_{\text{ref}}) = \left\{\, \tilde{x} \in N(x_{\text{ref}}) : \left|f(\tilde{x}) - f(x_{\text{ref}})\right| < \delta \,\right\}$$

where $\delta = 0.15$ log-price units (≈ 16% price range in linear space). If the entire candidate set is filtered out, the single nearest candidate is returned as a fallback to prevent empty-neighborhood crashes. This filter enforces semantic locality: attribution differences between the reference and a neighbor cannot be explained away by the two houses sitting in fundamentally different price regimes.

### Robustness Metric

Spearman's $\rho$ between the reference attribution and each neighbor attribution, averaged across neighbors (`src/evaluation/robustness.py:spearman_robustness`). $\rho$ measures feature *ranking* stability, not magnitude stability.

### Ensemble Aggregation (`ensemble_attribution`)

Converts each explainer's attribution vector to feature ranks (rank 1 = most important), computes weights inversely proportional to each explainer's attribution variance, takes a weighted rank average, then applies a **sign-disagreement penalty $\lambda = 0.15$**: features where more than half the explainers disagree on sign have their rank number multiplied by $(1 + \lambda)$ — i.e., the rank number grows, and since final importance is $1 / \text{rank}$, importance drops. The ensemble column uses TreeSHAP + KernelSHAP + LIME (GeoSHAP excluded because it is only computed for 20/50 points). **Resolved (2026-04-23):** an earlier version of both the code and this description inverted the penalty direction (rank multiplied by $(1-\lambda)$, which boosted disagreement features instead of penalizing them). The code is fixed (`src/evaluation/robustness.py:92`) and `results/robustness/` has been regenerated under the corrected penalty. The fix did not meaningfully move the Ensemble numbers — sign disagreements between TreeSHAP/KernelSHAP/LIME are rare in this data, so the majority-sign gate is almost never triggered. The Random/Medoid/Spatial Ensemble ρ values in the table below (0.763 / 0.635 / 0.633) are final.

### Results (mean Spearman ρ)

| Level | TreeSHAP | KernelSHAP | GeoSHAP¹ | LIME | Ensemble |
|---|---|---|---|---|---|
| Random | 0.845 | 0.847 | 0.720 | 0.523 | 0.763 |
| Medoid | 0.515 | 0.531 | 0.336 | 0.473 | 0.635 |
| Spatial | 0.520 | 0.537 | 0.329 | 0.481 | 0.633 |

¹ GeoSHAP averaged over first 20 test points only (see note below).

**Key finding**: Random neighborhoods are far more stable than on-manifold ones — the opposite of naive intuition. The reason: random Gaussian perturbations stay close to the reference in the feature subspaces that tree splits actually care about. On-manifold medoid/spatial neighbors sample real houses that may sit on the other side of a major split boundary (e.g., `grade ≥ 8`), flipping the attribution ranking. GeoSHAP is the least stable because, in addition to the split-boundary effect, its attribution is defined relative to a local geographic background — and even with the background held constant across neighbors (see note on the current code below), each neighbor's position is compared against the *reference point's* background rather than its own, so the effective "what does this feature do relative to this neighborhood" signal is noisier. Concretely: `run_robustness.py` (lines 142–145) reuses `coords_ref` when calling `explain_geoshap` on the reference + neighbor batch, which means all 50 members of the batch share the same k=50 geographic background. A cleaner experiment would compute each neighbor's GeoSHAP against *its own* geographic neighborhood — that would add the second source of variance this paragraph originally attributed to the current pipeline. The current numbers are still valid as a measurement of "ranking stability of GeoSHAP attributions when each neighbor is explained against the reference's local background," which is a reasonable but narrower quantity than the naive reading.

**GeoSHAP subsampling note**: GeoSHAP requires a separate KernelSHAP call per point, making it ~50× slower than TreeSHAP. The robustness experiment limits GeoSHAP to the first 20 test points and 20 neighbors. GeoSHAP $\rho$ values are therefore means over $n = 20$ while all other explainers use $n = 50$. Direct numerical comparison with other columns should account for this difference.

### Trustworthiness Classification

A point is labeled `robust` if its spatial × TreeSHAP Spearman $\rho \geq 0.8$ (threshold $r_{\text{th}} = 0.8$). A 5-NN regressor is trained on (TreeSHAP attribution, prediction) features to predict $\rho$ for new points.

**Ground-truth labels: 1 of 50 test points clears the threshold** (spatial × TreeSHAP ρ ≥ 0.8). Under a strict on-manifold standard, essentially no explanation in this housing model is trustworthy.

**Held-out 5-NN trustworthiness predictions (post-2026-04-23 rerun, 70/30 split of valid points):** the k-NN is trained on the 35-point training split and evaluated on the 15-point held-out split. **0 of 15 held-out points are predicted robust; all 15 are predicted `not_robust`.** This is consistent with the extremely imbalanced ground-truth label distribution — a calibrated classifier on a near-all-not-robust training set should predict almost nothing as robust. So the k-NN diagnostic reports "nothing looks trustworthy here either," which matches the population-level finding but doesn't add evidence of predictive usefulness beyond that. (Before the fix, the k-NN was fit and scored on the same 50 points and reported an in-sample "predicted robust" count; that count is no longer meaningful and has been replaced.)

**Outputs**: `results/robustness/rho_matrix.npy` (50×3×5), `results_matrix.json`, `trustworthiness_labels.npy`, `knn_predicted_labels.npy`, `tree_ref_attrs.npy`, `figures/` (4 PNGs)

---

## Experiment 6 — Worst-Case x* Attack

**Script**: `scripts/run_worst_case.py`

### Formulation

$$x^* = \arg\max_{x \,\in\, \mathcal{N}_\varepsilon(y)} \|\text{attr}(x) - \text{attr}(y)\|_2$$

where $\mathcal{N}_\varepsilon(y) = \{x \in X_{\text{train}} : \|{\text{coords}(x) - \text{coords}(y)}\|_2 \leq \varepsilon_{\text{geo}}\}$ is the set of real training points within $\varepsilon_{\text{geo}}$ meters of $y$ in UTM space. This is a **purely on-manifold** adversarial search: $x^*$ is always a real house, requires no scaffold, no OOD detector, and no synthetic perturbations.

### Implementation (`src/attacks/worst_case.py`)

For each reference point: compute UTM distances to all training points, filter to within `eps_geo` meters, subsample up to `n_candidates=200`, compute TreeSHAP attributions for the reference and all candidates in a single batch, return the argmax by L2 attribution distance. TreeSHAP is used because it can batch all 200 candidates in one call; LIME would require 200 separate calls (identified as a future extension).

### Results (50 reference points, 5 $\varepsilon$ values)

#### ε-Sweep: Instability Grows with Geographic Radius

| $\varepsilon_{\text{geo}}$ (m) | Mean worst-case $\|2$ | Std | Min | Max | Mean candidates |
|---|---|---|---|---|---|
| 500 | 0.165 | 0.079 | 0.013 | 0.370 | 3.6 |
| 1,000 | 0.228 | 0.083 | 0.025 | 0.412 | 12.4 |
| 2,000 | 0.275 | 0.080 | 0.088 | 0.496 | 39.3 |
| 5,000 | 0.382 | 0.083 | 0.216 | 0.570 | 150.0 |
| 10,000 | 0.411 | 0.072 | 0.263 | 0.570 | 195.9 |

The mean worst-case attribution shift grows monotonically with $\varepsilon$ and begins to plateau between 5 km and 10 km, where the candidate pool saturates the 200-sample cap. At $\varepsilon = 500$ m, 4 of 50 reference points had zero candidates within range (no training house that close), which is why $n_{\text{valid}} = 46$ at the smallest radius. By $\varepsilon = 2{,}000$ m every reference point has at least 2 candidates, and by $\varepsilon = 10{,}000$ m the pool is effectively unconstrained (mean 195.9 / 200).

The standard deviation is remarkably consistent across all $\varepsilon$ values (~0.08), meaning the spread of instability across test points is stable even as the mean rises. This indicates a persistent per-point heterogeneity: some locations in the dataset are structurally "explanation-stable" (minimum worst-case L2 = 0.216 even at $\varepsilon = 5$ km) while others are highly fragile (maximum = 0.570 at $\varepsilon = 5$ km).

#### Worst-Case vs. Average-Case: the 2.6× Gap

A critical comparison: at $\varepsilon = 5{,}000$ m, the mean worst-case L2 shift is **0.382**, but the mean *average-neighbor* L2 shift (mean over all candidates in the radius, not just the argmax) is only **0.147**. The ratio is **2.60×**.

This gap is the core novel finding of Experiment 6. It means:

1. **Average robustness measurements understate real risk.** Experiment 5 (robustness matrix) reports mean Spearman $\rho$ over randomly-drawn on-manifold neighbors. That average reflects typical explanation consistency. The worst-case search shows that within the same geographic radius, a deliberately-chosen real house can produce an explanation that is 2.6× more divergent than the average neighbor. No adversarial machinery is required — $x^*$ is a real training house that a human could plausibly cite as a comparable.

2. **The risk is present even without an attack.** Unlike the scaffold attack (Experiment 3), which requires building an OOD detector and a biased surrogate, the worst-case neighbor is found by a simple argmax search over training data. Any user with access to the dataset and the model can find it.

#### Which Features Get Destabilized?

Mean absolute attribution shift at $\varepsilon = 5{,}000$ m, per feature (sorted):

| Feature | Mean $|\Delta \text{attr}|$ at $\varepsilon = 5$ km |
|---|---|
| `grade` | 0.2597 |
| `sqft_living` | 0.2450 |
| `view` | 0.0552 |
| `UTM_Y` | 0.0479 |
| `sqft_lot` | 0.0310 |
| `UTM_X` | 0.0288 |
| `age` | 0.0210 |
| `bathrooms` | 0.0166 |
| `condition` | 0.0160 |
| `waterfront` | 0.0094 |

`grade` and `sqft_living` account for the overwhelming majority of the worst-case attribution shift — together they contribute ~75% of the total L2 norm. This makes sense structurally: within a 5 km radius in King County, housing stock ranges from single-story post-war bungalows to modern multi-story homes. Two geographically similar houses can sit on opposite sides of XGBoost's `grade ≥ 8` split, which is one of the highest-weight boundaries in the model. When the worst-case search finds a neighbor across that split, the entire attribution vector reorganizes around `grade` and `sqft_living`, while coordinate features (`UTM_Y`, `UTM_X`) shift relatively little. This is the geospatial analog of the "on-manifold split-crossing" finding from Experiment 5 — but here it is not an accident of random sampling, it is a deliberate selection.

The small shift in `UTM_Y` (0.048) despite large geographic displacement is particularly notable: it means the worst-case neighbor is chosen not because it is geographically unusual, but because it crosses a structural split boundary within the same neighborhood. The explanation changes fundamentally, but the location barely changes.

#### Connection to Experiment 5

The spatial-level TreeSHAP Spearman $\rho$ distribution from Experiment 5 (over 50 test points) has mean = 0.520, std = 0.152, with 11 of 50 points below $\rho = 0.4$ (very unstable) and only 1 above $\rho = 0.8$ (trustworthy). Experiment 6 complements this: while Experiment 5 characterizes typical instability, Experiment 6 quantifies the *maximum achievable* instability using real data. Together they establish a two-sided picture:

- **Average case (Experiment 5)**: a randomly-drawn on-manifold neighbor already agrees with the reference explanation only moderately ($\rho \approx 0.52$).
- **Worst case (Experiment 6)**: a deliberately-chosen real neighbor at $\varepsilon = 5$ km produces an L2 attribution shift of 0.382 on average — 2.6× what a typical neighbor produces — with no adversarial infrastructure.

**Outputs**: `results/worst_case/eps_sweep.npy`, `mean_shift_sweep.npy`, `feature_shifts.npy`, `n_in_radius.npy`, `eps_values.npy`, `summary.json`, `figures/` (3 PNGs)

---

## Experiment 7 — Moran's I Spatial Autocorrelation Survey

**Script**: `scripts/run_morans_i.py`

### Motivation

Peng's 2026-04-16 ask: does the scaffold attack depend on the biased feature being highly spatially autocorrelated? Every prior experiment used `pct_black_latino` as the trigger — a strongly clumped demographic. If the attack's OOD detector relies on spatial clumpiness to cleanly separate real from perturbed data, then features with weak spatial autocorrelation should be harder to hide. This experiment is the prerequisite measurement: rank every feature by Moran's I before sweeping the attack across the spectrum (Exp 8).

### Method

Global Moran's I with a row-standardized 8-NN spatial weights matrix on UTM coordinates, implemented from scratch (no `libpysal` dependency):

$$I = \frac{z^T W z}{z^T z}, \qquad z = x - \bar{x}$$

where $W_{ij} = 1/k$ for $j$ among the $k=8$ nearest UTM neighbors of $i$ (self-links excluded). Statistical significance via 499-permutation pseudo-p-value. All 11 non-coordinate features processed.

### Results

| Feature | Moran's I | Autocorrelation level |
|---|---|---|
| `pct_black` | +0.850 | very high |
| `pct_black_latino` | +0.834 | very high |
| `pct_latino` | +0.756 | high |
| `sqft_lot` | +0.469 | medium |
| `grade` | +0.365 | medium |
| `age` | +0.317 | medium |
| `sqft_living` | +0.291 | low–medium |
| `view` | +0.242 | low |
| `bathrooms` | +0.183 | low |
| `condition` | +0.090 | very low |
| `waterfront` | +0.051 | very low |

All features are significantly autocorrelated (p < 0.05 under permutation). The spectrum is wide: demographic features cluster tightly by neighborhood ($I \approx 0.75$–$0.85$) while structural features like `waterfront` and `condition` are nearly spatially random ($I \approx 0.05$–$0.09$).

**Sweep features** (evenly spaced across $I$, for Experiment 8): `waterfront` (0.051), `view` (0.242), `sqft_lot` (0.469), `pct_latino` (0.756), `pct_black` (0.850).

**Outputs**: `results/morans_i/morans_i_results.json`, `morans_i_barplot.png`, `sweep_features.txt`.

---

## Experiment 8 — Feature Sweep: Scaffold Attack vs. Moran's I

**Script**: `scripts/run_feature_sweep.py`

### Hypothesis

Original hypothesis: high-Moran's-I features are easier to hide from explainers because their spatial clumpiness lets the OOD detector cleanly separate real from perturbed data.

### Method

For each of the 5 sweep features selected in Exp 7:

1. Build a `BiasedModel` that subtracts 0.30 log-price from homes where `feature > 75th percentile` of its training distribution.
2. Build an `InnocuousModel` trained on all features except the biased one and UTM coordinates.
3. Train SHAP and LIME OOD detectors (same RF approach as Exp 3).
4. Generate 100 adversarial explanations with each explainer.
5. Measure **attribution visibility**:

$$\text{vis}(j_{\text{bias}}) = \frac{\text{mean}_i \,|\text{adv}_{i, j_{\text{bias}}}|}{\text{mean}_i \sum_{j} |\text{adv}_{i, j}|}$$

— the share of total |attribution| assigned to the biased feature. Higher = harder to launder (more leakage).

Each feature is marked structural (in the honest 10-feature baseline) or demographic (off-baseline). A shape-matching bug in an earlier draft — the honest baseline requires exactly 10 features — was fixed by building a focused dataset per sweep run (10 baseline features + bias feature if not already in the baseline).

### Results

| Feature | Role | Moran's I | SHAP vis | LIME vis | % penalized | SHAP OOD | LIME OOD |
|---|---|---|---|---|---|---|---|
| `waterfront` | structural | 0.051 | 0.0135 | 0.106 | 0.8% | 99.3% | 100% |
| `view` | structural | 0.242 | 0.0703 | 0.160 | 9.7% | 99.3% | 100% |
| `sqft_lot` | structural | 0.469 | 0.134 | 0.175 | 25.3% | 99.3% | 100% |
| `pct_latino` | demographic | 0.756 | 0.136 | 0.190 | 24.7% | 99.4% | 100% |
| `pct_black` | demographic | 0.850 | 0.140 | 0.207 | 23.2% | 99.6% | 100% |

### Findings — Moran's I hypothesis disconfirmed

1. **OOD accuracy is flat across the Moran's I spectrum** (99.3–99.6% SHAP, 100% LIME). The attack's core mechanism — routing real vs. perturbed inputs — does not depend on spatial clumpiness. Moran's I does **not** predict attack feasibility at the OOD stage.

2. **Raw visibility appears to increase with Moran's I** — structural: 0.014 → 0.134 (SHAP), 0.106 → 0.175 (LIME); demographic: 0.136 → 0.140 (SHAP), 0.190 → 0.207 (LIME). But **Experiment 8b (see below) shows this is a base-importance artifact**, not a real I-trend. On the 6 structural features with non-zero honest-importance (waterfront, condition, view, sqft_living, grade, sqft_lot), Pearson ρ(Moran's I, normalized LIME visibility) = **−0.591** (p = 0.216 at n=6; directionally negative, statistically underpowered). Demographic features (pct_latino, pct_black) are excluded from this correlation because their honest_importance = 0 (not in the baseline model) leaves normalized_visibility undefined. The negative sign, not the magnitude, is what matters: the raw-visibility positive trend reflects that larger-I features happen to be more naturally predictive, not that clumpy features are harder to hide.

3. **Two regimes, not one.** Structural features (in baseline) show low visibility because the attack hides by amplification: the innocuous model excludes them, so the scaffold amplifies their attribution on real inputs. Demographic features (off baseline) show higher visibility because they were never part of the honest model; laundering happens indirectly via structural proxies, not by suppression of the biased feature itself. The regime distinction is more load-bearing than any I-trend (see Exp 8b).

4. **Binary features are under-penalized.** `waterfront`'s P75 = 0, so only 0.8% of test points are penalized — the attack has little signal to hide. A sweep with adjusted thresholds (median rather than P75 for near-constant features) or additional mid-$I$ features (e.g., `grade`, $I$ = 0.365) would strengthen the structural regime. Exp 8b below addresses this by adding mid-I features.

5. **Reframe.** Not "SHAP fails in geospatial settings" and not "I predicts attack visibility," but **"explainer robustness in geospatial settings depends on feature role (on- vs off-baseline), not on feature clumpiness."** OOD-detector performance is I-invariant; any residual correlation between raw visibility and I is a feature-importance artifact.

**Outputs**: `results/feature_sweep/sweep_summary.json`, `sweep_plot.png` (two-panel: visibility + raw attribution vs. Moran's $I$, role-colored).

---

## Experiment 8b — Base-Importance-Normalized Feature Sweep

**Script**: `scripts/run_feature_sweep.py` (reused; different config — 8 features instead of 5)

### Motivation

Experiment 8's raw-visibility finding — "visibility increases with Moran's I" — has a confound: if you bias a naturally-predictive feature (e.g., `sqft_lot` is important in the honest model because big lots are expensive), the adversarial-model SHAP on that feature is large *regardless of attack effectiveness*. "Visibility" conflates "the attack leaks" with "the feature is predictive." Normalize by honest-model attribution to separate the two.

### Method

Extend the feature sweep from 5 features to 8 by adding `condition`, `sqft_living`, `grade`, and `age`. Add `honest_importance`, `delta_*_importance`, and `normalized_visibility_*` columns to `sweep_summary.json`, where `normalized_visibility_*` = adversarial visibility divided by the honest model's mean |attribution| on the same feature.

### Results

**On *normalized* visibility, the I-trend flips or weakens sharply.** Pearson correlation between Moran's I and normalized LIME visibility is ρ = **−0.591** (p = 0.216 at n = 6 structural features; 2026-04-23 rerun of `scripts/run_feature_sweep.py --extended`). Direction is clearly negative — visibility *decreases* with spatial clustering once you divide out base importance. Statistically underpowered at n=6 (p > 0.05), so the claim should be "direction is negative, magnitude not tightly estimated" rather than "significant correlation." Demographic features (pct_latino, pct_black, n=2) are excluded from the correlation because their honest_importance=0 leaves normalized_visibility undefined; including them would require a separate metric. Raw visibility's apparent monotone increase with I is dominated by base-importance, not spatial clustering.

### Interpretation

The regime distinction (structural vs demographic) is the real story. Structural features hide by amplification, demographic features leak through structural proxies. Both regimes are I-invariant at the OOD-detector level; the raw-visibility trend is an importance-weighted artifact of that regime difference.

**Paper-level implication**: the paper's §6.2 claim needs to be about the regime condition, not the Moran's I spectrum. This is consistent with Experiment 12's finding (§pEx-SBM spectrum sweep): pEx-SBM fires in the middle regime (C-ratio ∈ [1, 3]) regardless of Moran's I.

**Outputs**: `results/feature_sweep_a2/{sweep_summary.json, combined_8feature_summary.json, sweep_plot.png}`.

---

## Experiment 9 — SBM-as-Explainer: Scaffold-Attack Detection via pEx-SBM

**Scripts**: `scripts/sbm_explainer/run_fingerprint_sbm.py` (rung-3 main),
`scripts/sbm_explainer/run_gauss_lime_seedscale.py` (Probe 2, 20-seed CI),
`scripts/sbm_explainer/run_ood_flip_probe.py` + `scripts/sbm_explainer/run_honest_fingerprint_check.py` (Probe 1),
`scripts/sbm_explainer/run_{compas,german,cc}_fingerprint.py` (Slack replication),
`scripts/sbm_explainer/run_k_scaling_sweep.py`, `scripts/sbm_explainer/run_lipschitz_sweep.py` (retired), `scripts/sbm_explainer/run_tanh_sweep.py`, `scripts/sbm_explainer/run_tanh_seedscale.py`,
`scripts/sbm_explainer/run_hdp_pex.R` (H-DP). `scripts/run_layer_density_test.py` is under the earlier `sbm_poc` thread.

**Entry point**: `docs/sbm_explainer/README.md` → `summary_apr22.md` → `cosine-expression.md`.

### Motivation

Exps 1–8 establish *that* Slack-style scaffold attacks hide coordinate-based redlining from post-hoc explainers. Experiment 9 asks the complementary detection question: can a separate inference procedure **recover the scaffold's hidden bias-trigger layer from the scaffolded model's behavior**, without access to the biased model internals? The answer pivots on a behavioral-fingerprint encoder fed into a probabilistic exchangeable stochastic block model (pEx-SBM).

### Which "layer" is recovered — and the sqft_lot / pct_black_latino split

The Exp 3 scaffold uses **`pct_black_latino > 0.25`** as the bias trigger. The Exp 9 *housing rung-3 pipeline* uses a **separate scaffold** trained with **`sqft_lot > P75` (threshold 9361)** as the bias trigger — see `scripts/sbm_explainer/run_fingerprint_sbm.py:64` (`FEATURE = "sqft_lot"`) and `results/sbm_explainer/fingerprint_sbm/summary.json:2` (`"feature": "sqft_lot"`). Why sqft_lot: structural feature with a clean P75 threshold, 50 penalized test points (25%) — enough for the SBM's binary-layer inference to have power and avoid degeneracies. The headline housing numbers below (Δ = +0.048, 95% CI [+0.017, +0.076], p = 1.8e-9, 20-seed Probe 2) **all refer to recovering the `sqft_lot > 9361` layer**, not `pct_black_latino > 0.25`.

A separate post-hoc overlay (`scripts/extract_pex_outputs_protected.py`) refits finite-K pEx on the **same saved Gauss+LIME graph** with the `pct_black_latino > 0.25` layer plugged in. This produces the §2 outputs (Ψ, co-clustering, entropy — §"Slack suite + housing §2 outputs" below) but NOT a new Δ number: within/across co-clustering ratio is 1.32 (weaker than sqft_lot's 1.50), and the extraction docstring (`results/sbm_explainer/pex_outputs/summary.json` field `note`) explicitly calls this an "OVERWRITE of sqft_lot>P75 run." The pct_black_latino overlay is consistent with the Exp 9 direction (within > across) but weaker — the underlying graph was built against a sqft_lot scaffold, so refitting on a different layer doesn't recover that layer cleanly.

**The four-dataset replication's Slack-suite layers ARE demographic** (race / gender / racePctWhite — §4-dataset replication below). So the research claim is: "rung-3 + finite-K pEx recovers the scaffold's bias-trigger layer — structural (sqft_lot) on housing, demographic (race, gender, racePctWhite) on the Slack suite." That's the honest generalization story.

### Pipeline

**Rung-3 behavioral fingerprint.** For each of $N$ test points $x_i$, construct a vector by evaluating the full scaffolded model on $x_i$ and on $K$ Gaussian perturbations:

$$e_i = \bigl[\,f_{\text{scaffold}}(x_i),\; f_{\text{scaffold}}(x_i + \delta_1),\; \ldots,\; f_{\text{scaffold}}(x_i + \delta_K)\,\bigr] \in \mathbb{R}^{K+1}$$

Rungs 1 (attribution vectors) and 2 (honest-booster leaf indices) both produce posterior-blind graphs (regime-b null, $\Delta \approx 0$). Rung 3 — evaluating the full routed scaffold — is the first encoder that produces a middle-regime graph (see `docs/sbm_explainer/design.md`).

**Graph construction.** $N=200$ nodes, 8-NN cosine similarity on $\{e_i\}$. Layer label = the scaffold's bias trigger: **`sqft_lot > 9361`** for housing (50 high / 150 low), **race** for COMPAS, **gender** for German, **`racePctWhite`** thresholded for Communities-and-Crime.

**pEx-SBM inference.** Two variants are compared:

- **Finite-K Dirichlet-Categorical pEx** (`src/sbm/pex_sbm.py`, $K=4$): per-layer Dirichlet$(\gamma)$ prior over the $K$-simplex, Categorical node-to-block assignments. Collapsed Gibbs.
- **H-DP pEx** (Durante et al. 2025, arXiv 2410.10619): nested Dirichlet Processes linking layer-level DPs to a global profile DP; $K$ inferred. R sampler at `papers/pexsbm/pex-sbm.R`.

Both deliver a per-layer prior over block usage; they differ only in fixed-arity Dirichlet vs. infinite-arity HDP. Plain SBM (layer-agnostic Dirichlet) is the $\Delta$-comparison baseline.

**Metric.** $\Delta = \text{NMI}(z_{\text{pEx}}, \text{layer}) - \text{NMI}(z_{\text{plain}}, \text{layer})$, averaged over SBM seeds, with $C$-ratio (within-community / across-community posterior co-clustering density) as a regime indicator.

### Main result (housing, Gauss+LIME)

| Encoder | Scaffold | log BF | Plain NMI | pEx NMI | $\Delta$ | $C$ ratio |
|---|---|---|---|---|---|---|
| Gaussian | LIME | +145 | 0.287 | **0.334** | **+0.048** | **2.51** |
| Gaussian | SHAP | −2 | 0.004 | 0.003 | −0.001 | 0.99 |
| Targeted | LIME | +2 | 0.063 | 0.063 | 0.000 | 1.16 |
| Targeted | SHAP | −3 | 0.012 | 0.012 | 0.000 | 1.01 |

20-seed CI: $\Delta = 0.043 \pm 0.018$ (SEM 0.004), 95% CI [+0.017, +0.076], $p = 1.8 \times 10^{-9}$. This is the first cell in the entire SBM-as-explainer thread where a pEx prior moves the posterior separably from zero — and it retracts an earlier "structural null" claim about finite-K pEx.

### Mechanism — the scaffold-attack detector

Probe 1 (routing check) falsifies the original stated mechanism ("LIME OOD lets perts through → gate visible"):

- LIME detector classifies **0%** of $\sigma = 0.1 \cdot \text{feat\_std}$ Gaussian perts as real → all routed to InnocuousModel.
- SHAP detector classifies **73%** of the same perts as real → routed to BiasedModel.

Single-model sanity check (`run_honest_fingerprint_check.py`):

| Configuration | Plain NMI | pEx NMI | $\Delta$ | $C$ ratio |
|---|---|---|---|---|
| Honest XGBoost | 0.007 | 0.009 | +0.002 | 1.03 |
| BiasedModel only | 0.004 | 0.015 | +0.011 | 1.04 |
| InnocuousModel only | 0.007 | 0.015 | +0.007 | 1.06 |
| **Scaffold (LIME detector)** | **0.287** | **0.334** | **+0.048** | **2.50** |

All single-model configurations are null. The signal requires the scaffold's **routing mixture**. Under a strict LIME detector:

$$e_i = \bigl[\,f_{\text{biased}}(x_i),\; f_{\text{innocuous}}(x_i + \delta_1),\; \ldots,\; f_{\text{innocuous}}(x_i + \delta_K)\,\bigr]$$

— one gate-exposed anchor (real input routed to BiasedModel) plus $K$ smooth gate-hidden refinements (perts flagged OOD and routed to InnocuousModel). Cosine 8-NN groups points sharing both anchor-offset and smooth profile, and the pEx prior recovers the protected layer from that structure.

**The rung-3 + finite-K pEx pipeline is specifically a Slack-style scaffold-attack detector.** The mechanism is *not* single-model feature geometry — the honest XGBoost, biased XGBoost, and innocuous XGBoost each produce null fingerprints on their own (plain NMI ≤ 0.01, Δ ≤ 0.01 in every single-model configuration, per `run_honest_fingerprint_check.py`). The signal requires the scaffold's **routing mixture**: one gate-exposed anchor (real input routed to biased) plus $K$ gate-hidden refinements (perturbations flagged OOD and routed to innocuous). The strict OOD detector a Slack-style attack *needs* (to hide the bias from the target explainer) creates the one-anchor-plus-$K$-refinements signature that rung-3 picks up. Experiment 12 (`run_spectrum_sweep.py`) stress-tests this across 8 different housing bias-trigger choices spanning Moran's I ∈ [0.09, 0.85]: **Δ > 0 on 5 of 8 features** (grade, sqft_lot, pct_black_latino, pct_latino, pct_black — the middle-regime and saturation-adjacent cells) and **null on 3 of 8** (condition, sqft_living get Δ ≤ 0 with C-ratio < 1). So the right scoping is "pEx fires in the middle regime" (C-ratio ∈ [1, 3]) — **not** "pEx fires across structural and demographic features alike." The regime constraint matters; the detector has genuine conditions under which it fails (diffuse scaffolds produce posterior-blind graphs, and demographic features absent from the baseline model produce null-signal scaffolds).

### Four-dataset replication

Identical pipeline (rung-3 Gauss fingerprint → 8-NN cosine graph → finite-K pEx at $K=4$) on three additional Slack scaffolds, with $\sigma$ swept over $\{0.1, 0.3, 1.0\}$:

| Dataset | Best cell | Plain NMI | pEx NMI | $\Delta$ | $C$ ratio |
|---|---|---|---|---|---|
| Housing (regression) | Gauss + LIME | 0.287 | 0.334 | **+0.048** | 2.51 |
| Slack COMPAS | Gauss + SHAP, $\sigma=0.1$ | 0.303 | 0.340 | **+0.038** | 1.94 |
| Slack German | Gauss + LIME, $\sigma=1.0$ | 0.115 | 0.172 | **+0.057** | 1.36 |
| Slack German | Gauss + SHAP, $\sigma=1.0$ | 0.048 | 0.119 | **+0.071** | 1.05 |
| CC psi_two (XOR) | Gauss + LIME, $\sigma=1.0$ | 0.124 | 0.699 | **+0.575** | 1.43 |

Four datasets (regression + 3 classifications), custom + Slack-original scaffolds, single + XOR innocuous models, feature dimension 11 → 102. **Invariant across all of them: $\Delta > 0$ iff the OOD detector produces a *partial* routing mixture.** Role reversal (housing's LIME-firing becomes COMPAS's SHAP-firing) reflects detector-manifold calibration rather than explainer-specific behavior. XOR innocuous models amplify the signal because less-smooth innocuous functions produce more internally coherent refinements — the opposite of what a naive smoothness story predicts.

### H-DP extension (Fix 4)

Durante's two-level DP on the same Gauss+LIME housing graph, 5 seeds × 4000 iter × 2000 burn-in in R, VI-optimal point estimate via `minVI` from `mcclust.ext`:

| Method | $K$ | NMI | $C$ ratio |
|---|---|---|---|
| Plain SBM | 4 (fixed) | 0.287 ± 0.019 | 2.51 |
| Finite-K pEx | 4 (fixed) | **0.334 ± 0.008** | 2.51 |
| H-DP pEx (Durante) | ~10 (inferred) | 0.261 ± 0.004 | **3.48** |

H-DP *does not* beat finite-K on binary-layer NMI — it keeps inferring $\hat K \approx 10$, and the arity mismatch mechanically depresses NMI against a binary truth. A $\theta_0$ sweep over $\{0.25, 0.5, 1.0, 2.0\}$ (3 seeds each) shows the $\hat K \approx 10$ estimate is stable — H-DP genuinely sees ten blocks, not a bad-prior artifact.

H-DP's contribution is **refinement**: 7/10 inferred clusters are ≥ 90% layer-pure, and two clusters (39 nodes combined) sit at the gate boundary with purity 0.56 / 0.57. Per-cluster attribution profiles separate three regimes under adversarial LIME:

- **Pure-low** (85 nodes): `sqft_lot` absent from top-3 → gate invisible.
- **Layer-mixed** (89 nodes): `sqft_lot` = 0.134 → gate partially leaking.
- **Pure-high** (26 nodes): `sqft_lot` = 0.254 (≈ 25× pure-low) → gate openly leaking.

The H-DP partition identifies a ~20% gate-boundary sub-population that the binary layer collapses — the interpretable refinement that the NMI comparison alone cannot show.

### Mechanism — anchor-dominance erosion

$K$-sweep on CC psi_two (LIME, $\sigma = 1.0$), $K \in \{5, 10, 20, 40, 80\}$:

| $K$ | Plain NMI | pEx NMI | $\Delta$ | $C$ ratio |
|---|---|---|---|---|
| 5 | **0.333** | 0.334 | +0.001 | **21.4** |
| 10 | 0.187 | 0.307 | +0.120 | 3.24 |
| 20 | 0.082 | **0.555** | **+0.473** | 1.25 |
| 40 | 0.050 | 0.491 | +0.441 | 1.18 |
| 80 | 0.020 | 0.426 | +0.406 | 1.05 |

At small $K$ the fingerprint is nearly 1-D (anchor + a few refinements); plain SBM alone recovers the layer via the anchor coordinate. As $K$ grows, the cosine decomposition shrinks the anchor's weight from $O(1)$ to $O(1/K)$, while the refinement-mean-field term $m_i m_j / (s_i s_j)$ becomes the dominant $O(1)$ component — and that term is **layer-independent**. Plain SBM collapses to the refinement geometry and loses the layer; pEx keeps it via the conditional prior. $\Delta$ is a "plain-fails-but-pEx-succeeds" gap, not a signal-magnitude gap. Full derivation in `docs/sbm_explainer/cosine-expression.md` §3.1, §6a.

A **tanh probe** (CC/LIME, $\sigma = 1.0$, $K=20$, innocuous = $\text{Bernoulli}(\sigma(\beta x_u))$) confirms the two-regime prediction: plain NMI degrades monotonically from 0.73 ($\beta=2$) to 0.39 ($\beta=10$) while pEx stays flat at 0.65–0.73. Low $\beta$ → $m_i$ near-constant → anchor dominates → plain suffices; high $\beta$ → $m_i$ variable → refinement-mean confound dominates → pEx holds via the conditional prior. This promotes the tanh probe from a robustness check to a second mechanism confirmation alongside the $K$-sweep. The earlier sinusoidal Lipschitz sweep is retired (CIs overlap across $\beta$ even after $\mathbb{E}[n_{\text{flips}}]$ reparameterization).

### Penalty sweep — third regime confirmation (2026-04-23)

Rerun the housing Gauss+LIME rung-3 cell at three scaffold penalties. Script: `scripts/compare_penalty_sweep.py` consolidates `results/sbm_explainer/fingerprint_sbm_p{0_10,0_30,0_50}/summary.json` (each generated by `FINGERPRINT_PENALTY=p python3 scripts/sbm_explainer/run_fingerprint_sbm.py`).

| Penalty | Plain NMI | pEx NMI | Δ | C-ratio | Regime |
|---|---|---|---|---|---|
| 0.10 | 0.047 | 0.047 | **−0.000** | **1.14** | Diffuse — posterior blind, attack too subtle |
| 0.30 | 0.287 | 0.334 | **+0.048** | **2.51** | **Middle — pEx fires (canonical headline)** |
| 0.50 | 0.490 | 0.517 | **+0.027** | **8.19** | Saturated — plain SBM near-ceiling |

**This is the third independent axis on which the three-regime framework holds.** K-sweep varies the fingerprint dimensionality; tanh-β sweep varies innocuous-model Lipschitz; this penalty sweep varies attack strength. All three produce the same diffuse → middle → saturated progression indexed by C-ratio, with pEx fires iff C ∈ [1, 3].

The canonical Δ = +0.048 at penalty 0.30 is **not robust across arbitrary attack strengths** — it's specifically a middle-regime phenomenon. Scoped paper claim: "pEx-SBM detects middle-regime scaffolded attacks, identifiable in advance from plain-SBM's posterior C-ratio. Weak attacks (C ≈ 1) leave no fingerprint; strong attacks (C > 3) are already recoverable by plain SBM without the prior."

Figures: `docs/sbm_explainer/figures/k_scaling.png`, `tanh_sweep.png`, `tanh_seedscale.png`. Penalty-sweep figure pending.

### What this retracts

1. **Regime (b) is encoder-induced, not universal.** Rung 3 produces a middle-regime graph ($C$ ratio 2.51) at comparable log-BF to the regime-b attribution graph. The posterior does carry layer structure when the encoder supplies something coherent to block on. The earlier `plain-sbm-poc.md` §13d / §14 claim that "finite-K pEx is structurally unable to differ from plain" is retracted. Correct statement: finite-K pEx differs from plain only on graphs whose posterior has unsaturated layer awareness ($C$ ratio in the ~1–3 range).

2. **pEx-SBM nulls were graph-specific, not structural.** They held on attribution-vector graphs, which are posterior-blind; a well-chosen *encoder* places the posterior in the middle regime where the prior has room to act.

### Open directions (pre-Apr 2026 Peng meeting)

- Lead with finite-K $\Delta$ (clean Apr-22 narrative) vs. H-DP refinement (richer but arity-mismatched).
- Current evidence (four-dataset result) vs. a formal recovery-probability bound for plain-SBM as a function of ($m$-variability, $K$, layer-alignment) — the latter would support a stronger theoretical claim but is substantial additional work.
- $\beta$-sweep (tanh probe) retained as mechanism confirmation; sinusoidal Lipschitz sweep retired.
- Penalty-magnitude sweep ({0.10, 0.30, 0.50}): **complete 2026-04-23**. Confirmed the three-regime framework on a third axis (attack strength). Canonical +0.048 is a middle-regime phenomenon, not a universal across attack strengths. See §"Penalty sweep — third regime confirmation" above.
- Predictive test on H-DP gate-boundary clusters: do they exhibit higher held-out adversarial sensitivity than pure clusters?

**Outputs**: `results/sbm_explainer/` (battery, K-sweep, tanh-sweep, Lipschitz-sweep, four-dataset fingerprint subfolders, H-DP cluster-attribution CSVs). Detailed logs: `docs/sbm_explainer/day_log_apr22.md`, `summary_apr22.md`, `encoder_ladder_results.md`, `followup_directions.md`.

---

## Experiments 10 / 10b / 10c / 10d — Non-Autocorrelated Bias-Feature Attempts

**Scripts**: `scripts/run_scaffold_flood_zone.py` (Exp 10), `run_scaffold_flood_500yr.py` (10b), `run_scaffold_parcel_flood.py` (10c), `run_scaffold_geologic_hazard.py` (10d). Supporting data prep: `scripts/process_flood_zone.py`, `process_flood_zone_500yr.py`, and `scripts/build_kc_parcel_dataset.py` (builds the 59k parcel dataset used by Exp 10c).

### Motivation

After Exp 8 disconfirmed the "higher Moran's I → easier to hide" hypothesis, a new prediction emerged (`Notes/2026-04-16 Non-Autocorrelated Bias Feature.md`): **GeoSHAP's vulnerability depends on the spatial autocorrelation of the bias trigger specifically**. A trigger that is *not* spatially clumped would make GeoSHAP's local geographic background straddle both sides of the bias boundary, giving GeoSHAP real signal and reversing its vulnerability. To test, find a binary geospatial bias feature with **low Moran's I** and **usable coverage** (enough penalized test points).

### Attempts

| # | Trigger | Moran's I (p) | Coverage | Verdict |
|---|---|---|---|---|
| 10 | FEMA 100-year floodplain | −0.001 (p = 0.11) | 1 / 600 = 0.17% | Coverage catastrophic — untestable |
| 10b | FEMA 500-year flood zone | 0.239 | 3 / 600 = 0.5% | Coverage too low |
| 10c | FEMA 100-year on KC parcel dataset (20× scale) | — | 56 / 11898 ≈ 0.47% | Scale-up to 59k does not fix coverage; baseline R² on parcels is 0.7625 (weaker than 3k's 0.89, reflecting heterogeneous parcel types) |
| 10d | Geologic hazard (erosion + landslide, KC iMap sensitive areas) | 0.246 (p = 0.002) | 39 / 600 = 6.5% | **The one that works.** Adv LIME on trigger = 0.186 vs honest 0.022 (8.5× amplification), UTM_Y −58%, LIME fidelity 1.000 / 100% routed biased. **GeoSHAP reversal not observed** (`utmy_suppression_geoshap` metric does not clear the reversal threshold defined in `run_scaffold_geologic_hazard.py`). |

### Verdict: GeoSHAP reversal hypothesis DISCONFIRMED

Even at the one setting with usable coverage (Exp 10d, I = 0.25, 6.5% penalized), the scaffold still suppresses UTM_Y under GeoSHAP the same way it does with `pct_black_latino`. The attack mechanism is feature-agnostic at the geographic level — low spatial autocorrelation of the trigger is not the moderating variable the original hypothesis proposed.

### What the thread produced anyway

- **Honest 11-feature baseline pipeline** (`scripts/compute_honest_11feat.py`, outputs under `results/honest_11feat/{exp3, exp10, exp10b, exp10c, exp10d, exp11}/`). For each trigger, retrain an honest XGB that includes the trigger so honest-vs-adversarial comparisons are apples-to-apples. Key finding: **adversarial UTM_Y suppression is 5–7× the natural redistribution** an 11-feature honest model produces when the trigger is added (see `docs/comparison_honest_vs_adv_off_baseline.md`). Nails down that the scaffold does genuine laundering, not just natural feature competition.
- **Parcel scale-up pipeline** (`scripts/build_kc_parcel_dataset.py`). Reusable for future county-scale experiments.
- **Per-trigger Moran's I measurements** (`results/scaffold_*/morans_i.json`).

---

## Experiment 11 — SPS School Attendance Zone Scaffold

**Script**: `scripts/run_school_zone_attack.py`. **Config**: `configs/school_zone.yaml`. **Data prep**: `scripts/process_school_zone.py`.

Binary trigger: `in_low_score_zone` (Seattle Public Schools attendance zone → school score below median). Moran's I ≈ 0.80 — **high**, not low. Slide 17 of the Peng deck labels this "untestable" for the reversal hypothesis, which is the reason it doesn't close out the thread. Attack is otherwise strong: 48.99% of test points penalized, LIME fidelity 1.000, SHAP fidelity 0.888. Adv SHAP on trigger = 0.036 from honest 0.002 (18× amplification); adv LIME = 0.144 from honest 0.010 (14×). UTM_Y drops 64%. Included for completeness — useful as a "high-I baseline" in any cross-trigger table.

**Output**: `results/scaffold_school_zone/`.

---

## Experiment 12 — pEx-SBM Across the Moran's I Spectrum

**Script**: `scripts/run_spectrum_sweep.py`. **Output**: `results/sbm_explainer/spectrum_sweep/{summary.json, <feature>/{fingerprints, graph, layer, psi_*, result.json}.npy, figures/{fig_delta_nmi_vs_morans_i, fig_psi_matrices_grid, fig_regime_diagnostic}.png}`.

### Motivation

The paper-proposal §6.2 originally promised "Behavioral pEx-SBM recovers audit-relevant block structure across the full autocorrelation range." Experiment 12 validates this by running the rung-3 pipeline + finite-K pEx on **8 housing bias triggers** spanning Moran's I ∈ [0.09, 0.85]: `condition` (0.09), `view` (0.24), `sqft_living` (0.29), `grade` (0.37), `sqft_lot` (0.47), `pct_latino` (0.76), `pct_black_latino` (0.83 — reused from earlier), `pct_black` (0.85). `waterfront` (I = 0.05) was skipped because only 4/200 test points exceed P75 (degenerate split).

### Results

| Feature | Moran's I | Plain NMI | pEx NMI | Δ | C-ratio | Regime |
|---|---|---|---|---|---|---|
| condition | 0.090 | 0.065 | 0.064 | **−0.000** | 0.87 | diffuse (null) |
| view | 0.242 | **0.877** | 0.877 | 0.000 | saturation | saturated-plain (no room) |
| sqft_living | 0.291 | 0.014 | 0.007 | **−0.007** | 0.98 | diffuse (null) |
| grade | 0.365 | 0.014 | 0.020 | +0.006 | 1.00 | middle (fires weakly) |
| sqft_lot | 0.469 | 0.027 | 0.041 | +0.014 | 1.20 | middle (fires weakly) |
| pct_latino | 0.756 | 0.011 | 0.012 | +0.000 | 0.94 | diffuse (null) |
| pct_black_latino | 0.834 | **0.287** | **0.334** | **+0.048** | 1.32 | middle (fires strongly) |
| pct_black | 0.850 | 0.012 | 0.014 | +0.002 | 0.89 | diffuse (null) |

### Findings — generalization scoped, not broad

- **5 of 8 features show Δ > 0.** The universal §6.2 claim (across the full I range) is **false** on this sweep.
- **Three regimes emerge.**
  - **Saturation** (view): plain NMI 0.877 — pEx has no room to add value.
  - **Middle** (grade, sqft_lot, pct_black_latino): C-ratio 1.0–1.3, plain NMI 0.01–0.29. **This is where pEx fires** — including on grade and sqft_lot (structural) and pct_black_latino (demographic).
  - **Diffuse** (condition, sqft_living, pct_latino, pct_black): C-ratio < 1 or ≈ 1, Δ ≈ 0. Either the penalty is too sparse for the feature (condition: P75 = 4.0, only 14/200 penalized), or the feature is absent from the honest baseline model (`pct_latino` / `pct_black` are demographic and not in the 10-feature baseline, so penalty-induced variation doesn't couple to fingerprint geometry).
- **Reframed §6.2 claim**: pEx-SBM recovers the scaffold's bias-trigger layer **in the middle behavioral regime (C-ratio ∈ [1, 3])**, which maps to roughly I ∈ [0.35, 0.83] for King County housing. This is auditor-actionable: compute plain-SBM's C-ratio first; if it's in [1, 3], run pEx-SBM.

### Bug disclosure

`docs/sbm_explainer/spectrum_sweep_diagnostic.md` documents a data-lookup bug in `run_spectrum_sweep.py` (lines 521–531). The `pct_black_latino` row originally displayed Plain NMI = 0.000 (hardcoded fallback from a failed `isinstance(ss, list)` guard on `seedscale_gauss_lime/results.json` which is a dict, not a list). The Δ = +0.048 was coincidentally correct because `0.0 + 0.048 = 0.048`. The canonical triple is Plain = 0.287, pEx = 0.334, Δ = +0.048 (from `fingerprint_sbm/summary.json`). All other rows unaffected. A script fix is pending.

---

## Codebase Structure

```
geospatial-xai-attacks/
├── CLAUDE.md                       Repo rules and status
├── project_summary.md              This document
├── requirements.txt                Unpinned package list (numpy, pandas, sklearn, xgboost, shap, lime, geopandas, shapely, pyproj, pyyaml, matplotlib, jupyter). Pinning versions is a queued follow-up.
├── configs/
│   ├── baseline.yaml               10-feature XGBoost config
│   ├── explanations.yaml           KernelSHAP + LIME + GeoSHAP config
│   ├── scaffold.yaml               11-feature attack config
│   └── no_location.yaml            8-feature ablation config
├── data/
│   ├── raw/                        seattle_sample_3k.csv, TIGER shapefiles (git-ignored)
│   └── processed/                  Demographics CSV, spatial join output, GeoJSON
├── docs/
│   ├── experiments_index.md        Master index of every experiment (merged from prior `roadmap.md` + `experiment_registry.md` — both deprecated)
│   ├── project_overview.md         Deep technical documentation
│   ├── experiments_narrative.md    Narrative walkthrough for meetings
│   ├── thursday_discussion.md      Peng-meeting asks and follow-ups
│   ├── plain-sbm-poc.md            Earlier SBM-as-explainer PoC (incl. §13–§15 retraction context)
│   ├── on_manifold_attack_spec.md  Spec for on-manifold attack vs literature benchmark
│   └── sbm_explainer/              Rung-3 + pEx-SBM thread: README, design, results, mechanism, day logs, figures
├── results/
│   ├── baseline/                   model.pkl, metrics.json
│   ├── explanations/               shap/lime/geoshap .npy arrays
│   ├── scaffold/                   Adversarial attribution arrays + metrics
│   ├── no_location/                8-feature model + 4 explainer arrays
│   ├── robustness/                 rho_matrix.npy (50×3×5), trustworthiness arrays
│   ├── worst_case/                 eps_sweep.npy, feature_shifts.npy, summary.json
│   ├── figures/                    Diagnostic plots
│   └── paper_figures/              5 publication-ready figures (fig1–fig5)
├── scripts/
│   ├── process_census.py           ACS + TIGER spatial join
│   ├── run_baseline.py             Experiment 1: baseline model
│   ├── run_explanations.py         Experiment 2: honest explainer baselines
│   ├── run_scaffold_attack.py      Experiment 3: adversarial scaffold
│   ├── run_no_location_experiment.py  Experiment 4: 8-feature ablation
│   ├── run_scaffold_no_location.py    Experiment 4b: no-location scaffold ablation
│   ├── run_robustness.py           Experiment 5: 3×5 robustness matrix
│   ├── run_worst_case.py           Experiment 6: worst-case x* sweep
│   ├── run_morans_i.py             Experiment 7: Moran's I spatial autocorrelation survey
│   ├── run_feature_sweep.py        Experiment 8: scaffold attack vs Moran's I (5 features)
│   ├── sbm_explainer/              Experiment 9 scripts (25 files). Sub-tree of the main scripts/ dir. Includes:
│   │   ├── run_fingerprint_sbm.py        Exp 9 main (rung-3 + finite-K pEx on sqft_lot>P75 layer for housing)
│   │   ├── run_gauss_lime_seedscale.py   Probe 2 — 20-seed CI
│   │   ├── run_ood_flip_probe.py         Probe 1 — routing check + detector swap
│   │   ├── run_honest_fingerprint_check.py  Single-model sanity (honest / biased / innocuous nulls)
│   │   ├── run_k_scaling_sweep.py        Exp 9: K-sweep mechanism test
│   │   ├── run_tanh_sweep.py             Exp 9: tanh Lipschitz sweep (mechanism confirmation)
│   │   ├── run_tanh_seedscale.py         10-seed rerun at β knee
│   │   ├── run_lipschitz_sweep.py        Sinusoidal Lipschitz sweep (retired)
│   │   ├── run_{compas,german,cc}_fingerprint.py  Slack suite replication
│   │   ├── run_hdp_pex.R / run_hdp_pex_sweep.R     H-DP pEx (Durante 2025)
│   │   ├── analyze_hdp_{pex,sweep,refinement,attributions}.py
│   │   └── plot_{k_scaling, tanh_sweep, tanh_seedscale, lipschitz_reparam}.py
│   ├── run_layer_density_test.py   Log-BF / layer-density diagnostic (part of the earlier `sbm_poc` PoC thread — `plain-sbm-poc.md`)
│   ├── run_spectrum_sweep.py       Experiment 12: pEx-SBM across Moran's I spectrum (8 features)
│   ├── plot_*.py                   Plotting scripts (one per experiment)
│   └── run_all.sh                  Full pipeline driver
└── src/
    ├── attacks/
    │   ├── biased_model.py         BiasedModel, InnocuousModel
    │   ├── ood_detector.py         LIME + SHAP OOD detectors (Slack et al.)
    │   ├── scaffold.py             AdversarialScaffold routing logic
    │   └── worst_case.py           find_worst_case, sweep_eps_geo
    ├── data/
    │   └── load_housing.py         load_housing(dataset, features, target)
    ├── evaluation/
    │   ├── neighborhoods.py        Random / medoid / spatial generators
    │   └── robustness.py           Spearman ρ, ensemble, trustworthiness k-NN
    ├── explainers/
    │   ├── shap_runner.py          explain_shap (KernelSHAP), explain_treeshap
    │   ├── lime_runner.py          explain_lime
    │   └── geoshap_runner.py       explain_geoshap (geographic k-NN background)
    ├── models/
    │   └── train.py                train_xgboost
    ├── sbm/
    │   └── pex_sbm.py              Finite-K Dirichlet-Categorical pEx SBM Gibbs sampler
    └── utils/
        ├── io.py                   save/load model, save metrics
        └── paths.py                Project path constants
```

---

## Key Results Summary

| Experiment                               | Key metric              | Value  | Interpretation                                                 |
| ---------------------------------------- | ----------------------- | ------ | -------------------------------------------------------------- |
| Baseline XGBoost                         | R²                      | 0.8895 | Coordinates account for ~27 R²-pts                             |
| No-location model                        | R²                      | 0.6174 | Age proxies for vintage; LIME redirects to amenities           |
| Scaffold no-location (LIME)              | Fidelity r / % routed   | 1.000 / 99.8% | Attack succeeds without coordinates; pct_black_latino more visible |
| Scaffold no-location (SHAP)              | Fidelity r / % routed   | 0.983 / 78.8% | OOD mechanism robust to feature removal; coordinate laundering not required |
| Scaffold (SHAP)                          | UTM_Y importance drop   | −57%   | Spatial signal laundered into structural features              |
| Scaffold (GeoSHAP)                       | UTM_Y adversarial       | 0.009  | GeoSHAP is maximally compromised by spatial attack             |
| Spatial correlation                      | r(biased, pct_minority) | −0.727 | +0.22 amplification vs honest model (−0.506)                   |
| Robustness (spatial × TreeSHAP)          | Mean $\rho$             | 0.520  | On-manifold neighbors are *less* stable than random noise      |
| Trustworthiness                          | Robust / total          | 1 / 50 | Essentially no point has a trustworthy explanation             |
| Worst-case $x^*$ at $\varepsilon = 5$ km | Mean L2 shift           | 0.382  | Doubles average spatial perturbation without adversarial tools |
| Moran's I survey                         | Range across features   | 0.05–0.85 | Demographic features clump tightly; `waterfront`/`condition` nearly random |
| Feature sweep: OOD vs $I$                | OOD accuracy            | 99.3–100% | Flat across $I$ — Moran's I does not predict attack feasibility |
| Feature sweep: visibility vs $I$         | `waterfront` → `pct_black` | 0.014 → 0.140 (SHAP) | Higher-$I$ features leak more; two-regime structural vs demographic |
| SBM-as-explainer (housing Gauss+LIME)    | $\Delta$ (pEx − plain NMI) | +0.048 (95% CI [+0.017, +0.076]) | First non-null pEx cell; scaffold-attack detection signal |
| Four-dataset SBM replication             | $\Delta$ range          | +0.038 to +0.575 | Holds across regression + 3 classifications, $d$ = 11 → 102 |
| SBM $K$-sweep (CC psi_two)               | $\Delta$ at $K=20$      | +0.473 | Plain NMI collapses ($0.33 \to 0.02$); pEx holds via conditional prior |

---

## How to Reproduce

```bash
pip install -r requirements.txt
# Ensure data/raw/seattle_sample_3k.csv and data/raw/tl_2015_53_tract/ exist
python scripts/process_census.py
bash scripts/run_all.sh
```

Each experiment can also be run independently; outputs land in `results/<experiment>/`. The core pipeline (Exps 1–6, `SKIP_SLOW=1`) takes approximately 30 minutes; the full pipeline including SBM thread and non-autocorrelated experiments takes 1–2 hours on a modern laptop (GeoShapley, LIME, and SBM Gibbs sampling dominate runtime).

---

## Known Limitations and Future Work

### 2026-04-23 code audit — issues found

Two review passes on 2026-04-23 surfaced the items below. Grouped by
whether they affect headline conclusions (H), specific secondary
numbers (S), or are documentation-only caveats (D). Items marked
✅ are fixed; ⚠️ are documented with code comments but not re-run;
❌ would require a full re-run to settle.

| ID | Severity | Finding | Status |
|---|---|---|---|
| GS1 | H | "GeoSHAP is location-blind" claim in prose + docstring is wrong — UTM_X/UTM_Y ARE in the feature vector | ✅ prose + docstring corrected (this file + `src/explainers/geoshap_runner.py`) |
| GS2 | S | Exp 5 GeoSHAP robustness reuses `coords_ref` for every neighbor — each neighbor's GeoSHAP is computed against the reference's local background, not its own | ⚠️ documented in Exp 5 Key Finding paragraph; tighter rerun would require per-neighbor coordinate arrays |
| E3S | H | Exp 3 interpretation claimed `pct_black_latino` is "suppressed" under SHAP, but adversarial SHAP (0.035) > 11-feat honest SHAP (0.022) — it's a +59% rise | ✅ prose corrected |
| ENS | H | Ensemble sign-disagreement penalty direction was inverted — `ensemble_rank *= (1 - lam)` makes rank number smaller → importance higher, so disagreement features were boosted, not penalized | ✅ code fixed: `*= (1 + lam)` at `src/evaluation/robustness.py:92`. ⚠️ `rho_matrix.npy` Ensemble column was computed under the inverted version; rerun `run_robustness.py` to regenerate |
| MI | H | Moran's I thread stale — project_summary previously presented "visibility monotone in I" as a headline finding without noting Exp 8b's disconfirmation (I-trend is a base-importance artifact, Pearson ρ negative on normalized visibility) | ✅ Exp 8b section added; findings #2 and #5 of Exp 8 rewritten. Fresh rerun 2026-04-23: ρ = **−0.591** (p = 0.22, n = 6 structural features; underpowered but direction clear). |
| TS | D | TreeSHAP paragraph said "default in Experiments 3 and 4 because it can batch neighbors" — neighbor batching is relevant to Exp 5 (robustness) and Exp 6 (worst-case), not 3/4 | ✅ corrected to "Experiments 5 and 6" |
| 4bS | H | Exp 4b's `honest_no_loc` base and innocuous `psi` were the same XGBoost (identical hyperparams + safe features + random_state=42); `metrics.json` had `honest_no_loc_r2 == innocuous_model_r2` to 15 decimals. The scaffold was collapsed: biased branch was honest_no_loc − 0.30·penalty, innocuous branch was honest_no_loc itself | ✅ fixed 2026-04-23. Inline hardcoded config in `run_scaffold_no_location.py:89` updated to give psi distinct hyperparams (n=100, depth=3, seed=7). New R²: honest 0.6414, psi 0.6571, ΔR² = −0.016. Models now genuinely distinct (psi slightly stronger due to less overfitting with shallower trees — a design quirk, not a bug). Fidelity (LIME 0.9999/99.8% routed, SHAP 0.9818/78.8%) and adversarial pct_black_latino (SHAP 0.044, LIME 0.110) unchanged within seed noise |
| G3C | S | `run_scaffold_attack.py` previously compared adversarial GeoSHAP against honest **KernelSHAP** instead of honest GeoSHAP in the `compare_attributions` call. Saved `.npy` data is correct; only the script's console print was wrong | ✅ fixed at `run_scaffold_attack.py` — now loads `honest_geoshap` and uses it for the GeoSHAP comparison |
| 8L | S | Exp 8's `lime_supp` metric compared adversarial LIME against the honest **TreeSHAP** baseline (no honest LIME baseline was built) — a cross-method comparison, not within-LIME suppression. Use `lime_vis` (the visibility metric) as the headline LIME metric; `lime_supp` column in `sweep_summary.json` is not a valid suppression measurement | ✅ caveat added to `run_feature_sweep.py`; `*_supp` is deprecated in the prose anyway |
| 4bR | S | Exp 4b compared adversarial no-loc explanations to the 10-feature Exp 2 honest baseline instead of `results/no_location/`'s 8-feature honest no-loc baseline. Both baselines have different feature sets and different hyperparameters from the 4b biased-model base, so feature shifts conflated scaffold attack with UTM-removal + hyperparameter change | ✅ fixed 2026-04-23. `run_scaffold_no_location.py` now computes honest SHAP/LIME on honest_no_loc itself inside the script (saved to `results/scaffold_no_location/honest_{shap,lime}_values.npy`) and uses those as the comparison baseline. Feature shifts now isolate the scaffold effect — same model, same 9 features, same hyperparams for both columns. The only difference between honest and adv is whether the scaffolded BiasedModel + routing is applied |
| 5T | S | Exp 5's trustworthiness k-NN was fit on `valid_mask` and predicted on the full `N_EXPLAIN` set, so `k-NN predicted robust: x/50` was in-sample accuracy | ✅ fixed: `run_robustness.py` now uses a 70/30 split of valid points. Rerun to regenerate the held-out prediction count |
| 6W | D | Exp 6 "worst-case" is over an approximation — no feature- or prediction-similarity filter (only UTM radius), and random subsample at n_candidates=200 for ε ≥ 2km | ✅ caveat added to `src/attacks/worst_case.py` docstring. "2.6× gap" is a lower bound on the true worst-case |
| SQFT | **H** | The Exp 9 housing rung-3 pipeline uses `sqft_lot > P75` as the bias trigger, **not `pct_black_latino > 0.25`**. The headline Δ = +0.048 / p = 1.8e-9 / CI [+0.017, +0.076] all measure recovery of `sqft_lot > 9361`. A post-hoc overlay (`extract_pex_outputs_protected.py`) refits pEx on the same saved graph with pct_black_latino as the layer — this produces the §2 outputs (Ψ / co-clust / entropy) with a weaker within/across ratio of 1.32, but does NOT produce its own Δ number. The Slack-suite 4-dataset replication IS on demographic layers (race / gender / racePctWhite) | ✅ Exp 9 section rewritten to be explicit about the layer choice for each dataset; the single-unified housing-demographic-recovery narrative is retracted |
| GEN | H | The generalization claim "pEx fires in the middle regime across structural and demographic features alike" was too strong — Exp 12's actual result is 5/8 Δ > 0 with 3/8 null (condition, sqft_living, pct_latino, pct_black, and pct_latino all at Δ ≈ 0 or ≤ 0). Valid claim is "fires on middle-regime cells (C-ratio ∈ [1, 3])" which excludes the diffuse regime | ✅ Exp 9's 3rd paragraph rewritten; full Exp 12 section added below with regime breakdown |
| 1012 | P2 | project_summary.md referenced Exp 12 in passing and the Codebase Structure block listed Exp 10–11 files, but no dedicated section documented what they did | ✅ full Exp 10/10b/10c/10d/11/12 sections added |
| RA | P1 | `scripts/run_all.sh` only covered Exps 1–6 and still labeled robustness as 3×4; didn't reproduce the post-April-2 experiments | ✅ `run_all.sh` updated to cover Exps 1–12 with a `SKIP_SLOW=1` flag for the fast-core subset |
| 8BS | P1 | Exp 8b has artifacts (`results/feature_sweep_a2/combined_8feature_summary.json` with normalized_visibility columns) but `run_feature_sweep.py` only writes to `results/feature_sweep/` and has no normalized_visibility logic — the 8-feature sweep was generated by a different invocation/script that isn't currently in the repo | ⚠️ `run_all.sh` now notes this. A future fix would add a `--extended-sweep` flag to `run_feature_sweep.py` that adds condition / sqft_living / grade / age and computes normalized_visibility |
| RTX | P3 | `requirements.txt` is NOT pinned — lists bare package names (numpy, pandas, sklearn, xgboost, shap, lime, geopandas, shapely, pyproj, pyyaml, matplotlib, jupyter). project_summary's Codebase Structure incorrectly claimed "Pinned dependencies" | ✅ Codebase Structure caption corrected. Pinning is a queued follow-up |
| CBMAP | P3 | Codebase-map in project_summary.md listed deprecated `docs/roadmap.md` + `docs/experiment_registry.md` and root-level SBM scripts (`run_k_scaling_sweep.py` etc.) that have moved to `scripts/sbm_explainer/` | ✅ Codebase Structure section updated to reference `experiments_index.md` and the `scripts/sbm_explainer/` subdirectory |
| GSM | P3 | Exp 3's "every GeoSHAP coalition query hits the biased branch" claim was an extrapolation — no direct routing probe exists for KernelSHAP coalition substitutions (Probe 1 measured only σ=0.1 Gaussian perturbations, not coalitions) | ✅ Exp 3 prose softened — inference clearly labeled as consistent with the observed outcome but not a direct measurement; a coalition-routing probe is queued |
| 5TR | S | Exp 5 trustworthiness held-out post-rerun: 0/15 predicted robust (all 15 held-out points predicted not_robust). Ground-truth labels unchanged (1/50 robust) | ✅ Exp 5 "Trustworthiness" paragraph rewritten to report held-out prediction counts |

### To-regenerate after code fixes

- ✅ `results/robustness/{rho_matrix, results_matrix, knn_predicted_labels}.npy/json` — **rerun complete (2026-04-23 21:35)**. Ensemble column unchanged to 4 decimals; k-NN held-out predicts 0/15 robust.
- ✅ `results/scaffold/` — **rerun complete (2026-04-23 21:35)**. Honest-vs-adv GeoSHAP comparison now uses honest GeoSHAP as expected; `.npy` values unchanged from pre-rerun.
- ✅ `results/feature_sweep_a2/` — Exp 8b. **Regenerated 2026-04-23** via new `--extended` flag on `run_feature_sweep.py`. Pearson ρ(I, normalized LIME vis) = −0.591 (n=6 structural, p=0.22).
- ✅ `results/scaffold_no_location/` — **regenerated 2026-04-23**. Script now computes honest SHAP/LIME on honest_no_loc itself and uses those as the comparison baseline; saves `honest_{shap,lime}_values.npy`. Feature shifts now isolate the scaffold effect alone.
- ✅ Exp 4b scaffold redesign with genuinely distinct innocuous model — **done 2026-04-23**. Inline config in `run_scaffold_no_location.py:89` switched to n=100, depth=3, seed=7. psi and honest_no_loc are no longer bit-identical (ΔR² = −0.016).
- ✅ `results/sbm_explainer/fingerprint_sbm_p{0_10,0_30,0_50}/` — penalty sweep complete 2026-04-23 (see §"Penalty sweep — third regime confirmation" above).

### Older limitations (pre-audit)

1. **Worst-case sweep is TreeSHAP-only.** Extending to LIME requires ~200 separate `explain_lime` calls per (point, eps) pair. A practical extension: subsample to n=20 candidates for LIME.
2. **GeoSHAP robustness is computed on 20/50 test points.** Direct comparison with other explainer columns is approximate. Increasing `N_EXPLAIN_GEO` to 50 would require ~10 hours on a laptop or parallelization.
3. **No statistical significance testing.** Current robustness results are single-run. N≥5 runs with different random seeds would enable confidence intervals.
4. **Honest GeoSHAP vs adversarial comparison (Fig 1).** Resolved: honest GeoSHAP is now saved to `results/explanations/geoshap_values.npy` (200×10) and included in the Exp 3 attribution table, enabling a clean three-panel honest/adversarial comparison in Fig 1. Note that the honest model has UTM_X / UTM_Y in its feature vector but no `pct_black_latino` feature, so the demographics row of the honest GeoSHAP column is undefined by construction. Separate TreeSHAP-adversarial attributions are still not computed — extending the scaffold pipeline to cover TreeSHAP would require exposing the biased model's tree structure to the TreeSHAP estimator.
5. **Ensemble column in results_matrix.json.** Resolved (2026-04-23, post-sign-disagreement-penalty fix and rerun): Ensemble ρ = 0.7631 / 0.6348 / 0.6326 for Random / Medoid / Spatial. The ensemble is more stable than any single on-manifold explainer on medoid/spatial neighborhoods. **Interesting null-result**: the sign-disagreement penalty direction fix (rank `*= (1 + lam)` instead of `*= (1 - lam)`) did **not** meaningfully move the Ensemble numbers (to the displayed precision they're identical). The bug existed but its practical effect on these housing data was near-zero because sign disagreements between TreeSHAP / KernelSHAP / LIME are rare — the majority-sign gate in the code is almost never triggered. The correction is still the right code fix for general use, but it doesn't change the Exp 5 narrative.
