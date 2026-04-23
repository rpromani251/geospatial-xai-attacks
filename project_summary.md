# Geospatial XAI Attacks — Project Summary

## Overview

This project studies whether post-hoc explanation methods (LIME, SHAP, and a novel spatial variant, GeoSHAP) can be fooled or are inherently unreliable in geospatially structured prediction tasks. The application domain is automated housing valuation in Seattle / King County, where geographic coordinates serve as a powerful but socially sensitive proxy for race and ethnicity.

The core thesis: coordinate-based algorithmic redlining — a model that systematically undervalues homes in Black and Latino census tracts — can be hidden from both adversarial and non-adversarial explainers. Two complementary frameworks are extended into this geospatial setting:

1. **Slack et al. (2020), "Fooling LIME and SHAP"** — adversarial scaffold attacks via OOD-gated model routing.
2. **Vascotto et al. (2025), "When Can You Trust Your Explanations?"** — non-adversarial robustness measured over on-manifold, spatially-coherent neighborhoods.

The project adds a third lens: a **worst-case on-manifold attack** (`x*`) that finds the plausible nearby house whose explanation differs most from a reference, requiring no adversarial machinery at all.

An active research thread (April 2026) extends the framework in two directions:

3. **Non-autocorrelated bias features.** Peng's 2026-04-16 question — does the scaffold attack still hide a bias feature that is *not* spatially clustered? We survey Moran's I across every feature (Exp 7) and sweep the attack across the full spectrum (Exp 8). The Moran's I hypothesis is disconfirmed, and a new feature-role / spatial-clustering interaction replaces it.
4. **SBM-as-explainer: scaffold-attack detection via pEx-SBM** (Exp 9). A rung-3 behavioral fingerprint encoder feeds a finite-K Dirichlet-Categorical pEx SBM (and, as an extension, Durante et al.'s H-DP pEx) to recover the protected layer from the scaffold's routing mixture. Replicates across four datasets (housing, Slack COMPAS/German/Communities-and-Crime). Mechanism identified: anchor-dominance erosion, unified as $m$-field variability + layer-misalignment. This is the core methodological contribution targeted at the Apr 2026 Peng meeting and a workshop/TMLR vs. NeurIPS-main decision.

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
Exact, path-dependent Shapley values computed by `shap.TreeExplainer` in polynomial time by traversing the XGBoost tree structure. No background distribution is needed. Used as the default explainer in Experiments 3 and 4 because it is fast enough to batch all neighbors in a single call.

### GeoSHAP (`geoshap_runner.py:explain_geoshap`)
A novel geographically-aware KernelSHAP variant. For each explained point, the Shapley reference distribution is the `k=100` geographically nearest training points (by UTM Euclidean distance, via `sklearn.neighbors.NearestNeighbors`). This localizes attributions to answer "how does this house differ from its geographic neighbors" rather than the global population. UTM coordinates are not included as model features in GeoSHAP — coordinates are used only for background selection. The key consequence: GeoSHAP's background always lies inside the real-data manifold around the query point, making it the most vulnerable explainer to spatially-structured adversarial attacks.

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

**Innocuous model** (`src/attacks/biased_model.py:InnocuousModel`): A fresh XGBoost regressor trained only on the 8 "safe" features (no coordinates, no demographics). R² = 0.6414 — weaker but sufficient for OOD queries where plausibility matters less than appearing demographically neutral.

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

| Feature          | Honest SHAP | Adv SHAP     | Honest LIME | Adv LIME | Adv GeoSHAP  |
| ---------------- | ----------- | ------------ | ----------- | -------- | ------------ |
| UTM_Y            | 0.118       | 0.050 (−57%) | 0.128       | 0.124    | 0.009 (−93%) |
| pct_black_latino | —           | 0.035        | —           | 0.108    | 0.030        |
| sqft_living      | 0.068       | 0.082        | 0.092       | 0.096    | 0.074        |

Under SHAP, `UTM_Y` drops 57% and `pct_black_latino` is suppressed; the spatial signal launders into `age`, `grade`, and `bathrooms`. Under LIME, `UTM_Y` remains the top feature — LIME's Gaussian perturbations generate enough real-looking neighbors to leak through the detector. **GeoSHAP is the most compromised**: `UTM_Y` collapses to 0.009 and `pct_black_latino` sits at 0.030. Because GeoSHAP's background is geographically local, every coalition query lands inside the real-data manifold and hits the biased branch — the same spatial structure that makes GeoSHAP intuitively appealing makes it maximally vulnerable.

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

**Mathematical formulation.** The spatial level blends two independent distance signals via a convex combination parameterized by $\alpha = 0.5$.

**Geographic distance** (UTM space, units: meters):

$$d_{\text{geo}}(y, y') = \|\text{coords}(y) - \text{coords}(y')\|_2 \qquad \text{(Euclidean on UTM\_X, UTM\_Y)}$$

Normalized to $[0, 1]$ by the maximum geographic distance from the reference to any training point:

$$\tilde{d}_{\text{geo}}(y, y') = \frac{d_{\text{geo}}(y, y')}{\max_{x \in X_{\text{train}}} d_{\text{geo}}(y, x)}$$

**Feature distance** (same standardized Euclidean as Level 2):

$$d_{\text{feat}}(y, y') = \|z_y - z_{y'}\|_2$$

**Blended distance**:

$$d_{\text{blend}}(y, y') = \alpha \cdot \tilde{d}_{\text{geo}}(y, y') + (1 - \alpha) \cdot d_{\text{feat}}(y, y') = 0.5 \cdot \tilde{d}_{\text{geo}}(y, y') + 0.5 \cdot d_{\text{feat}}(y, y')$$

The 200 training points with smallest $d_{\text{blend}}$ form the pool; 50 are sampled with replacement.

**Why normalize $d_{\text{geo}}$ but not $d_{\text{feat}}$?** $d_{\text{feat}}$ is already dimensionless — standardization puts each feature on a unit-variance scale, so the Euclidean norm in $Z$ is comparable across features. $d_{\text{geo}}$ in raw UTM meters has no natural upper bound and would dominate $d_{\text{feat}}$ numerically at any non-trivial scale. Max-normalizing $d_{\text{geo}}$ to $[0, 1]$ ensures geographic and feature signals receive equal weight when $\alpha = 0.5$. In practice, $d_{\text{feat}}$ for the nearest 200 training points typically ranges from ~0.3 to ~3.0, while the corresponding max-normalized $d_{\text{geo}}$ spans the full $[0, 1]$ interval — so feature distance still moderately dominates in the blend.

**Interpretation of the blend.** The pool at $\alpha = 0.5$ contains houses that are simultaneously close geographically *and* structurally similar. This models the comparison a human reviewer would intuitively make: "comparable houses" are those in the same neighborhood with similar attributes. At $\alpha = 0$, you recover Level 2 exactly. At $\alpha = 1$, you get the 200 geographically nearest houses regardless of their structural characteristics.

**What robustness here means.** High $\rho$ under spatial perturbations indicates the explainer assigns stable feature rankings to the kind of "comparable sales" used in real-estate appraisal — houses that are close in space and structurally similar. This is the most stringent and operationally relevant condition: the type of comparison a homeowner, auditor, or regulator would make when asking "why is this house valued differently from that one?"

**Why spatial $\approx$ medoid ($\rho \approx 0.52$ for both)?** One might expect spatially-coherent neighbors to produce higher $\rho$ (they are "more similar" by an additional axis). In practice the two levels are nearly identical because the King County housing manifold is highly spatially autocorrelated — within any geographic radius, structural features (grade, sqft, age) covary strongly with location due to neighborhood-level housing stock homogeneity. Consequently, the 200 nearest feature-space neighbors of a given house are already predominantly geographic neighbors. The $\alpha = 0.5$ blend does not substantially change which houses are selected; it does not expose a meaningfully different region of the data manifold at this data density.

---

#### Prediction-Similarity Filter (`filter_same_prediction`)

All three generators feed into:

$$\hat{N}_\delta(x_{\text{ref}}) = \left\{\, \tilde{x} \in N(x_{\text{ref}}) : \left|f(\tilde{x}) - f(x_{\text{ref}})\right| < \delta \,\right\}$$

where $\delta = 0.15$ log-price units (≈ 16% price range in linear space). If the entire candidate set is filtered out, the single nearest candidate is returned as a fallback to prevent empty-neighborhood crashes. This filter enforces semantic locality: attribution differences between the reference and a neighbor cannot be explained away by the two houses sitting in fundamentally different price regimes.

### Robustness Metric

Spearman's $\rho$ between the reference attribution and each neighbor attribution, averaged across neighbors (`src/evaluation/robustness.py:spearman_robustness`). $\rho$ measures feature *ranking* stability, not magnitude stability.

### Ensemble Aggregation (`ensemble_attribution`)

Converts each explainer's attribution vector to feature ranks (rank 1 = most important), computes weights inversely proportional to each explainer's attribution variance, takes a weighted rank average, then applies a **sign-disagreement penalty $\lambda = 0.15$**: features where more than half the explainers disagree on sign have their ranks inflated by $(1 - \lambda)$. The ensemble column uses TreeSHAP + KernelSHAP + LIME (GeoSHAP excluded because it is only computed for 20/50 points).

### Results (mean Spearman ρ)

| Level | TreeSHAP | KernelSHAP | GeoSHAP¹ | LIME | Ensemble |
|---|---|---|---|---|---|
| Random | 0.845 | 0.847 | 0.720 | 0.523 | — |
| Medoid | 0.515 | 0.531 | 0.336 | 0.473 | — |
| Spatial | 0.520 | 0.537 | 0.329 | 0.481 | — |

¹ GeoSHAP averaged over first 20 test points only (see note below).

**Key finding**: Random neighborhoods are far more stable than on-manifold ones — the opposite of naive intuition. The reason: random Gaussian perturbations stay close to the reference in the feature subspaces that tree splits actually care about. On-manifold medoid/spatial neighbors sample real houses that may sit on the other side of a major split boundary (e.g., `grade ≥ 8`), flipping the attribution ranking. GeoSHAP is the least stable because its geographically-local background distribution also varies across the neighborhood, adding a second source of variance.

**GeoSHAP subsampling note**: GeoSHAP requires a separate KernelSHAP call per point, making it ~50× slower than TreeSHAP. The robustness experiment limits GeoSHAP to the first 20 test points and 20 neighbors. GeoSHAP $\rho$ values are therefore means over $n = 20$ while all other explainers use $n = 50$. Direct numerical comparison with other columns should account for this difference.

### Trustworthiness Classification

A point is labeled `robust` if its spatial × TreeSHAP Spearman $\rho \geq 0.8$ (threshold $r_{\text{th}} = 0.8$). A 5-NN regressor is trained on (TreeSHAP attribution, prediction) features to predict $\rho$ for new points.

**Result: 1 of 50 test points clears the threshold.** Under a strict on-manifold standard, essentially no explanation in this housing model is trustworthy.

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

2. **Visibility increases monotonically with Moran's I** — structural: 0.014 → 0.134 (SHAP), 0.106 → 0.175 (LIME); demographic: 0.136 → 0.140 (SHAP), 0.190 → 0.207 (LIME). Counterintuitively, *higher-I features leak more*, not less. Spatial clumpiness makes laundering harder, not easier.

3. **Two regimes, not one.** Structural features (in baseline) show low visibility because the attack hides by amplification: the innocuous model excludes them, so the scaffold amplifies their attribution on real inputs. Demographic features (off baseline) show higher visibility because they were never part of the honest model; laundering happens indirectly via structural proxies, not by suppression of the biased feature itself. Visibility fraction captures both regimes in a single metric.

4. **Binary features are under-penalized.** `waterfront`'s P75 = 0, so only 0.8% of test points are penalized — the attack has little signal to hide. A sweep with adjusted thresholds (median rather than P75 for near-constant features) or additional mid-$I$ features (e.g., `grade`, $I$ = 0.365) would strengthen the structural regime.

5. **Reframe.** Not "SHAP fails in geospatial settings" but "explainer robustness in geospatial settings depends on feature role (on- vs off-baseline) and spatial clustering in non-obvious ways." Residual leakage grows with spatial homogeneity, even though OOD-detector performance is $I$-invariant.

**Outputs**: `results/feature_sweep/sweep_summary.json`, `sweep_plot.png` (two-panel: visibility + raw attribution vs. Moran's $I$, role-colored).

---

## Experiment 9 — SBM-as-Explainer: Scaffold-Attack Detection via pEx-SBM

**Scripts**: `scripts/run_sbm_battery.py`, `scripts/run_layer_density_test.py`, `scripts/run_k_scaling_sweep.py`, `scripts/run_lipschitz_sweep.py`, `scripts/run_tanh_sweep.py`; R sampler at `papers/pexsbm/pex-sbm.R`.
**Entry point**: `docs/sbm_explainer/README.md` → `summary_apr22.md` → `cosine-expression.md`.

### Motivation

Exps 1–8 establish *that* Slack-style scaffold attacks hide coordinate-based redlining from post-hoc explainers. Experiment 9 asks the complementary detection question: can a separate inference procedure **recover the protected layer from the scaffold's behavior**, without access to the biased model internals? The answer pivots on a behavioral-fingerprint encoder fed into a probabilistic exchangeable stochastic block model (pEx-SBM).

### Pipeline

**Rung-3 behavioral fingerprint.** For each of $N$ test points $x_i$, construct a vector by evaluating the full scaffolded model on $x_i$ and on $K$ Gaussian perturbations:

$$e_i = \bigl[\,f_{\text{scaffold}}(x_i),\; f_{\text{scaffold}}(x_i + \delta_1),\; \ldots,\; f_{\text{scaffold}}(x_i + \delta_K)\,\bigr] \in \mathbb{R}^{K+1}$$

Rungs 1 (attribution vectors) and 2 (honest-booster leaf indices) both produce posterior-blind graphs (regime-b null, $\Delta \approx 0$). Rung 3 — evaluating the full routed scaffold — is the first encoder that produces a middle-regime graph (see `docs/sbm_explainer/design.md`).

**Graph construction.** $N=200$ nodes, 8-NN cosine similarity on $\{e_i\}$. Protected layer ($\text{pct\_black\_latino} > 0.25$ for housing; race / gender / `racePctWhite` on external datasets) labels each node.

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

**The rung-3 + finite-K pEx pipeline is specifically a Slack-style scaffold-attack detector.** It is not a feature-geometry detector (honest null), not a bare-gate detector (biased null), and not an innocuous-smoothness detector (innocuous null). The signal is a structural fingerprint of the scaffold's routing mixture — detectable precisely because the strict OOD detector a Slack attack *needs* (to hide the bias from the explainer) creates the one-anchor-plus-$K$-refinements pattern rung-3 picks up.

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

Figures: `docs/sbm_explainer/figures/k_scaling.png`, `tanh_sweep.png`, `tanh_seedscale.png`.

### What this retracts

1. **Regime (b) is encoder-induced, not universal.** Rung 3 produces a middle-regime graph ($C$ ratio 2.51) at comparable log-BF to the regime-b attribution graph. The posterior does carry layer structure when the encoder supplies something coherent to block on. The earlier `plain-sbm-poc.md` §13d / §14 claim that "finite-K pEx is structurally unable to differ from plain" is retracted. Correct statement: finite-K pEx differs from plain only on graphs whose posterior has unsaturated layer awareness ($C$ ratio in the ~1–3 range).

2. **pEx-SBM nulls were graph-specific, not structural.** They held on attribution-vector graphs, which are posterior-blind; a well-chosen *encoder* places the posterior in the middle regime where the prior has room to act.

### Open directions (pre-Apr 2026 Peng meeting)

- Lead with finite-K $\Delta$ (clean Apr-22 narrative) vs. H-DP refinement (richer but arity-mismatched).
- Workshop / TMLR target with the four-dataset result, or NeurIPS-main with a formal recovery-probability bound for plain-SBM as a function of ($m$-variability, $K$, layer-alignment).
- $\beta$-sweep (tanh probe) retained as mechanism confirmation; sinusoidal Lipschitz sweep retired.
- Penalty-magnitude sweep ({0.10, 0.30, 0.50}) to confirm $\Delta$ robustness.
- Predictive test on H-DP gate-boundary clusters: do they exhibit higher held-out adversarial sensitivity than pure clusters?

**Outputs**: `results/sbm_explainer/` (battery, K-sweep, tanh-sweep, Lipschitz-sweep, four-dataset fingerprint subfolders, H-DP cluster-attribution CSVs). Detailed logs: `docs/sbm_explainer/day_log_apr22.md`, `summary_apr22.md`, `results.md`, `followup_directions.md`.

---

## Codebase Structure

```
geospatial-xai-attacks/
├── CLAUDE.md                       Repo rules and status
├── project_summary.md              This document
├── requirements.txt                Pinned dependencies
├── configs/
│   ├── baseline.yaml               10-feature XGBoost config
│   ├── explanations.yaml           KernelSHAP + LIME + GeoSHAP config
│   ├── scaffold.yaml               11-feature attack config
│   └── no_location.yaml            8-feature ablation config
├── data/
│   ├── raw/                        seattle_sample_3k.csv, TIGER shapefiles (git-ignored)
│   └── processed/                  Demographics CSV, spatial join output, GeoJSON
├── docs/
│   ├── roadmap.md                  Step-by-step plan with status
│   ├── experiment_registry.md      Summary table of all experiments
│   ├── project_overview.md         Deep technical documentation
│   ├── results_summary.md          Narrative walkthrough for meetings
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
│   ├── run_robustness.py           Experiment 5: 3×5 robustness matrix
│   ├── run_worst_case.py           Experiment 6: worst-case x* sweep
│   ├── run_morans_i.py             Experiment 7: Moran's I spatial autocorrelation survey
│   ├── run_feature_sweep.py        Experiment 8: scaffold attack vs Moran's I (5 features)
│   ├── run_sbm_battery.py          Experiment 9: rung-3 + finite-K pEx battery
│   ├── run_layer_density_test.py   Log-BF / layer-density diagnostic for rung-3 graphs
│   ├── run_k_scaling_sweep.py      Experiment 9: K-sweep mechanism test
│   ├── run_lipschitz_sweep.py      Experiment 9: sinusoidal Lipschitz sweep (retired)
│   ├── run_tanh_sweep.py           Experiment 9: tanh Lipschitz sweep (mechanism confirmation)
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

Each experiment can also be run independently; outputs land in `results/<experiment>/`. The full pipeline takes approximately 20–40 minutes on a modern laptop (GeoSHAP and LIME steps dominate runtime).

---

## Known Limitations and Future Work

1. **Worst-case sweep is TreeSHAP-only.** Extending to LIME requires ~200 separate `explain_lime` calls per (point, eps) pair. A practical extension: subsample to n=20 candidates for LIME.
2. **GeoSHAP robustness is computed on 20/50 test points.** Direct comparison with other explainer columns is approximate. Increasing `N_EXPLAIN_GEO` to 50 would require ~10 hours on a laptop or parallelization.
3. **No statistical significance testing.** Current robustness results are single-run. N≥5 runs with different random seeds would enable confidence intervals.
4. **Honest GeoSHAP vs adversarial comparison (Fig 1).** Honest GeoSHAP is now generated by `run_explanations.py` and saved to `results/explanations/geoshap_values.npy`, enabling a clean three-panel comparison in Fig 1.
5. **Ensemble column in results_matrix.json.** The 3×5 matrix (with Ensemble as 5th column) is computed in `run_robustness.py` but requires re-running the experiment to populate `rho_matrix.npy` with the updated shape (50×3×5).
