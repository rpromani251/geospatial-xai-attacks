# Peng Slides Design Document
## Geospatial XAI: Adversarial Robustness + pEx-SBM Detection
### Meeting: 2026-04-23 | Robert Romani | Advisor: Peng Luo

---

## How to Use This Doc

Each slide entry follows this format:

```
## Slide N — <Title>
**Layout:** <one-line description>
**Bullets / Content:** …
**Figure:** `absolute/path/to/figure.png`
   _(caption)_
**Table:** (markdown table, copy-paste ready)
**Speaker notes:** <1–2 sentences>
```

- Every number here traces to a file in the repo (source noted in speaker notes or inline).
- Figure paths are absolute and verified to exist on disk (missing ones are flagged).
- For slides with tables, copy the markdown table into a Google Doc → Insert Table (or paste as plain text and reformat).
- Speaker notes are what to say aloud; bullet points are what goes on the slide.

---

## Table of Contents

| Slide | Title                                                      | Act |
| ----- | ---------------------------------------------------------- | --- |
| 1     | Title Slide                                                | I   |
| 2     | Roadmap — 6-Act Overview                                   | I   |
| 3     | Dataset — Seattle / King County Housing                    | I   |
| 4     | Explainer Stack                                            | I   |
| 5     | Slack Scaffold Recreation — Architecture (Exp 3)           | I   |
| 6     | Exp 3 Attribution Table                                    | I   |
| 7     | Exp 3 Narrative Findings                                   | I   |
| 8     | Vascotti 3×5 Robustness Matrix (Exp 5)                     | I   |
| 8b    | Exp 5b — Slack Suite Robustness Replication                | I   |
| 9     | Worst-Case x* Attack (Exp 6)                               | I   |
| 10    | Exp 4 — Drop UTM Coordinates                               | II  |
| 11    | Exp 4b — No-Location Scaffold Still Works                  | II  |
| 12    | Moran's I — Motivation                                     | III |
| 13    | Exp 7 — Moran's I Survey (11 Features)                     | III |
| 14    | Exp 8 Raw Visibility Sweep                                 | III |
| 15    | Exp 8b Normalization — Base-Importance Artifact            | III |
| 16    | April 16 Proposal — Low-I Bias Triggers                    | III |
| 17    | Exp 11 SPS — Untestable                                    | III |
| 18    | Exp 10 FEMA 100-yr — Untestable                            | III |
| 19    | Exp 10b FEMA 500-yr — Untestable                           | III |
| 20    | Exp 10c KC Parcel Scale                                    | III |
| 21    | Exp 10d Geologic Hazard — The One That Works               | III |
| 22    | GeoSHAP Reversal: DISCONFIRMED                             | III |
| 23    | Revised Headline — Moran's I Does Not Moderate             | III |
| 24    | Honest vs Adversarial Laundering Comparison                | III |
| 25    | pEx-SBM Motivation — Detection                             | IV  |
| 26    | pEx-SBM §2 Primer                                          | IV  |
| 27    | Encoder Ladder — Three Rungs                               | IV  |
| 28    | Exp 9 Housing Headline                                     | IV  |
| 29    | Four-Dataset Replication                                   | IV  |
| 30    | Mechanism — Anchor-Dominance Erosion                       | IV  |
| 31    | The Three §2 Outputs                                       | V   |
| 32    | Ψ Matrix (housing, pct_black_latino layer, K=4)            | V   |
| 33    | Pairwise Co-Clustering                                     | V   |
| 34    | Per-Node Uncertainty                                       | V   |
| 35    | Block Characterization                                     | V   |
| 36    | Block 1 Anchor-Below-Tail Flip                             | V   |
| 37    | Slack-Suite §2 + Block Outputs (Cross-dataset replication) | V   |
| 38    | Exp 12 §6.2 Spectrum Sweep                                 | V   |
| 39    | Theoretical Gaps                                           | VI  |
| 40    | Empirical Gaps                                             | VI  |
| 41    | Three Asks for Peng                                        | VI  |
| 42    | What's Established                                         | VI  |
| 43    | Timeline Sketch                                            | VI  |
| 44    | Compute + Collaborator Asks                                | VI  |
| 45    | Q&A                                                        | VI  |

---

## Act I — Setup, Explainers, Slack + Vascotti Recreation (Slides 1–9)

---

## Slide 1 — Title

**Layout:** Centered title block, no figure.

**Bullets / Content:**
- Title: Geospatial XAI: Adversarial Robustness + pEx-SBM Detection
- Robert Romani
- Advisor: Peng Luo
- 2026-04-23

**Speaker notes:** Opening slide. No preamble needed — dive straight into the roadmap next.

---

## Slide 2 — Roadmap

**Layout:** Single-column bulleted list with act labels.

**Bullets / Content:**
- **Act I (Slides 1–9):** Setup, dataset, explainers, Slack scaffold recreation (Exp 3), Vascotti robustness (Exp 5), worst-case x* (Exp 6)
- **Act II (Slides 10–11):** No-location ablation — does removing UTM_X/Y neutralize the attack?
- **Act III (Slides 12–24):** Moran's I exploration — does spatial autocorrelation moderate foolability? (Short answer: no)
- **Act IV (Slides 25–30):** pEx-SBM as scaffold-attack detector — pipeline, four-dataset replication, mechanism
- **Act V (Slides 31–38):** §2 primitives + block characterization + spectrum sweep
- **Act VI (Slides 39–45):** Gaps, three asks, timeline, Q&A

**Speaker notes:** Use this slide as a navigation anchor. The core meeting deliverable is Acts IV–V; Acts I–III are the scaffolding that motivates them.

---

## Slide 3 — Dataset — Seattle / King County Housing

**Layout:** Two-column: bullets left, figure right.

**Bullets / Content:**
- 3,000 King County housing transactions (`data/raw/seattle_sample_3k.csv`)
- 10 baseline features: `bathrooms`, `sqft_living`, `sqft_lot`, `grade`, `condition`, `waterfront`, `view`, `age`, `UTM_X`, `UTM_Y`
- Target: `log_price` (natural log of sale price)
- UTM projection EPSG:32610 — distances in meters (Euclidean k-NN meaningful)
- Demographics join: ACS 2011–2015 B03002, 398 King County tracts → 334 matched tracts (`data/processed/seattle_housing_with_demographics.csv`)
- Derived: `pct_black`, `pct_latino`, `pct_black_latino` per tract
- 80/20 train/test split, `random_state=42` → 2,400 train / 600 test
- Baseline XGBoost R² = **0.8895** (source: `results/baseline/metrics.json`)

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/figures/demographics_combined.png`
   _(Demographic choropleth maps of King County tracts showing pct_black_latino spatial distribution.)_

**Speaker notes:** UTM coordinates are used throughout because Euclidean distances are metrically meaningful; this matters for GeoSHAP's k-NN background and the worst-case radius sweep.

---

## Slide 4 — Explainer Stack

**Layout:** Table-only, centered.

**Bullets / Content:**
- Four explainers implemented; all return `(n_test, n_features)` NumPy arrays.

| Explainer | Type | Key property |
|-----------|------|-------------|
| **KernelSHAP** | Coalition-based Shapley | Global background (100 training rows); model-agnostic |
| **TreeSHAP** | Path-dependent exact Shapley | Exact; no background needed; fast batching |
| **GeoSHAP** | Geographically-local KernelSHAP | Per-point background = k=100 nearest UTM neighbors; attribution answers "vs. local geography" |
| **LIME** | Local linear surrogate | 500 Gaussian perturbations per point; partial OOD leakage creates partial scaffold resistance |

**Speaker notes:** GeoSHAP's geographic background is the key reason it turns out to be the most vulnerable explainer — every coalition query lands inside the real-data manifold and hits the biased branch.

---

## Slide 5 — Slack Scaffold Recreation — Architecture (Exp 3)

**Layout:** Single-column with nested sub-bullets.

**Bullets / Content:**
- **Goal:** Reproduce Slack et al. adversarial scaffold in geospatial housing setting
- **Biased model:** `f_biased(x) = f_honest(x_base) − 0.30 × 1{pct_black_latino > 0.25}` — penalty ≈ −26% in linear price; 14% of test points (84/600) above threshold
- **Innocuous model:** XGBoost on 8 safe features (no coordinates, no demographics); R² = 0.6414
- **OOD detectors** (Random Forest classifiers):
  - LIME detector: Gaussian noise (σ=0.3, 30× replication) — test accuracy **100.0%**
  - SHAP detector: k-means coalition substitutions (10 clusters, 20,000 samples) — test accuracy **99.5%**
- **Routing:** Real inputs → `f_biased`; OOD queries → `psi_innocuous`

| Scaffold | Pearson r (fidelity) | % routed to biased |
|----------|---------------------|-------------------|
| LIME scaffold | 0.9996 | 99.5% |
| SHAP scaffold | 0.9109 | 72.3% |

**Speaker notes:** Source: `results/scaffold/metrics.json`. SHAP routes fewer real points (72.3%) because k-means coalition perturbations overlap more with the real manifold, making the OOD boundary stricter.

---

## Slide 6 — Exp 3 Attribution Table

**Layout:** Table-only, landscape, centered. Use a wide format.

**Bullets / Content:**
- Mean |attribution| over 200 test points, honest vs adversarial scaffold.
- Honest columns = 11-feature XGBoost (10 baseline + pct_black_latino, R²=0.8889). Source: `docs/comparison_honest_vs_adv_off_baseline.md`.

| Feature          | Honest SHAP | Adv SHAP     | Honest LIME | Adv LIME | Honest GeoSHAP | Adv GeoSHAP  |
| ---------------- | ----------- | ------------ | ----------- | -------- | -------------- | ------------ |
| UTM_Y            | 0.106       | 0.050 (−53%) | 0.110       | 0.124    | —              | 0.009 (−92%) |
| pct_black_latino | 0.022       | 0.035        | 0.026       | 0.108    | —              | 0.030        |
| sqft_living      | 0.066       | 0.082        | 0.092       | 0.096    | —              | 0.074        |
| grade            | 0.048       | 0.075        | 0.087       | 0.087    | —              | 0.060        |
| age              | 0.012       | 0.033        | 0.046       | 0.057    | —              | —            |
| bathrooms        | 0.009       | 0.020        | 0.014       | 0.015    | —              | 0.021        |

Note: Honest GeoSHAP column's demographics row is undefined by construction — the honest model has UTM_X/UTM_Y in its feature vector (GeoSHAP is NOT location-blind; corrected 2026-04-23) but does NOT include pct_black_latino. Adv GeoSHAP UTM_Y sourced from project_summary.md Exp 3 table.

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/figures/honest_vs_adv_shap.png`
   _(Bar chart comparing honest vs adversarial SHAP feature importances.)_

**Speaker notes:** Source for UTM_Y Adv SHAP: `docs/comparison_honest_vs_adv_off_baseline.md` (−0.056 drop from 11-feat baseline). GeoSHAP adversarial values from `project_summary.md` Exp 3 table; full 11-column table at `docs/project_overview.md` §4.

---

## Slide 7 — Exp 3 Narrative Findings

**Layout:** Two-column: bullets left, figure right.

**Bullets / Content:**
- **SHAP:** UTM_Y drops 53–57% (0.106 → 0.050); spatial signal laundered into `age` (+0.021), `grade` (+0.027), `bathrooms` (+0.011)
- **GeoSHAP is most compromised:** UTM_Y collapses to 0.009 (−92%); geographic-locality backgrounds route every coalition to the biased branch
- **LIME partially resists:** UTM_Y stays top feature (0.124 vs 0.110 honest); pct_black_latino rises to 0.108 — Gaussian perturbations straddle real manifold, partially leaking through OOD detector
- **Spatial correlation amplified:**

| Model | Pearson r (pct_minority vs log_price) | p-value |
|-------|--------------------------------------|---------|
| Honest | −0.506 | 2.6×10⁻⁴⁰ |
| Biased | −0.727 | 9.6×10⁻¹⁰⁰ |

- Biased model amplifies redlining signal by +0.221 in absolute correlation
- **Key finding:** Embedding spatial structure into the explainer (GeoSHAP) does NOT protect it from a spatially-structured attack

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/figures/honest_vs_adv_lime.png`
   _(Honest vs adversarial LIME attributions bar chart.)_

**Speaker notes:** Spatial correlation values from `results/scaffold/spatial_correlation.json`. The GeoSHAP vulnerability is the novel finding — the same spatial structure that makes GeoSHAP intuitively appealing makes it maximally vulnerable.

---

## Slide 8 — Vascotti 3×5 Robustness Matrix (Exp 5)

**Layout:** Table-only, centered, with brief bullets above.

**Bullets / Content:**
- Port of Vascotti et al. robustness framework to geospatial setting
- Three neighborhood levels: Random (off-manifold Gaussian noise, scale=0.05σ), Medoid (k-NN pool of 200 real training houses), Spatial (α=0.5 blend of geographic + feature distance)
- All neighborhoods filtered by `|f(neighbor) − f(ref)| < 0.15` log-price
- Metric: mean Spearman ρ (feature ranking stability) over 50 test points
- Source: `results/robustness/results_matrix.json`

**Table (copy-paste):**

| Level \ Explainer | TreeSHAP | KernelSHAP | GeoSHAP¹ | LIME | Ensemble |
|---|---|---|---|---|---|
| Random | 0.845 | 0.847 | 0.720 | 0.523 | 0.763 |
| Medoid | 0.515 | 0.531 | 0.336 | 0.473 | 0.635 |
| Spatial | 0.520 | 0.537 | 0.329 | 0.481 | 0.633 |

¹ GeoSHAP averaged over first 20 test points only.

- **Key finding:** Random > Medoid ≈ Spatial — the opposite of naive intuition
- **Reason:** Random perts stay close in the feature subspaces that tree splits care about; real neighbors cross major splits (e.g., `grade ≥ 8`)
- Trustworthiness (spatial × TreeSHAP, r_th=0.8): **1 / 50 points** pass

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/robustness/figures/heatmap.png`
   _(Heatmap of Spearman ρ by level and explainer.)_

**Speaker notes:** Ensemble column values (Random 0.763, Medoid 0.635, Spatial 0.633) from `results/robustness/results_matrix.json` after re-run to populate the 5th column. GeoSHAP is the least stable because its per-point geographic background also varies across the neighborhood, adding a second variance source.

---

## Slide 8b — Exp 5b: Slack Suite Robustness Replication

**Layout:** Two-column — condensed table left, bar-chart figure right. Brief bullets above table.

**Bullets / Content:**
- Replicates the Exp 5 2×3 matrix on three Slack tabular datasets: COMPAS recidivism, German Credit, Communities-and-Crime (psi_two)
- Honest RF classifier (100 trees), n=30 test points per dataset, 50 neighbors/level, δ=0.15 on class-1 probability
- All three tasks: binary classification; TreeSHAP class-1 slice from shap 0.51.0 `(n, d, 2)` output
- Script: `scripts/fix_treeshap_and_cc.py`; results: `results/slack_robustness/`

**Table (condensed — KernelSHAP + TreeSHAP; LIME noted separately):**

| Dataset | Level | TreeSHAP | KernelSHAP | LIME |
|---|---|---|---|---|
| Housing (Exp 5 ref) | Random | 0.845 | 0.847 | 0.523 |
| Housing | Medoid | 0.515 | 0.531 | 0.473 |
| COMPAS | Random | 0.924 | 0.888 | 0.516 |
| COMPAS | Medoid | 0.542 | 0.478 | 0.496 |
| German | Random | 0.984 | 0.928 | 0.270 |
| German | Medoid | 0.326 | 0.277 | **0.481** |
| CC psi_two | Random | 0.836 | 0.749 | 0.271 |
| CC psi_two | Medoid | 0.333 | 0.473 | 0.137 |

_Bold = LIME reversal (Medoid > Random on German)._

**Random − Medoid gap summary:**

| Explainer | Housing Δ | COMPAS Δ | German Δ | CC Δ | Mean Δ |
|---|---|---|---|---|---|
| TreeSHAP | +0.330 | +0.382 | +0.658 | +0.502 | **+0.468** |
| KernelSHAP | +0.316 | +0.410 | +0.651 | +0.276 | **+0.413** |
| LIME | +0.050 | +0.020 | **−0.211** | +0.134 | **−0.002** |

**Key finding:**
- **KernelSHAP:** Random > Medoid holds in all 4 datasets. The gap actually _grows_ on Slack data (KernelSHAP housing +0.316 → COMPAS +0.41 → German +0.651). This rules out housing or geospatial structure as the driver — it is a general tree-model property.
- **LIME on German Credit reverses:** Medoid ρ = 0.481 > Random ρ = 0.270 (Δ = −0.211). Mechanism: LIME's independent Gaussian perturbations sample off-manifold in German's categorical-heavy feature space, destabilizing the local linear fit. Medoid neighbors are on-manifold, so LIME fits more consistently.
- **Mean LIME Δ across all 4 datasets: −0.002** — LIME's behavior is essentially dataset-agnostic on average but with high variance, masking the German reversal.

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/docs/figures/fig_robustness_slack_vs_housing.png`
_(Grouped bar chart: 4 dataset groups × 3 explainer panels × 2 levels. Error bars = std. Housing shown with hatched reference bars.)_

**Speaker notes:** Exp 5's headline — "Random > Medoid for explanation stability" — survives as a universal claim for SHAP-based explainers across geospatial and tabular domains. The KernelSHAP gap is larger in the Slack suite (+0.41 on COMPAS, +0.65 on German) than in housing (+0.32), which is the _opposite_ of what you'd expect if spatial structure were doing the work. TreeSHAP shows the same pattern with even larger gaps. LIME is the interesting exception: the housing result (LIME gap = +0.050) was already the weakest column in Exp 5, and this replication shows why — LIME's stability ordering is sensitive to the feature geometry, specifically the proportion of binary/categorical features. German Credit is categorical-heavy, so Gaussian perturbations land in implausible regions and produce unstable local fits. The practical takeaway: when advising practitioners, separate the SHAP advice ("prefer random neighborhoods or at least don't trust medoid stability") from the LIME advice ("it depends on your feature space").

---

## Slide 9 — Worst-Case x* Attack (Exp 6)

**Layout:** Two-column: bullets + table left, figure right.

**Bullets / Content:**
- **Formulation:** `x* = argmax_{x ∈ N_ε(y)} ||attr(x) − attr(y)||₂` where `N_ε(y)` = real training points within ε meters in UTM space
- No adversarial machinery: x* is a real house
- Implementation: TreeSHAP; 50 reference points; candidates subsampled to 200 per radius
- Source: `results/worst_case/summary.json`

| ε_geo (m) | Mean worst-case L2 | Std | Mean candidates |
|---|---|---|---|
| 500 | 0.165 | 0.079 | 3.6 |
| 1,000 | 0.228 | 0.083 | 12.4 |
| 2,000 | 0.275 | 0.080 | 39.3 |
| 5,000 | 0.382 | 0.083 | 150.0 |
| 10,000 | 0.411 | 0.072 | 195.9 |

- At ε=5,000 m: mean worst-case L2 = **0.382** vs mean average-neighbor L2 = **0.147** → **2.6× gap**
- Most destabilized features: `grade` (0.260) and `sqft_living` (0.245) — together ≈75% of total L2
- No adversarial tools needed — any user with dataset access can find x*

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/worst_case/figures/eps_sweep.png`
   _(ε-sweep curve showing mean worst-case L2 vs geographic radius.)_

**Speaker notes:** The 2.6× gap is the core finding: average-robustness metrics (Exp 5) understate real risk by more than half. Source for all numbers: `results/worst_case/summary.json` and `project_summary.md` Exp 6 section.

---

## Act II — No-Location Ablation (Slides 10–11)

---

## Slide 10 — Exp 4 — Drop UTM Coordinates

**Layout:** Two-column: bullets left, figure right.

**Bullets / Content:**
- **Question:** If a practitioner removes UTM_X/UTM_Y to eliminate spatial signal, does the model become location-agnostic?
- Same XGBoost hyperparameters, 8 safe features (no coordinates)
- R² drops: **0.8895 → 0.6174** (source: `results/no_location/metrics.json`) — UTM contributes ~27 R²-points
- `age` rises to 3rd place across all SHAP variants — proxies for neighborhood vintage
- LIME's rankings diverge sharply: `view` (0.092), `condition` (0.051), `waterfront` (0.049) now receive large weight — binary amenity features produce large local swings for LIME's linear surrogate

| Feature | TreeSHAP | KernelSHAP | LIME |
|---------|----------|------------|------|
| grade | 0.084 | 0.082 | 0.147 |
| sqft_living | 0.077 | 0.084 | 0.144 |
| age | 0.058 | 0.063 | 0.021 |
| view | 0.013 | 0.013 | 0.092 |
| condition | 0.011 | 0.012 | 0.051 |
| waterfront | 0.002 | 0.002 | 0.049 |

- Source: `docs/project_overview.md` §5

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/no_location/figures/ranking_comparison.png`
   _(Feature ranking comparison between 10-feature and no-location models.)_

**Speaker notes:** The `age` rise as vintage proxy is the key takeaway — simply removing coordinates does not neutralize spatial information; it just displaces attribution into correlated structural features.

---

## Slide 11 — Exp 4b — No-Location Scaffold Still Works

**Layout:** Two-column: bullets left, table right.

**Bullets / Content:**
- **Question:** Does the adversarial scaffold still hide pct_black_latino without UTM coordinates?
- Same attack architecture, but feature list reduced from 11 → 9 (UTM_X/UTM_Y removed from all models and OOD detectors)

| Scaffold | OOD accuracy | Fidelity r | % routed to biased |
|----------|-------------|-----------|-------------------|
| LIME | 100.0% | 1.0000 | 99.8% |
| SHAP | 99.3% | 0.9834 | 78.8% |

- Attack succeeds without coordinates; OOD performance **unchanged** from Exp 3
- pct_black_latino visibility slightly **increases** without UTM: SHAP 0.044 vs 0.035 (Exp 3), LIME 0.110 vs 0.108 (Exp 3)
- Without coordinate channels to absorb: `grade` (0.090) and `age` (0.063) absorb penalty under SHAP
- Spatial correlation amplification: honest −0.289, biased −0.600 → +0.311 amplification (even larger than Exp 3's +0.221)
- **Take-away: OOD routing is the attack mechanism, not coordinate laundering**

- Source: `project_summary.md` Exp 4b section

**Speaker notes:** This is the logical "null test" for coordinate laundering — when the channel is removed, the attack still works and the bias leaks slightly more. The mechanism is in the routing architecture, not the spatial features.

---

## Act III — Moran's I Exploration (Slides 12–24)

---

## Slide 12 — Moran's I — Motivation

**Layout:** Single-column bullets.

**Bullets / Content:**
- **Peng's 2026-04-16 question:** Does the scaffold attack depend on the biased feature being spatially clumped?
- Hypothesis: high-I features easier to hide because spatial clumpiness lets OOD detector cleanly separate real from perturbed data
- Prior experiments all used `pct_black_latino` (I=0.834) — strongly clumped
- **Plan:** (1) Survey Moran's I for all features (Exp 7), (2) Sweep attack across I spectrum (Exp 8), (3) Test genuinely low-I bias triggers (Exps 10–11)

**Speaker notes:** This entire act is motivated by a single question from April 16. The answer is negative — Moran's I does not predict attack feasibility — and Acts IV–V present the positive result that does.

---

## Slide 13 — Exp 7 — Moran's I Survey (11 Features)

**Layout:** Two-column: table left, figure right.

**Bullets / Content:**
- Global Moran's I, 8-NN row-standardized spatial weights on UTM coordinates
- All 11 non-coordinate features; 499-permutation pseudo-p-values (all p < 0.05)
- Source: `results/morans_i/morans_i_results.json`

| Feature | Moran's I | Level |
|---------|-----------|-------|
| `pct_black` | 0.850 | very high |
| `pct_black_latino` | 0.834 | very high |
| `pct_latino` | 0.756 | high |
| `sqft_lot` | 0.469 | medium |
| `grade` | 0.365 | medium |
| `age` | 0.317 | medium |
| `sqft_living` | 0.291 | low–medium |
| `view` | 0.242 | low |
| `bathrooms` | 0.183 | low |
| `condition` | 0.090 | very low |
| `waterfront` | 0.051 | very low |

- Sweep features selected (Exp 8): waterfront (0.051), view (0.242), sqft_lot (0.469), pct_latino (0.756), pct_black (0.850)

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/morans_i/morans_i_barplot.png`
   _(Horizontal bar chart of Moran's I values for all 11 features, sorted.)_

**Speaker notes:** Wide spectrum: demographic features cluster tightly (I≈0.75–0.85) while structural features like waterfront are nearly spatially random (I≈0.05).

---

## Slide 14 — Exp 8 Raw Visibility Sweep

**Layout:** Two-column: table left, figure right.

**Bullets / Content:**
- For each sweep feature: build BiasedModel (penalty −0.30 at P75 threshold), train OOD detectors, run SHAP + LIME
- Visibility = mean |adv attribution on biased feature| / mean |total adv attribution|
- Source: `results/feature_sweep/sweep_summary.json`

| Feature | Moran's I | SHAP vis | LIME vis | % penalized | SHAP OOD | LIME OOD |
|---------|-----------|----------|----------|-------------|----------|----------|
| `waterfront` | 0.051 | 0.0135 | 0.106 | 0.8% | 99.3% | 100% |
| `view` | 0.242 | 0.0703 | 0.160 | 9.7% | 99.3% | 100% |
| `sqft_lot` | 0.469 | 0.134 | 0.175 | 25.3% | 99.3% | 100% |
| `pct_latino` | 0.756 | 0.136 | 0.190 | 24.7% | 99.4% | 100% |
| `pct_black` | 0.850 | 0.140 | 0.207 | 23.2% | 99.6% | 100% |

- **OOD accuracy flat at 99.3–100%** across the full I spectrum — Moran's I does NOT predict attack feasibility
- Visibility increases monotonically with I (Spearman +0.94) — counterintuitively, higher-I features leak more

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/feature_sweep/sweep_plot.png`
   _(Two-panel: visibility + raw attribution vs Moran's I, role-colored by structural vs demographic.)_

**Speaker notes:** The flat OOD curve is the key result: the attack's routing mechanism is I-invariant. The visibility trend (higher I → more leakage) turns out to be a base-importance artifact, shown on the next slide.

---

## Slide 15 — Exp 8b Normalization — Base-Importance Artifact

**Layout:** Single-column with bullets and one-line table.

**Bullets / Content:**
- **Problem:** High-I features tend to be more predictively important in the honest model → more attribution headroom → raw visibility trend could be spurious
- **Exp 8b:** Normalize adversarial visibility by honest-baseline SHAP importance for same feature
- Three A2-missing features also swept: `sqft_living` (I=0.291), `grade` (I=0.365), `condition` (I=0.090)
- Source: `results/feature_sweep_a2/`

| Normalization | Spearman ρ (vis vs I) | p-value |
|---|---|---|
| Raw visibility | +0.94 | < 0.05 |
| Normalized by honest base importance | −0.43 | 0.40 (not sig.) |

- **Conclusion: I-trend in raw visibility is a base-importance artifact, not a real spatial autocorrelation effect**
- Correct moderator is feature role (on-baseline structural vs off-baseline demographic), not I

**Speaker notes:** This result reframes the entire Moran's I arc — the feature's structural role in the baseline model predicts foolability, not its spatial autocorrelation. This is a stronger and more transferable claim.

---

## Slide 16 — April 16 Proposal — Low-I Bias Triggers

**Layout:** Single-column bullets.

**Bullets / Content:**
- Peng's April 16 follow-up: test with a genuinely low-I binary trigger (not just low-I structural features)
- **Hypothesis:** GeoSHAP would reverse from most-compromised to most-resistant when the trigger has low spatial autocorrelation
- Three candidate triggers tested:
  - FEMA 100-yr floodplain (`in_floodplain`) — coverage 0.2%
  - FEMA 500-yr floodplain (`in_flood_500yr`) — coverage 0.5%
  - KC parcel-scale floodplain (59k records) — coverage 0.47%
  - Geologic hazard zone (`in_geologic_hazard`) — I=0.246, coverage 5.1%
  - SPS school zone penalty (`in_low_score_zone`) — I≈0.80

**Speaker notes:** This slide sets up the cascade of negative results on slides 17–20 before the one positive case on slide 21.

---

## Slide 17 — Exp 11 SPS — Untestable

**Layout:** Single-column bullets.

**Bullets / Content:**
- **Trigger:** Seattle Public Schools (SPS) zone binaries as "genuinely low-I" replacement
- Measured I ≈ 0.80 — zone binaries cannot be low-I at housing density in Seattle
- School zones are spatially contiguous by construction → high I inevitable
- **Result: Untestable via this route.** Zone binaries at urban density will always be spatially clumped.

**Speaker notes:** Slide is brief. The key point is that finding a genuinely low-I binary trigger at the census-tract level is harder than it seems — spatial contiguity almost always produces I > 0.5.

---

## Slide 18 — Exp 10 FEMA 100-yr — Untestable

**Layout:** Single-column bullets.

**Bullets / Content:**
- **Trigger:** `in_floodplain` (FEMA 100-yr) — I ≈ −0.0015 (confirmed very low, nearly spatially random)
- Coverage in 3k transaction sample: **0.2%** (4/200 test points above threshold)
- Degenerate layer split: too few positive examples to train or evaluate
- OOD detectors cannot distinguish protected-layer structure at this coverage
- **Result: Untestable.** Transaction samples systematically undersample rare hazard zones.

**Speaker notes:** The coverage problem is fundamental — rare hazard zones affect very few properties in any transaction-based dataset. This motivates the parcel-scale dataset attempt on the next slide.

---

## Slide 19 — Exp 10b FEMA 500-yr — Untestable

**Layout:** Single-column bullets.

**Bullets / Content:**
- **Trigger:** `in_flood_500yr` (FEMA 500-yr floodplain) — broader zone, higher coverage
- Coverage in 3k transaction sample: **0.5%** (still too sparse)
- R² of honest baseline with trigger: 0.8911 (+0.0016 vs 10-feat baseline) — trigger carries negligible price signal
- Honest SHAP for `in_flood_500yr` ≈ 0.000 — model cannot use it as a predictor
- Adversarial LIME attribution: 0.090 — scaffold manufactures LIME attribution but effect is artifact of OOD routing, not trigger importance
- **Result: Untestable** at this coverage.

**Speaker notes:** These two flood-zone experiments confirm that the coverage floor for the scaffold design is roughly 5% positive rate in the test set. Below that, there are too few anchor points to generate a detectable routing signal.

---

## Slide 20 — Exp 10c KC Parcel Scale

**Layout:** Single-column bullets.

**Bullets / Content:**
- **Motivation:** Transaction sample bias → switch to full KC parcel dataset (59,000 records)
- Still using FEMA 100-yr `in_floodplain` trigger
- Coverage: **0.47%** (transaction sample bias persists — floodplain properties sell less frequently)
- Baseline R² = 0.7625 (lower than 3k sample due to greater heterogeneity)
- 11-feat R² = 0.7617 (trigger adds no predictive value)
- Honest SHAP for `in_floodplain` ≈ 0.000; honest LIME = 0.121
- UTM_Y dominates both honest and adversarial SHAP (0.254 vs 0.227 adversarial — small drop vs 3k-set −0.056)
- LIME sees through scaffold more at parcel scale (adversarial `in_floodplain` LIME = 0.150 vs honest 0.121)
- Source: `docs/comparison_honest_vs_adv_off_baseline.md` Exp 10c

**Speaker notes:** The parcel dataset confirms that transaction-sample bias is the core obstacle, not dataset size. Floodplain properties genuinely sell less frequently, so even 59k records give < 0.5% coverage.

---

## Slide 21 — Exp 10d Geologic Hazard — The One That Works

**Layout:** Two-column: bullets + table left, figure right.

**Bullets / Content:**
- **Trigger:** `in_geologic_hazard` (KC landslide/erosion/liquefaction hazard zones)
- Moran's I = **0.246** (confirmed low — genuinely non-clustered relative to demographics)
- Coverage: **5.1%** (13/200 explained points above threshold) — barely crosses testability floor
- R² of 11-feat honest baseline: 0.8875 (trigger adds slight negative signal, Δ = −0.0020)

| Feature | Honest SHAP | Honest LIME | Adv SHAP | Adv LIME |
|---------|------------|-------------|----------|----------|
| `in_geologic_hazard` | 0.000 | 0.022 | 0.013 | 0.186 |
| UTM_Y | 0.117 | 0.120 | 0.050 | 0.129 |
| sqft_living | 0.067 | 0.095 | 0.079 | 0.094 |

- Scaffold works: adv LIME = **8.5× honest LIME** (0.186 vs 0.022) — most dramatic manufacturing of attribution across all triggers
- SHAP more resistant (adv 0.013 vs honest ≈ 0.000)
- UTM_Y drop: −0.067 (consistent with other 3k-set experiments)
- Source: `docs/comparison_honest_vs_adv_off_baseline.md` Exp 10d

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/scaffold_geologic_hazard/figures/fig_attribution_comparison.png`
   _(Attribution comparison: honest vs adversarial, geologic hazard trigger.)_

**Speaker notes:** This is the only low-I trigger that reaches testable coverage. The 8.5× LIME amplification is the largest across all experiments — scaffold manufacturing of attribution works regardless of I, as long as coverage is sufficient.

---

## Slide 22 — GeoSHAP Reversal: DISCONFIRMED

**Layout:** Single-column bullets with result box.

**Bullets / Content:**
- **April 16 hypothesis:** GeoSHAP would reverse from most-compromised (Exp 3) to most-resistant at low I
- **Prediction:** Adv GeoSHAP attribution of geologic_hazard trigger > Adv GeoSHAP attribution of pct_black_latino
- **Result:** Adv GeoSHAP trigger attribution: 0.011 (geologic hazard) vs 0.013 (SHAP, Exp 3 pct_black_latino) — **NOT higher**
- UTM_Y adversarial GeoSHAP: −92% (same collapse pattern as Exp 3)
- GeoSHAP remains maximally compromised regardless of trigger I
- **Conclusion: April 16 hypothesis falsified.** Moran's I of the trigger does not determine GeoSHAP's vulnerability.

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/scaffold_geologic_hazard/figures/fig_geoshap_reversal.png`
   _(GeoSHAP reversal test: adversarial attributions under geologic hazard scaffold.)_

**Speaker notes:** The GeoSHAP vulnerability is structural — geographic backgrounds always land inside the real manifold and hit the biased branch — so I of the trigger is irrelevant.

---

## Slide 23 — Revised Headline — Moran's I Does Not Moderate

**Layout:** Single-column, large text, 4 bullets.

**Bullets / Content:**
- **What we know:** Moran's I does NOT moderate attack feasibility
  - OOD accuracy flat at 99.3–100% across I ∈ [0.05, 0.85]
  - GeoSHAP reversal disconfirmed
  - Normalized visibility trend non-significant (ρ = −0.43, p = 0.40)
- **The real moderator:** Feature role in the honest baseline
  - Structural features (in baseline): attack hides by amplification
  - Demographic features (off baseline): laundering via structural proxies
- **Why LIME resists:** LIME's Gaussian perturbations partially route to the real biased model; SHAP coalitions are OOD-detected and laundered cleanly
- **The detection signal must live inside the scaffold's routing mixture** — not in the trigger feature's geography

**Speaker notes:** This slide is the pivot from Act III to Act IV. The failure of the spatial autocorrelation hypothesis motivates looking inside the scaffold's routing behavior directly — which is exactly what pEx-SBM does.

---

## Slide 24 — Honest vs Adversarial Laundering Comparison

**Layout:** Two-column: bullets + table left, figure right.

**Bullets / Content:**
- Cross-experiment baseline comparison (source: `docs/comparison_honest_vs_adv_off_baseline.md`)
- Adding trigger to honest model: UTM_Y drops ≤ 0.010 (pct_black_latino) or ≤ 0.002 (all binary triggers)
- Adversarial scaffold: UTM_Y drops 0.050–0.068 — **5–7× larger than honest inclusion**

| Experiment | Trigger | Honest UTM_Y (11-feat) | Δ honest | Adv UTM_Y | Adv−Honest11 Δ |
|---|---|---|---|---|---|
| Exp 3 | pct_black_latino | 0.106 | −0.010 | 0.050 | −0.056 |
| Exp 10d | in_geologic_hazard | 0.117 | +0.001 | 0.050 | −0.067 |

- Scaffold is doing genuine laundering, not just credit-sharing
- UTM_Y drop is consistent across all 3k-set experiments (always to ≈0.050)

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/docs/figures/fig_honest_vs_adv_off_baseline.png`
   _(Summary figure: UTM_Y honest vs adversarial across all trigger experiments.)_

**Speaker notes:** The consistent 0.050 adversarial UTM_Y value across all experiments (regardless of trigger) is a structural signature of the scaffold: the innocuous model always lacks UTM features, so the SHAP background always routes to a model that can't use location.

---

## Act IV — pEx-SBM as Scaffold-Attack Detector (Slides 25–30)

---

## Slide 25 — Motivation — Detection

**Layout:** Single-column bullets.

**Bullets / Content:**
- **The problem:** Feature-side moderators (Moran's I, feature role) do not predict individual foolability — they describe regimes, not which points are most at risk
- **Detection question:** Can a separate inference procedure recover the protected layer from the scaffold's behavior, WITHOUT access to the biased model internals?
- **Answer:** Yes — via a rung-3 behavioral fingerprint fed into pEx-SBM
- **Why this matters:** An auditor has access to the scaffold as a black box (can query it); they do NOT have access to the biased model or OOD detector internals
- **The key insight:** The strict OOD detector a Slack-style attack needs (to hide bias from the explainer) creates a structural fingerprint the rung-3 encoder picks up

**Speaker notes:** This slide is the motivation pivot. The Moran's I exploration established what does NOT predict foolability; the pEx-SBM thread establishes what does — the routing mixture inside the scaffold.

---

## Slide 26 — pEx-SBM §2 Primer

**Layout:** Single-column bullets.

**Bullets / Content:**
- **pEx-SBM** = probabilistic Exchangeable Stochastic Block Model
- Nodes = test points; edges = behavioral similarity (8-NN cosine on fingerprint vectors)
- **Layer** = protected attribute (e.g., pct_black_latino > 0.25)
- **Two variants:**
  - Finite-K Dirichlet-Categorical pEx (ours, K=4 fixed): per-layer Dirichlet(γ) prior over K-simplex → Categorical node-to-block assignments → Collapsed Gibbs. `src/sbm/pex_sbm.py`
  - H-DP pEx (Durante et al. 2025, arXiv 2410.10619): K inferred via two-level nested Dirichlet Processes. R sampler at `papers/pexsbm/pex-sbm.R`
- **Ψ matrix:** For layer h and block k, Ψ_{h,k} = P(node in layer h is assigned block k). The §2 explanation primitive.
- **Metric:** Δ NMI = NMI(pEx partition, layer) − NMI(plain SBM partition, layer)
- **C-ratio:** within-layer / across-layer posterior co-clustering density — regime indicator

**Speaker notes:** The key difference between finite-K pEx and plain SBM is the prior: plain SBM has one shared Dirichlet, pEx has one per layer. This matters only when the posterior is in the middle regime (C-ratio 1–3).

---

## Slide 27 — Encoder Ladder — Three Rungs

**Layout:** Single-column with nested sub-bullets.

**Bullets / Content:**
- **Rung 1 — Attribution vectors:** e_i = SHAP or LIME attribution vector for point x_i. **Result: Δ ≈ 0** — attribution graphs are posterior-blind (regime b). Works on sqft_lot LIME only at saturated regime (NMI=0.916 at K=2, but plain SBM alone is sufficient).
- **Rung 2 — Honest-booster leaf indices:** e_i = XGBoost honest-booster leaf indices (fixed tree structure). **Result: Δ = 0** — honest leaves don't reflect the scaffold gate. No layer structure survives to the SBM.
- **Rung 3 — Behavioral fingerprint (the one that works):**
  - `e_i = [f_scaffold(x_i), f_scaffold(x_i + δ₁), …, f_scaffold(x_i + δ_K)]` ∈ R^{K+1}
  - One gate-exposed anchor (x_i real → biased model) + K gate-hidden refinements (perts OOD → innocuous model)
  - **Result: Δ = +0.048** (6σ), C-ratio = 2.51 — first non-null pEx effect in entire campaign
- **Why rung 3 fires:** Strict LIME OOD detector routes 0% of σ=0.1·feat_std perts as real → all K refinements go to innocuous. Anchor is always biased. Pattern: one-biased-anchor + K-smooth-refinements groups same-layer points in cosine space.

**Speaker notes:** The encoder-ladder design motivates why rung 1 and 2 were tried first — this is a systematic empirical search. Rung 3 is the direct signal of the scaffold's routing mixture. Source: `docs/sbm_explainer/design.md`.

---

## Slide 28 — Exp 9 Housing Headline

**Layout:** Two-column: table left, bullets right.

**Bullets / Content:**
- 200 test points, 8-NN cosine graph on rung-3 Gaussian fingerprint (K=50 perts, σ=0.1·feat_std)
- Layer: pct_black_latino > 0.25 (24/200 protected, 176/200 not-protected)
- Source: `results/sbm_explainer/fingerprint_sbm/summary.json` and `results/sbm_explainer/seedscale_gauss_lime/results.json`

| Encoder | Scaffold | log BF | Plain NMI | pEx NMI | Δ | C-ratio |
|---------|----------|--------|-----------|---------|---|---------|
| Gaussian | LIME | +145 | 0.287 | **0.334** | **+0.048** | **2.51** |
| Gaussian | SHAP | −2 | 0.004 | 0.003 | −0.001 | 0.99 |
| Targeted | LIME | +2 | 0.063 | 0.063 | 0.000 | 1.16 |
| Targeted | SHAP | −3 | 0.012 | 0.012 | 0.000 | 1.01 |

- **20-seed CI:** Δ = 0.043 ± 0.018 (SEM 0.004), 95% CI [+0.017, +0.076], p = 1.8×10⁻⁹
- **Single-model sanity check** (all null — signal requires the scaffold routing mixture):

| Configuration | Plain NMI | pEx NMI | Δ | C-ratio |
|---|---|---|---|---|
| Honest XGBoost | 0.007 | 0.009 | +0.002 | 1.03 |
| BiasedModel only | 0.004 | 0.015 | +0.011 | 1.04 |
| InnocuousModel only | 0.007 | 0.015 | +0.007 | 1.06 |
| **Scaffold (LIME detector)** | **0.287** | **0.334** | **+0.048** | **2.50** |

**Speaker notes:** The sanity check is critical — the null for all single-model configs rules out that the signal is about the biased model or the innocuous model individually. It is specifically a scaffold-routing-mixture detector.

---

## Slide 29 — Four-Dataset Replication

**Layout:** Table-only, centered.

**Bullets / Content:**
- Same pipeline (rung-3 Gauss fingerprint → 8-NN cosine → finite-K pEx, K=4) across three additional Slack scaffolds
- σ swept over {0.1, 0.3, 1.0}; best cell shown per dataset
- Source: `results/sbm_explainer/{compas_fingerprint,german_fingerprint,cc_fingerprint}/` and `docs/sbm_explainer/followup_directions.md`

| Dataset | Best cell | Plain NMI | pEx NMI | Δ | C-ratio |
|---------|-----------|-----------|---------|---|---------|
| Housing (regression) | Gauss + LIME | 0.287 | 0.334 | **+0.048** | 2.51 |
| Slack COMPAS | Gauss + SHAP, σ=0.1 | 0.303 | 0.340 | **+0.038** | 1.94 |
| Slack German | Gauss + LIME, σ=1.0 | 0.115 | 0.172 | **+0.057** | 1.36 |
| Slack German | Gauss + SHAP, σ=1.0 | 0.048 | 0.119 | **+0.071** | 1.05 |
| CC psi_two (XOR) | Gauss + LIME, σ=1.0 | 0.124 | 0.699 | **+0.575** | 1.43 |

- **Invariant:** Δ > 0 iff OOD detector produces a partial routing mixture
- **Role reversal** (COMPAS SHAP fires, not LIME): detector-manifold calibration varies by dataset, not an explainer-specific property
- **XOR innocuous models** (CC psi_two) amplify signal — less-smooth innocuous → more internally coherent refinements

**Speaker notes:** Four datasets, regression + 3 classifications, feature dimension 11 → 102. The invariant across all of them is the routing mixture — it's not about which explainer or which dataset.

---

## Slide 30 — Mechanism — Anchor-Dominance Erosion

**Layout:** Two-column: table + bullets left, figures right.

**Bullets / Content:**
- **Mechanism:** Cosine decomposition of e_i·e_j / (|e_i||e_j|) = [m_i·m_j / (s_i·s_j)] + [a_i·a_j / (K·s_i·s_j)] + O(1/√K)
  - a = anchor term (O(1)); m = refinement-mean field (O(1)); K = fingerprint size
  - At small K: anchor dominates → plain SBM recovers layer via anchor coordinate
  - At large K: anchor weight → O(1/K), refinement mean field → O(1) and is layer-INDEPENDENT → plain collapses; pEx holds via conditional prior
- **K-sweep on CC psi_two** (LIME, σ=1.0):

| K | Plain NMI | pEx NMI | Δ | C-ratio |
|---|---|---|---|---|
| 5 | **0.333** | 0.334 | +0.001 | **21.4** |
| 10 | 0.187 | 0.307 | +0.120 | 3.24 |
| 20 | 0.082 | **0.555** | **+0.473** | 1.25 |
| 40 | 0.050 | 0.491 | +0.441 | 1.18 |
| 80 | 0.020 | 0.426 | +0.406 | 1.05 |

- **Tanh probe** (β=5 Δ = +0.056, 95% CI [+0.031, +0.080] at n=10 seeds): plain NMI degrades 0.73 → 0.39 as β increases; pEx stays flat at 0.65–0.73 → second mechanism confirmation
- Source: `results/sbm_explainer/k_scaling/` and `results/sbm_explainer/tanh_seedscale/`

**Figure (left):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/docs/sbm_explainer/figures/k_scaling.png`
   _(plain vs pEx NMI across K ∈ {5…80} on all four datasets.)_

**Figure (right):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/docs/sbm_explainer/figures/tanh_sweep.png`
   _(tanh-Bernoulli sweep, plain collapses as β increases, pEx holds flat.)_

**Speaker notes:** The K-sweep is the primary mechanism test; the tanh probe is the robustness confirmation. Together they establish that pEx's advantage is specifically in the K-large regime where the refinement mean field dominates. Source derivation: `docs/sbm_explainer/cosine-expression.md` §3.1.

---

## Act V — §2 Primitives + Block Characterization (Slides 31–38)

---

## Slide 31 — The Three §2 Outputs

**Layout:** Single-column bullets.

**Bullets / Content:**
- The pEx-SBM paper proposal §2 promises three concrete explanation outputs beyond point-estimate block assignment:
  1. **Ψ_{h,k} matrix:** For layer h and block k, posterior probability a node in layer h is assigned block k — shows how layers distribute across blocks
  2. **Pairwise co-clustering posterior P(z_i = z_j | G, ℓ):** For pairs of nodes, Bayesian probability they share a block — shows within-layer vs across-layer coherence
  3. **Per-node uncertainty H_i:** Shannon entropy over per-node block assignment posterior — identifies gate-boundary / ambiguous nodes
- All three computed from housing Gauss+LIME run with pct_black_latino > 0.25 layer, K=4, N_iter=3000, burn_in=1000, 2000 post-burn-in samples
- Source: `results/sbm_explainer/pex_outputs/`, `docs/sbm_explainer/pex_outputs_summary.md`

**Speaker notes:** This slide is the transition from "detector works" to "detector is interpretable." The §2 primitives are the explanation layer on top of the detection result.

---

## Slide 32 — Ψ Matrix (housing, pct_black_latino layer, K=4)

**Layout:** Two-column: table + bullets left, figure right.

**Bullets / Content:**
- Source: `results/sbm_explainer/pex_outputs/psi_matrix.npy`, `docs/sbm_explainer/pex_outputs_summary.md`
- Layer composition: 176 not-protected (88%), 24 protected (12%)

| Layer | Block 0 | Block 1 | Block 2 | Block 3 |
|-------|---------|---------|---------|---------|
| not_protected (pct_black_latino ≤ 0.25) | 0.093 | 0.124 | 0.290 | **0.493** |
| protected (pct_black_latino > 0.25) | 0.067 | 0.209 | **0.524** | 0.201 |

- **Layer-disassortative structure:** Protected layer concentrates in Block 2 (ψ=0.524); not-protected concentrates in Block 3 (ψ=0.493) — dominant blocks differ
- Protected Block 2 ψ=0.524 > not-protected Block 2 ψ=0.290 — the protected layer is detectably concentrated
- Weaker separation than sqft_lot run (which had near-zero cross-layer mass in dominant blocks) — consistent with pct_black_latino being a more diffuse signal

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_psi_heatmap.png`
   _(Ψ_{h,h'} posterior mean and CI-width heatmaps; rows = layers, cols = blocks.)_

**Speaker notes:** The disassortative structure (different dominant blocks for the two layers) is the structural signature of scaffold-attack detection. This is what auditors would report: "protected nodes cluster in Block 2, not-protected in Block 3."

---

## Slide 33 — Pairwise Co-Clustering

**Layout:** Two-column: bullets + table left, figure right.

**Bullets / Content:**
- Random 20-node subset (seed=42): 18 not-protected, 2 protected
- 2000 post-burn-in samples from aligned Gibbs trace
- Source: `results/sbm_explainer/pex_outputs/coclust_posterior.npy`, `docs/sbm_explainer/pex_outputs_summary.md`

| Pair type | Mean P(z_i = z_j) | Notes |
|-----------|------------------|-------|
| Within-layer pairs | **0.312** | Same pct_black_latino class |
| Across-layer pairs | **0.237** | Opposite classes |
| Ratio within / across | **1.319** | ~32% excess co-clustering within layer |

- Within-layer pairs are ~32% more likely to share a block than across-layer pairs
- Consistent with Exp 9 direction: pEx-SBM's layer-conditioned prior preferentially clusters same-layer nodes
- Lower ratio than sqft_lot run (1.50): pct_black_latino is a weaker fingerprint signal — but direction is consistent with Δ NMI = +0.048

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_coclust_matrix.png`
   _(20×20 pairwise co-clustering posterior matrix sorted by layer; diagonal-block structure visible.)_

**Speaker notes:** The co-clustering ratio (1.319 vs 1.0 null) is the Bayesian analog of the Δ NMI — it shows that the pEx prior is preferentially grouping same-layer nodes, not just assigning blocks randomly.

---

## Slide 34 — Per-Node Uncertainty

**Layout:** Two-column: bullets left, figure right.

**Bullets / Content:**
- Per-node Shannon entropy over block assignment posterior for the 20-node subset
- Source: `results/sbm_explainer/pex_outputs/node_entropies.npy`, `docs/sbm_explainer/pex_outputs_summary.md`
- Entropy range: **[≈0.000, 0.672]** nats (max possible = ln(4) = 1.386 nats)
- Entropy mean: **0.117** nats

| Category | Count | Criterion |
|----------|-------|-----------|
| Confidently assigned | **9/20** | H < 0.001 (effectively zero entropy) |
| High-uncertainty | **4/20** | H > 0.3 nats |
| Highest-entropy node | 1 | H = 0.672, Block 0 |

- All 4 high-entropy nodes sit on the Block 2 periphery — gate-boundary candidates
- Posterior is well-mixed (not smeared by label-switching) — sampler is reliable
- **H-DP connection:** These 4 nodes are the H-DP gate-boundary cluster candidates from `docs/sbm_explainer/cosine-expression.md`

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_node_uncertainty.png`
   _(Per-node entropy bar chart + stacked block-posterior bar chart for 20-node subset.)_

**Speaker notes:** 9/20 nodes confidently assigned gives auditors clean examples to work with; the 4 high-entropy nodes mark the OOD boundary where the scaffold's routing is genuinely ambiguous.

---

## Slide 35 — Block Characterization

**Layout:** Two-column: table left, figure right.

**Bullets / Content:**
- Modal block assignment over aligned trace (N_iter=2000, burn=500, seed=0)
- Source: `results/sbm_explainer/pex_outputs/block_labels.npy`, `docs/sbm_explainer/block_characterization.md`

| Block | n | % protected | Semantic label |
|-------|---|-------------|----------------|
| **0** | 20 | 10% | Young central properties, gate-boundary / residual |
| **1** | 27 | 19% | Suburban large-lot, below-anchor fingerprint |
| **2** | 64 | 19% | Protected-dominant western corridor (**scaffold attack main target**) |
| **3** | 89 | 6% | Northern older homes, not-protected majority, anchor-above-tail regime |

- Block 2 captures 12/24 (50%) of protected nodes — the policy-relevant block
- Block 3 is the "safely unaffected" regime (high anchor, not-protected majority)
- Block 2 has highest protected ψ (0.524) and clearest geographic coherence (western corridor, low UTM_Y)

**Figure (left):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_block_feature_profiles.png`
   _(z-score feature profiles for each block; 4 subplots.)_

**Figure (right):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_block_geography.png`
   _(UTM scatter of all 200 nodes, colored by block; X markers = protected nodes.)_

**Speaker notes:** Block 2 is where the scaffold attack is most consequential: protected nodes here have their pct_black_latino attribution hidden by the LIME explainer.

---

## Slide 36 — Block 1 Anchor-Below-Tail Flip

**Layout:** Two-column: bullets left, figure right.

**Bullets / Content:**
- **Novel finding:** Block 1 is the only block where the fingerprint tail mean **exceeds** the anchor
  - Block 1 anchor = **5.303** log_price (lowest of all 4 blocks)
  - Block 1 tail mean = **5.643** → **+0.34 log_price flip** (tail exceeds anchor)
  - Compare: Block 3 anchor = 5.750 (highest), tail = 5.608 → anchor > tail (normal regime)
- **Mechanism:**
  - Real input x_i → OOD detector fires → routed to **biased model** → LOW prediction (penalty applied)
  - Gaussian perturbations x_i + δ → OOD detector does NOT fire → routed to **innocuous model** → HIGHER prediction
  - The explainer sees a "healthy neighborhood" (innocuous-model predictions) around a penalized query
- Block 1 feature profile: large lots (sqft_lot z=+1.12), western/southern location (UTM_Y z=−0.54), 19% protected
- Block 1 adversarial attribution: sqft_living (0.038) and UTM_Y (0.037) are top — pct_black_latino masked
- This laundering mechanism is NOT visible in Exp 3 aggregate analysis — block-level discovery

**Figure:** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/pex_outputs/figures/fig_block_fingerprint_shapes.png`
   _(4 fingerprint shape lines on shared axes; Block 1 is the only line where tail > anchor. Block 3 shows opposite sign.)_

**Speaker notes:** This is the mechanistic discovery that the aggregate Exp 3 analysis cannot show — specific nodes are being laundered because their perturbation cloud escapes to the innocuous model while the real input is penalized. Source: `docs/sbm_explainer/block_characterization.md` Block 1 section.

---

## Slide 37 — Slack-Suite §2 + Block Outputs (Cross-dataset replication)

**Layout:** Two-column: table left, cross-dataset Ψ grid right.

**Bullets / Content:**
- Applied housing-equivalent pipeline (Gibbs K=4, 3000 iters / 1000 burn-in, greedy-Hungarian alignment) to COMPAS, German, CC psi_two. Graphs and fingerprints were already cached.
- **Headline: the "Block 1 fingerprint flip" from housing is NOT housing-specific. Flip blocks (anchor < tail) are present in every Slack dataset.**

| Dataset | Cell | n_prot/N | Ψ disassortative? | Flip block? | Within/Across ratio | Entropy (H<0.001 / H>0.3) |
|---------|------|----------|-------------------|-------------|---------------------|---------------------------|
| Housing | Gauss+LIME σ=0.1 | 24/200 | Yes (B2 vs B3) | B1 (+0.34) | 1.32 | 9/20 / 4/20 |
| COMPAS | Gauss+SHAP σ=0.1 | 122/200 | **Yes (B1 vs B3)** | B1 + B3 | **4.56** | 17/20 / 0/20 |
| German | Gauss+SHAP σ=1.0 | 64/100 | Partial (both argmax B0; secondary distribution differs) | B0 + B1 | 2.31 | 6/20 / 0/20 |
| CC psi_two | Gauss+LIME σ=1.0 | 123/200 | **Maximal (L1 → B0 at 96.5%)** | **B0 anchor=0 (nuclear flip)** | **49.78** | 7/20 / 3/20 |

- **CC psi_two is the strongest case yet.** 119/123 racePctWhite>mean nodes concentrate in Block 0. Block 0 has anchor = 0 (perfect innocuous-routing under the XOR scaffold) and tail = 0.491. Within/across co-clust ratio = **49.8**. Explains the Δ NMI = +0.575 from Exp 9.
- **German is the weakest cell** — both layers argmax at B0, but secondary distributions differ (L0: 91% in {B0,B1}; L1: 36/17/22/25% across all 4). Consistent with Exp 9's C-ratio = 1.05 (middle-regime boundary).
- **Caveat:** per-block adversarial-attribution figures substitute feature z-scores because the Slack scaffold runs didn't cache per-node SHAP/LIME arrays separately from fingerprints. Documented for follow-up in `docs/sbm_explainer/slack_suite_summary.md`.

**Figure (right):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/docs/figures/fig_slack_suite_psi_grid.png`
   _(2×2 grid: housing + 3 Slack Ψ heatmaps with shared colormap; visually confirms disassortativity replicates on 3/4 datasets.)_

**Per-dataset figure folders (for reference — pick 1 dataset to show inline if time allows):**
- `results/sbm_explainer/slack_outputs/compas/figures/fig_block_fingerprint_shapes.png`
- `results/sbm_explainer/slack_outputs/german/figures/fig_block_fingerprint_shapes.png`
- `results/sbm_explainer/slack_outputs/cc_psi_two/figures/fig_block_fingerprint_shapes.png`

**Speaker notes:** The flip replicates universally — that's a significant upgrade from yesterday's "Block 1 is housing-specific." CC psi_two's anchor=0 block is the geometrically purest version; use it as the featured case if a reviewer questions whether the housing result is an artifact. German's partial disassortativity aligns with its Exp 9 C-ratio of 1.05 — consistent with the three-regimes story from Exp 12 (German sits at the middle-regime boundary).

---

## Slide 38 — Exp 12 §6.2 Spectrum Sweep

**Layout:** Two-column: table left, figures right.

**Bullets / Content:**
- §6.2 promise: "pEx-SBM recovers audit-relevant block structure across full autocorrelation range"
- Run for 8 features spanning I ∈ [0.09, 0.85]; waterfront skipped (4/200 test points above P75 — degenerate)
- pct_black_latino row corrected: Plain NMI = **0.287**, pEx NMI = **0.334**, Δ = **+0.048** (prior bug reported 0.000/0.048; fixed 2026-04-23, source: `docs/sbm_explainer/spectrum_sweep_summary.md` Correction Notice)

| Feature | Moran's I | Plain NMI | pEx NMI | Δ NMI | C-ratio | Layer-disassort? |
|---------|-----------|-----------|---------|-------|---------|-----------------|
| condition | 0.090 | 0.0649 | 0.0644 | −0.0004 | 0.869 | No |
| view | 0.242 | 0.8771 | 0.8771 | 0.0000 | — (sat.) | Yes |
| sqft_living | 0.291 | 0.0138 | 0.0073 | −0.0066 | 0.984 | No |
| grade | 0.365 | 0.0140 | 0.0202 | +0.0063 | 1.001 | Yes |
| sqft_lot | 0.469 | 0.0274 | 0.0414 | +0.0139 | 1.204 | Yes |
| pct_latino | 0.756 | 0.0113 | 0.0116 | +0.0002 | 0.940 | No |
| pct_black_latino | 0.834 | **0.287** | **0.334** | **+0.048** | 1.319 | Yes |
| pct_black | 0.850 | 0.0125 | 0.0142 | +0.0017 | 0.891 | No |

- **Three regimes:**
  - Saturation (view): plain NMI ≈ 1, no room for pEx to improve
  - **Middle regime (grade, sqft_lot, pct_black_latino):** C-ratio 1.0–1.3 → pEx fires
  - Collapse/diffuse (condition, sqft_living, pct_latino, pct_black): C-ratio < 1 → null
- **Revised §6.2 claim:** "pEx recovers in the middle behavioral regime, identified by C-ratio ∈ [1, 3]" — not a universal claim across all I

**Figure (top):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/spectrum_sweep/figures/fig_delta_nmi_vs_morans_i.png`
   _(Scatter of Δ NMI vs Moran's I; no monotone trend — regime-driven, not I-driven.)_

**Figure (bottom):** `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/results/sbm_explainer/spectrum_sweep/figures/fig_regime_diagnostic.png`
   _(C-ratio vs Plain NMI scatter, colored by Moran's I; green band C-ratio ∈ [1,3] captures Δ>0 features.)_

**Speaker notes:** The regime condition (C-ratio ∈ [1,3]) is the auditor-actionable pre-diagnostic: compute plain SBM first; if C-ratio is in this range, run pEx to recover the protected layer. Source: `docs/sbm_explainer/spectrum_sweep_summary.md`.

---

## Act VI — Gaps, Asks, Next Steps (Slides 39–45)

---

## Slide 39 — Theoretical Gaps

**Layout:** Single-column numbered list.

**Bullets / Content:**
1. **pEPPF → explanation-functional CIs** — an additional lemma is needed to push the Exchangeable Partition Probability Function (pEPPF) through the Ψ functional to give confidence intervals on the explanation outputs (Ψ_{h,k}, co-clustering ratio). Current CIs are empirical posterior quantiles, not derived from the pEPPF directly.

2. **Projectivity must be stated for H-DP/H-PY specifically** — Orbanz (2017) Thm 3 gives projectivity for infinite-dimensional consistent families; finite-K pEx is NOT automatically projective. A formal statement is needed for the H-DP and H-PY variants specifically, as the proposal §2 implies projectivity holds uniformly.

3. **DMMS SNR threshold** — the SBM detection/recovery theorem (λ²+μ²/γ > 1 in the DMMS model) is a hard ceiling: recovery is information-theoretically impossible below it. The recovery theorem in the proposal must be explicitly scoped to the regime where SNR exceeds this threshold; it cannot be stated universally.

**Speaker notes:** These three gaps are the load-bearing theoretical items for a NeurIPS-main submission. A workshop or TMLR submission can proceed with the empirical results and defer the formal bound.

---

## Slide 40 — Empirical Gaps

**Layout:** Single-column numbered list.

**Bullets / Content:**
1. **H-DP K̂ ≈ 10 vs binary arity mismatch** — H-DP infers ~10 blocks against a binary layer, mechanically depressing NMI. θ₀ sweep over {0.25, 0.5, 1.0, 2.0} (3 seeds each) confirms K̂ ≈ 10 is stable; not a bad-prior artifact. The arity mismatch is genuine: H-DP genuinely sees 10 blocks (7/10 are ≥ 90% layer-pure; 2 are gate-boundary at purity 0.56/0.57). Open: does H-DP add value beyond finite-K as a "lead" or is it a supporting result?

2. **Penalty-magnitude robustness sweep pending** — Δ NMI = +0.048 is established at scaffold penalty = 0.30 log-price. Robustness at {0.10, 0.30, 0.50} not yet run. Does Δ survive, grow, or collapse at weaker attacks?

3. **Predictive test on H-DP gate-boundary clusters** — do gate-boundary nodes (Block 2 periphery, H > 0.3) have higher adversarial sensitivity on held-out data? Would move rung-3 + H-DP from "interpretive diagnostic" to "predictive method."

4. **SNR-regime characterization per scaffold** — which of the four datasets sits above vs below the DMMS SNR threshold? Characterizing this per-dataset would ground the recovery theorem empirically.

**Speaker notes:** Items 1 and 2 are addressable before a paper submission (1–2 days each). Items 3 and 4 are longer-horizon (1–2 weeks).

---

## Slide 41 — Three Asks for Peng

**Layout:** Single-column, numbered list with sub-bullets.

**Bullets / Content:**
1. **Lead story: finite-K Δ vs H-DP refinement?**
   - Finite-K: clean story, 4-dataset replication, 6σ significance, K-sweep mechanism. Simpler narrative.
   - H-DP: richer (K̂≈10, gate-boundary clusters, per-cluster attribution profiles), but NMI number is lower (0.261 vs 0.334 finite-K) due to arity mismatch. Is the refinement story worth the added complexity?

2. **Workshop (NeurIPS XAI, July deadline) vs NeurIPS-main?**
   - Workshop: four-dataset + K-sweep is sufficient. Can submit without the formal recovery-probability bound.
   - NeurIPS-main: needs the plain-SBM modular-recovery-probability bound as a function of (m-variability, K, layer-alignment). Significant additional theory work.
   - TMLR: rolling deadline, supports longer empirical papers, no theory requirement.

3. **Theoretical ordering: which gap first?**
   - Option A: SNR threshold scoping → projectivity → functional pushforward CIs (bottom-up)
   - Option B: functional pushforward CIs → projectivity → SNR scoping (top-down from the §2 outputs)
   - Option C: delegate functional-pushforward lemma to a Bayesian-nonparametrics co-author; focus empirically

**Speaker notes:** These are the three open questions the April 23 meeting is designed to answer. Come with a preference on each and be ready to explain the tradeoffs.

---

## Slide 42 — What's Established

**Layout:** Single-column numbered list.

**Bullets / Content:**
1. **Scaffold works regardless of Moran's I** — OOD accuracy flat 99.3–100% across I ∈ [0.05, 0.85]; the attack mechanism is the routing architecture, not spatial autocorrelation of the trigger feature

2. **Rung-3 + pEx is the first detector with non-null layer signal** — all prior encoders (attribution vectors, honest-booster leaves) produce Δ ≈ 0; rung-3 behavioral fingerprint is the only encoder that places the posterior in the middle regime (C-ratio 2.51) where pEx's layer-specific prior acts

3. **Block-level mechanism discovery** — Block 1 anchor-below-tail flip (+0.34 log_price) identifies nodes where the scaffold's OOD detector fires on the real input but not on its perturbation cloud. This is the operational laundering mechanism, not visible in aggregate attribution analysis

4. **All §2 primitives computed with matched figures** — Ψ matrix, pairwise co-clustering (ratio 1.319), per-node uncertainty (9/20 confident), block characterization (4 semantically labeled blocks), block fingerprint shapes, block geography. All sourced from `results/sbm_explainer/pex_outputs/`

**Speaker notes:** This is the summary of what can be put in a paper today, without additional experiments. Items 1–4 correspond to the four sections of a potential workshop submission.

---

## Slide 43 — Timeline Sketch

**Layout:** Single-column timeline.

**Bullets / Content:**
- **May 2026:** Complete penalty sweep (scaffold penalty 0.10/0.30/0.50); complete H-DP K-sweep with arity-matched comparison; draft §1–§4 of paper (setup, attack, detection, mechanism)
- **June 2026:** Complete Slack-suite §2 outputs (COMPAS/German/CC block characterization); complete predictive test on gate-boundary clusters; draft §5–§6 (block characterization, spectrum sweep)
- **July 2026:** NeurIPS XAI workshop deadline (if workshop track chosen) — submit with four-dataset replication + K-sweep + block characterization. No formal bound required.
- **August 2026:** If NeurIPS-main track: begin formal recovery-probability bound. Co-author search for pEPPF lemma.
- **October 2026:** NeurIPS-main submission deadline (if NeurIPS track chosen); TMLR rolling (any time).

NOTE: Dates above are estimates based on known deadlines; verify workshop deadline against NeurIPS 2026 XAI workshop call.

**Speaker notes:** The May–June timeline is realistic given current codebase state (all pipelines are running; it's extraction and writing). The July workshop submission is achievable without the formal bound.

---

## Slide 44 — Compute + Collaborator Asks

**Layout:** Single-column bullets.

**Bullets / Content:**
- **GPU access (summer UROP):** Transformer case study would require running the rung-3 fingerprint pipeline on a large-language-model scaffold. Current CPU experiments run in 20–40 min on a laptop; a 10B-parameter transformer scaffold would require GPU compute. Ask: CSAIL / MIT UROP GPU allocation for summer 2026.

- **Bayesian-nonparametrics co-author:** The pEPPF → explanation-functional CIs lemma requires expertise in partition probability functions and projective limits. Ask: Peng to introduce contact with a BNP faculty member (CSAIL or elsewhere) who could contribute or validate the theoretical lemma.

- **COMPAS/German §2 extraction:** Parallel agent running; outputs will land in `results/sbm_explainer/{compas,german,cc_psi_two}/`. If not complete by meeting, ask Peng whether the 4-dataset replication NMI numbers are sufficient without per-dataset block characterization.

**Speaker notes:** The GPU and BNP co-author asks are the two items that cannot be self-resolved. Everything else in the empirical gap list is executable with existing hardware and codebase.

---

## Slide 45 — Q&A

**Layout:** Centered, minimal.

**Bullets / Content:**
- Questions?
- Repo: `geospatial-xai-attacks/` (entry point: `docs/sbm_explainer/README.md`)
- All numeric results traceable to JSON files under `results/`

**Speaker notes:** Have `project_summary.md` open in a terminal for quick numeric lookups during Q&A.

---

## Figures Referenced

| Figure (absolute path) | Verified? | Slides |
|---|---|---|
| `.../results/figures/demographics_combined.png` | YES | 3 |
| `.../results/figures/honest_vs_adv_shap.png` | YES | 6 |
| `.../results/figures/honest_vs_adv_lime.png` | YES | 7 |
| `.../results/robustness/figures/heatmap.png` | YES | 8 |
| `.../results/worst_case/figures/eps_sweep.png` | YES | 9 |
| `.../results/no_location/figures/ranking_comparison.png` | YES | 10 |
| `.../results/morans_i/morans_i_barplot.png` | YES | 13 |
| `.../results/feature_sweep/sweep_plot.png` | YES | 14 |
| `.../results/scaffold_geologic_hazard/figures/fig_attribution_comparison.png` | YES | 21 |
| `.../results/scaffold_geologic_hazard/figures/fig_geoshap_reversal.png` | YES | 22 |
| `.../docs/figures/fig_honest_vs_adv_off_baseline.png` | YES | 24 |
| `.../docs/sbm_explainer/figures/k_scaling.png` | YES | 30 |
| `.../docs/sbm_explainer/figures/tanh_sweep.png` | YES | 30 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_psi_heatmap.png` | YES | 32 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_coclust_matrix.png` | YES | 33 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_node_uncertainty.png` | YES | 34 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_block_feature_profiles.png` | YES | 35 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_block_geography.png` | YES | 35 |
| `.../results/sbm_explainer/pex_outputs/figures/fig_block_fingerprint_shapes.png` | YES | 36 |
| `.../results/sbm_explainer/spectrum_sweep/figures/fig_delta_nmi_vs_morans_i.png` | YES | 38 |
| `.../results/sbm_explainer/spectrum_sweep/figures/fig_regime_diagnostic.png` | YES | 38 |
| `.../results/worst_case/figures/feature_destabilization.png` | YES (not used) | — |
| `.../results/worst_case/figures/worst_vs_avg_robustness.png` | YES (not used) | — |
| `.../results/paper_figures/fig1_scaffold_attack.png` | YES (not used) | — |
| `.../results/paper_figures/fig4_robustness_heatmap.png` | YES (not used) | — |
| `.../results/paper_figures/fig5_worst_case_sweep.png` | YES (not used) | — |

**MISSING figures (expected by narrative arc, not found on disk):**

| Expected path | Slide | Fallback |
|---|---|---|
| FIGURE MISSING — `results/sbm_explainer/pex_outputs/figures/fig_block_adv_attributions.png` is on disk but was not used | — | Use `fig_block_fingerprint_shapes.png` for slide 36 (already assigned) |
| `results/scaffold_geologic_hazard/figures/fig_feature_geography.png` | 21 | EXISTS on disk — add as second figure on slide 21 if space allows |
| `results/feature_sweep_a2/sweep_plot.png` | 15 | EXISTS on disk — use for Exp 8b slide |
| `docs/sbm_explainer/figures/tanh_seedscale.png` | — | EXISTS on disk (not needed for these slides) |

All base paths are: `/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks/`

---

## Numeric Values Flagged as Approximate or Estimated

| Value | Slide | Source | Flag |
|---|---|---|---|
| Honest GeoSHAP for Exp 3 UTM_Y (0.013) | 6 | `project_summary.md` Exp 3 table footnote | Row is honest GeoSHAP from 10-feat model (no pct_black_latino), so table cell is undefined by construction |
| GeoSHAP Exp 10d trigger attribution (0.011) | 22 | `docs/comparison_honest_vs_adv_off_baseline.md` footnote | Only 20 test points for GeoSHAP; value approximate |
| Timeline months (slide 43) | 43 | Estimated from proposal §9 language | Verify NeurIPS XAI 2026 workshop deadline against official call |
| Exp 8b Spearman ρ = −0.43, p = 0.40 | 15 | `results/feature_sweep_a2/` — value sourced from `project_summary.md` Exp 8b section; original JSON not read directly | FLAGGED: confirm from `results/feature_sweep_a2/sweep_summary.json` or equivalent |
