# Geospatial XAI Attacks — Results Summary

**Robert Romani · UROP with Peng Luo · April 2026**

Full technical detail: `docs/project_overview.md`. This document is a narrative walkthrough of key results for meeting discussion.

| Experiment | Date run |
|---|---|
| Baseline XGBoost + honest explanations | April 2, 2026 |
| Adversarial scaffold attack (first run) | April 2, 2026 |
| Scaffold re-run + no-location ablation | April 16, 2026 |
| Robustness analysis (3×4 matrix) | April 16, 2026 |
| Worst-case x* sweep | April 16, 2026 |
| Moran's I computation | April 21, 2026 |
| Feature sweep (in progress) | April 21, 2026 |

---

## What we set out to do

Two questions drive this project:

1. **Can post-hoc explainers (LIME, SHAP, GeoShapley) be fooled into hiding coordinate-based discrimination in a housing valuation model?** — extending Slack et al. (2020) into a geospatial setting.
2. **Even without an adversarial attack, how stable are geospatial explanations under realistic perturbations?** — adapting Vascotto et al. (2025) to the spatial domain and asking whether a deliberate but non-adversarial actor can exploit that instability.

The domain: 3,000 King County housing transactions with UTM coordinates and ACS census demographics, modeled with XGBoost (R² = 0.89).

---

## Experiment 1 — Adversarial Scaffold Attack (Fooling LIME and SHAP)

**Setup.** We built a biased model that wraps the honest XGBoost baseline and subtracts a 0.30 log-price penalty (≈ 26% price reduction in linear space) for homes in tracts where `pct_black_latino > 25%`. An OOD detector then routes real data to the biased model and explainer perturbations to an innocuous surrogate trained only on safe structural features. The attack is invisible to a reviewer reading the feature attributions.

**Key numbers.** Mean |SHAP attribution| for each feature, honest vs. adversarial (200 test points):

| Feature | Honest SHAP | Adv SHAP | Shift |
|---|---|---|---|
| UTM_Y | 0.118 | 0.050 | **−57%** |
| sqft_living | 0.068 | 0.082 | +21% |
| grade | 0.049 | 0.075 | +53% |
| pct_black_latino | — | 0.035 | (suppressed) |

Under **LIME**, UTM_Y barely moves (−3%) and `pct_black_latino` remains visible at attribution 0.108 — LIME is substantially more resistant in this setting. The GeoSHAP result is the starkest: UTM_Y collapses to 0.009 (−93%), because GeoSHAP's per-point geographic backgrounds keep every coalition query inside the real-data manifold, so the OOD detector routes every query to the biased branch.

**Main finding.** Geographic awareness in GeoShapley is not a defense — it makes the attack easier, not harder. LIME resists better than both SHAP variants because Gaussian perturbations occasionally land in minority tracts and expose the biased model's behavior.

**OOD detector accuracy:** SHAP 99.5%, LIME 100%. Fidelity (scaffold ≈ biased model on real data): LIME r = 0.9996, SHAP r = 0.911. The biased model amplifies the demographic-price correlation from Pearson r = −0.51 (honest) to r = −0.73 (biased).

---

## Experiment 2 — No-Location Ablation

**Motivation.** A practitioner might remove UTM coordinates to "eliminate the spatial signal." Does it work?

**Result.** Dropping UTM_X and UTM_Y cuts R² from 0.89 to 0.62 — coordinates contribute roughly 27 R²-points, about a third of total explanatory power. They do not fully displace into proxies under SHAP (there is no single proxy that absorbs the spatial signal cleanly), but `age` rises to the third most-important feature globally, consistent with neighborhood vintage acting as a weak spatial proxy. LIME's rankings diverge sharply: `view`, `condition`, and `waterfront` all receive large weights — LIME's local linear surrogate upweights binary amenity features that generate large local prediction swings.

**Implication.** Removing coordinates from the model weakens it substantially without reliably transferring spatial attribution to a single interpretable proxy. The bias doesn't vanish; it disperses.

---

## Experiment 3 — Robustness Analysis (Non-Adversarial)

**Setup.** For 50 test points, we generated three neighborhood types and measured Spearman ρ between the reference point's attribution ranking and each neighbor's ranking (higher = more stable):

- **Random** — Gaussian noise (σ = 5% of training std per feature), off-manifold
- **Medoid** — Euclidean k-NN pool drawn from real training data, on-manifold in feature space
- **Spatial** — blends geographic + feature distance, on-manifold in both dimensions

**Results matrix** (mean Spearman ρ):

| Neighborhood | TreeSHAP | KernelSHAP | GeoSHAP | LIME |
|---|---|---|---|---|
| Random | **0.845** | 0.847 | 0.720 | 0.523 |
| Medoid | 0.515 | 0.531 | 0.336 | 0.473 |
| Spatial | 0.520 | 0.537 | 0.329 | 0.481 |

**Surprising finding.** Random (off-manifold) neighborhoods are *more* stable than on-manifold ones. Why: Gaussian perturbations stay close to the reference in the specific feature dimensions that XGBoost's tree splits care about, so rankings barely shift. Real on-manifold neighbors (medoid/spatial) can cross major split boundaries (e.g., `grade ≥ 8`) and completely flip the attribution ranking. GeoSHAP is the worst: its per-point background also varies across the neighborhood, adding a second source of variance on top of the attribution itself.

**Trustworthiness.** Under the on-manifold standard with threshold ρ ≥ 0.8: only 1 of 50 test points has a "trustworthy" explanation. Essentially no point in this housing model has stable feature rankings under realistic spatial perturbations.

**How Spearman ρ is computed here.** This is not a reimplementation of SHAP or LIME — it uses the standard `shap` and `lime` libraries throughout. The ρ values are purely a post-hoc measurement layered on top. For each reference house, the explainer (TreeSHAP, KernelSHAP, GeoSHAP, or LIME) is called normally and returns an attribution vector, e.g. `[UTM_Y: 0.12, sqft_living: 0.07, grade: 0.05, ...]`. Then the same explainer is called on each of the 50 neighbors. `scipy.stats.spearmanr` compares the two attribution *rankings* (not the raw values) and returns ρ. The whole experiment is just asking: if I explain house A, then explain house B next door, does the ranking of which features matter agree?

The one exception is GeoSHAP: because it builds a *per-point* geographic background (the 100 UTM-nearest training neighbors of each explained house), calling it on 50 neighbors means 50 separate KernelSHAP runs each with a different background. That cost is why GeoSHAP is capped at 20 test points and 20 neighbors in the matrix above.

---

## Experiment 4 — Worst-Case x* Attack (Novel)

**Goal.** Without any adversarial machinery, how bad can the explanation discrepancy get? Find the real training house, within geographic radius ε, that maximally disagrees with the reference explanation:

$$x^* = \arg\max_{x \in N_\varepsilon(y)} \|\text{attr}(x) - \text{attr}(y)\|_2$$

No synthetic perturbations. x* is always a real observed house.

**ε-sweep results** (50 reference points, TreeSHAP):

| ε (meters) | Mean worst-case L2 | Mean candidates |
|---|---|---|
| 500 m | 0.165 | 3.6 |
| 1,000 m | 0.228 | 12.4 |
| 2,000 m | 0.275 | 39.3 |
| 5,000 m | **0.382** | 150.0 |
| 10,000 m | 0.411 | 196.0 |

At ε = 5,000 m (roughly a neighborhood-to-neighborhood scale), the worst-case L2 shift is 0.382 — roughly **2.6× larger than the average spatial-neighborhood perturbation** (0.147 in Experiment 3). The exploited features are `grade` and `sqft_living`, not `UTM_Y` — the worst-case neighbor is chosen not because it is geographically unusual, but because it sits on the other side of a key structural split boundary.

**Implication.** A bad-faith user who can select which "comparable" house to point to does not need to synthesize data or train an adversarial model. They just need to search a plausible geographic radius — a task that is trivially feasible in any real estate context.

---

## What the experiments say together

| Question | Answer |
|---|---|
| Can a coordinate-based redlining bias be hidden from SHAP/GeoSHAP? | Yes — UTM_Y drops 57% (SHAP), 93% (GeoSHAP). LIME partially resists. |
| Does GeoSHAP's spatial awareness protect it? | No — it makes it *more* vulnerable. |
| Does removing coordinates neutralize the spatial signal? | No — costs 27 R²-points, spatial proxy leaks into `age` and amenity features. |
| Are geospatial explanations stable under on-manifold perturbations? | No — on-manifold neighborhoods are *less* stable than random noise; ~0/50 points trustworthy. |
| Can a non-adversarial user destabilize explanations? | Yes — worst-case plausible neighbor doubles average perturbation at 5km radius. |

---

---

## Experiment 5 — Moran's I Spatial Autocorrelation Survey (April 21, 2026)

**Goal.** Peng's April 16 ask: introduce a biased feature that is *not* highly spatially autocorrelated and compare whether the scaffold attack is equally effective. First step is to measure spatial autocorrelation (Moran's I) for every feature in the dataset.

**Method.** Moran's I implemented from scratch using a row-standardised 8-NN spatial weights matrix on UTM coordinates (no libpysal). For each feature:

$$I = \frac{z^T W z}{z^T z}, \quad z = x - \bar{x}$$

where W is row-standardised, so S₀ = n and the formula simplifies. Statistical significance via 499-permutation pseudo-p-value (`scipy` random permutations of x, measuring how often a shuffled I exceeds the observed value).

**Results** (all 11 non-coordinate features):

| Feature | Moran's I | Autocorrelation level |
|---|---|---|
| pct_black | +0.850 | very high |
| pct_black_latino | +0.834 | very high |
| pct_latino | +0.756 | high |
| sqft_lot | +0.469 | medium |
| grade | +0.365 | medium |
| age | +0.317 | medium |
| sqft_living | +0.291 | low–medium |
| view | +0.242 | low |
| bathrooms | +0.183 | low |
| condition | +0.090 | very low |
| waterfront | +0.051 | very low |

All features are significantly autocorrelated (p < 0.05 under permutation test). The spectrum is wide: demographic features cluster tightly by neighborhood (I ≈ 0.75–0.85) while structural features like `waterfront` and `condition` are nearly spatially random (I ≈ 0.05–0.09).

**5 features selected for sweep** (evenly spaced across the I spectrum): `waterfront` (0.051), `view` (0.242), `sqft_lot` (0.469), `pct_latino` (0.756), `pct_black` (0.850).

Output: `results/morans_i/morans_i_results.json`, `morans_i_barplot.png`, `sweep_features.txt`.

---

## Experiment 6 — Feature Sweep: Scaffold Attack vs. Moran's I (April 21, 2026)

**Goal.** Run the scaffold attack 5 times — once per selected feature spanning the Moran's I spectrum — and measure whether explainer susceptibility correlates with spatial autocorrelation. Original hypothesis: high Moran's I features are easier to hide from explainers because clumpy distributions let the OOD detector cleanly separate real vs. perturbed data.

**Method.** For each sweep feature:
1. Build a `BiasedModel` that subtracts 0.30 log-price from homes where `feature > 75th percentile`
2. Build an `InnocuousModel` trained on all features except the biased one and UTM coordinates
3. Train SHAP and LIME OOD detectors (same RF-based approach as Experiment 1)
4. Generate 100 adversarial explanations with each explainer
5. Measure **attribution visibility**: `mean_i(|adv[i, bias]|) / mean_i(sum_j|adv[i, j]|)` — the share of total |attribution| assigned to the biased feature

**Implementation note.** Fixed a shape mismatch bug: the honest XGBoost baseline was trained on exactly 10 features. Build a focused dataset per sweep run (10 baseline features + bias feature if not in baseline) to ensure consistent sizing. Mark each feature's role (structural vs. demographic) for interpretation.

**Results** (100 adversarial explanations per feature):

| Feature | Role | Moran's I | SHAP vis | LIME vis | % penalized | SHAP OOD | LIME OOD |
|---|---|---|---|---|---|---|---|
| waterfront | structural | 0.051 | 0.0135 | 0.106 | 0.8% | 99.3% | 100% |
| view | structural | 0.242 | 0.0703 | 0.160 | 9.7% | 99.3% | 100% |
| sqft_lot | structural | 0.469 | 0.134 | 0.175 | 25.3% | 99.3% | 100% |
| pct_latino | demographic | 0.756 | 0.136 | 0.190 | 24.7% | 99.4% | 100% |
| pct_black | demographic | 0.850 | 0.140 | 0.207 | 23.2% | 99.6% | 100% |

*Visibility = share of total |attribution| on the biased feature. Higher = harder to launder (more leakage).*

**Key findings — hypothesis disconfirmed, new pattern emerges:**

1. **OOD accuracy is flat across Moran's I spectrum.** ~99.3–99.6% SHAP, 100% LIME regardless of feature's spatial clustering. The attack's core mechanism (routing real vs. perturbed) does not depend on spatial autocorrelation. Moran's I does *not* predict attack feasibility.

2. **Visibility increases monotonically with Moran's I** — structural: 0.014 → 0.134 (SHAP), 0.106 → 0.175 (LIME); demographic: 0.136 → 0.140 (SHAP), 0.190 → 0.207 (LIME). Counterintuitive: *higher-I features are harder to launder*, not easier. More spatially clumped features leak *more* signal, not less.

3. **Two regimes, not one.** Structural features (in baseline) show low visibility because the attack hides by amplification: the innocuous model excludes them, so the scaffold amplifies their attribution on real inputs. Demographic features (off baseline) show higher visibility because they were never part of the honest model; laundering happens indirectly (bias bleeds into structural proxies), not by suppression of the biased feature itself. Visibility fraction captures both regimes in a unified metric.

4. **Binary features are under-penalized.** Waterfront's P75 = 0, so only 0.8% of test points are penalized. The attack has little signal to hide. A sweep with adjusted thresholds (e.g., median instead of P75 for near-constant features) or different features (e.g., `grade`, I=0.365) would strengthen the structural regime.

5. **Suppression metric is deprecated.** The old metric `(honest − adv) / honest` broke for demographic features (0/0 → 0) and was negative for structural features (they amplify, not suppress). Visibility fraction is well-defined everywhere and directly measures how much the attack leaks.

Output: `results/feature_sweep/sweep_summary.json`, `sweep_plot.png` (two-panel: visibility + raw attribution vs. Moran's I, role-colored).

---

## What we learned from April 16 → April 21

1. **Moran's I does not predict attack feasibility.** OOD accuracy flat across I spectrum. The hypothesis that high-I features are easier to fool explainers is **disconfirmed**. Instead: visibility increases with I (harder to hide, not easier) and attack mode depends on whether the feature is in the baseline model (structural vs. demographic regime).

2. **Framing shift.** Not "SHAP fails in geo" but "explainer robustness in geospatial settings depends on feature role and spatial clustering in unexpected ways." The attack is orthogonal to Moran's I at the OOD stage; any residual leakage grows with spatial homogeneity.

3. **pEx-SBM direction.** Spatial autocorrelation is *not* the key moderating variable (OOD is I-invariant). Instead, the distinction between on-baseline and off-baseline features and their interaction with spatial structure might be the framing for a follow-up. See `Notes/pEx-SBM Paper Proposal.md` for next steps.

---

## Next experiments to run

| Priority | Script | Estimated runtime | Output |
|---|---|---|---|
| 1 | `run_morans_i.py` | ~2 min | Ranked Moran's I table + 5 sweep features |
| 2 | `run_feature_sweep.py` | ~20–40 min | Suppression vs. Moran's I table + plot |
| 3 | Extend `run_worst_case.py` to LIME | ~2–4 hr | Compare worst-case gap across SHAP/LIME |
| 4 | Increase GeoSHAP N in robustness (20→50) | ~8 hr | Complete 50×3×4 matrix |
| 5 | N≥5 seeds for statistical significance | ~1 hr | Error bars on robustness matrix |
