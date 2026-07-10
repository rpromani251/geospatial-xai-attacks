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
| pct_black_latino | 0.022¹ | 0.035 | +59% |

¹ Honest under the 11-feature model (`docs/comparison_honest_vs_adv_off_baseline.md`). Apples-to-apples with Adv SHAP. Note this is a **rise**, not a suppression — an earlier version of this row claimed "(suppressed)," which was wrong: the attribution on `pct_black_latino` under SHAP *increases* under the adversarial scaffold. The bias gets partially laundered into UTM_Y's drop and structural proxies' rise, but SHAP still reports more demographic attribution on the scaffolded model than on the honest one.

Under **LIME**, UTM_Y barely moves (−3%) and `pct_black_latino` *rises* from 0.026 (11-feat honest) to 0.108 (adv) — a 4× increase. LIME does not "resist the attack by preserving UTM_Y"; it reports a very visible demographic signature in addition to preserving UTM_Y. Whether this is a useful property depends on what the auditor is looking for: LIME is a better leading indicator than SHAP here. The GeoSHAP result is the starkest on coordinates: UTM_Y collapses from 0.013 (honest) to 0.009 (−31% on the GeoSHAP scale, a much smaller base); combined with `pct_black_latino = 0.030`, GeoSHAP's adversarial attribution is the hardest to interpret because its geographic background forces every coalition query inside the biased manifold.

**Main finding.** Geographic awareness in GeoShapley is not a defense — it makes the attack easier, not harder. LIME surfaces the bias on `pct_black_latino` because its Gaussian perturbations occasionally land in minority tracts and expose biased-model behavior to the explainer.

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

## Experiment 4b — No-Location Scaffold Ablation

**Motivation.** Experiment 4 shows that stripping UTM from the honest model costs 27 R²-points. But does the *scaffold attack* depend on UTM being in the feature set? If the OOD detector relies on the coordinate distribution to distinguish real from perturbed, removing coordinates should break the attack.

**Setup.** Identical to Experiment 3 (same penalty, same threshold, same OOD detector training protocol), but all three models (biased, innocuous, OOD detectors) use 9 features instead of 11 — `UTM_X` and `UTM_Y` are dropped everywhere.

**Results.**

| Scaffold | OOD accuracy | Fidelity (Pearson r) | % routed biased |
|---|---|---|---|
| LIME | 100.0% | 1.000 | 99.8% |
| SHAP | 99.3% | 0.983 | 78.8% |

The attack still works without coordinates. LIME fidelity is perfect, SHAP fidelity is near-perfect, OOD accuracy matches Experiment 3 exactly. `pct_black_latino` becomes *more* visible in the adversarial explanations (SHAP 0.044 vs 0.035, LIME 0.110 vs 0.108) — with no UTM channel to launder into, the demographic signal leaks slightly. The spatial correlation amplification still holds: biased model r = −0.600 vs honest no-loc r = −0.289 (+0.31 amplification).

**Main finding.** The attack is not *coordinate laundering*; it is an **OOD-routing mechanism** that works on whatever feature set you have. Removing coordinates does not defend against the attack — it just redirects the bias into structural proxies. Negative result, but a load-bearing one for the paper's threat model: "strip coordinates" is not a valid defense.

Output: `results/scaffold_no_location/` (metrics, spatial correlation, adversarial attribution arrays).

---

## Experiment 5b — Slack-Suite Robustness Replication

**Motivation.** Experiment 5's headline — "Random > Medoid" — is weird. If it's a spatial-data property, it's niche. If it's a general tree-model property, it generalizes to every tabular setting. To test: rerun the robustness protocol on the three Slack-suite datasets with RF honest models (matching Slack's original setup).

**Results (Random − Medoid gap, mean over 30 test points).**

| Dataset | ΔTreeSHAP | ΔKernelSHAP | ΔLIME |
|---|---|---|---|
| Housing | +0.330 | +0.316 | +0.050 |
| COMPAS | +0.382 | +0.410 | +0.020 |
| German | +0.658 | +0.651 | **−0.211** |
| CC psi_two | +0.502 | +0.276 | +0.134 |
| **Mean** | **+0.468** | **+0.413** | −0.002 |

**TreeSHAP and KernelSHAP: Random > Medoid universally.** The effect is *stronger* on Slack tabular data than on housing, which rules out the spatial-autocorrelation hypothesis. The mechanism is split-boundary crossings in tree models — general tabular, not geospatial.

**LIME reverses on German.** Random ρ = 0.270, Medoid ρ = 0.481, Δ = −0.211 (outside noise). Explanation: German Credit is categorical-heavy (binary indicators for account status, purpose, housing type). LIME's independent Gaussian perturbations generate fractional values for binary features — off-manifold in a way that destabilizes the local linear fit. Medoid neighbors are real rows, so the fit is stable.

**Implication.** Experiment 5's headline should be qualified: "Random > Medoid is universal for SHAP variants; LIME's ordering depends on feature geometry and may reverse in categorical-heavy settings." Full analysis in `docs/slack_robustness_comparison.md`; figure `docs/figures/fig_robustness_slack_vs_housing.png`.

Output: `results/slack_robustness/{compas, german, cc_psi_two}/{rho_matrix.npy, results_matrix.json, splits.pkl, honest_model.pkl}`.

---

## Experiment 8b — Base-Importance-Normalized Feature Sweep

**Motivation.** Experiment 8's "visibility monotone in Moran's I" finding has a confound: features that are naturally predictive have larger honest attribution, so adversarial attribution on them is larger *regardless* of attack effectiveness. Normalize by honest importance to see if the I-trend survives.

**Setup.** Extend the sweep from 5 features to 8 (add `condition`, `sqft_living`, `grade`, `age`). Add columns for `honest_importance`, `delta_*_importance`, and `normalized_visibility_*` to the summary JSON.

**Finding.** On *normalized* visibility, the I-trend **flips or weakens sharply**. Pearson correlation between Moran's I and normalized LIME visibility is **ρ = −0.43** (slightly *negative*, not monotone positive). The raw-visibility I-trend in Experiment 8 was largely an artifact of base feature importance, not spatial clustering.

**Revised claim.** The regime distinction is more important than the I-trend: structural features (on-baseline) hide by *amplification*, demographic features (off-baseline) hide by *laundering through proxies*. Both regimes are I-invariant at the OOD-detector level; the visibility difference is a feature-role artifact.

Output: `results/feature_sweep_a2/{sweep_summary.json, combined_8feature_summary.json, sweep_plot.png}`.

---

## Experiments 10 / 10b / 10c / 10d / 11 — Non-Autocorrelated Bias-Feature Attempts

**Motivation.** After Experiment 8 disconfirmed the original Moran's I hypothesis, a new prediction emerged (`Notes/2026-04-16 Non-Autocorrelated Bias Feature.md`): GeoSHAP's vulnerability might depend on spatial autocorrelation of the *bias trigger* specifically. For a low-I trigger, GeoSHAP's geographic background would straddle both sides of the bias boundary, giving GeoSHAP real signal instead of systematically biased signal. **Prediction: GeoSHAP reverses from most-compromised → most-resistant.**

**Five attempts.**

| # | Trigger | Moran's I | Coverage | Verdict |
|---|---|---|---|---|
| 10 | FEMA 100-year floodplain (`in_floodplain`) | −0.001 (ns) | 0.17% (1/600) | Coverage too low — untestable |
| 10b | FEMA 500-year flood zone | 0.239 | 0.5% (3/600) | Coverage too low |
| 10c | FEMA 100-year at KC-parcel scale (59k) | — | 0.47% at 20× scale | Coverage still too low; scale doesn't fix it |
| 10d | Geologic hazard (erosion + landslide, KC iMap) | 0.246 (p=0.002) | 6.5% (39/600) | **The one that works.** Adv LIME 8.5× honest; GeoSHAP reversal **not confirmed** — GeoSHAP still gets fooled |
| 11 | SPS school attendance zones (`in_low_score_zone`) | ≈0.80 | 49% | High Moran's I — doesn't test the reversal hypothesis |

**Verdict.** GeoSHAP reversal **disconfirmed** at achievable coverage. Experiment 10d is the only one with enough penalized test points to draw a conclusion, and even there, at Moran's I = 0.25, GeoSHAP is still compromised (the `utmy_suppression_geoshap` metric doesn't cross the reversal threshold). The attack's mechanism is feature-agnostic at the geographic level.

**What the thread still produced.**
- The **honest-11-feature comparison pipeline** (`scripts/compute_honest_11feat.py`, `results/honest_11feat/{exp3, exp10, exp10b, exp10c, exp10d, exp11}/`): for each trigger, retrain an honest XGB with the trigger included, so honest-vs-adversarial is apples-to-apples. Pivotal finding: **adversarial UTM_Y suppression is 5–7× the natural redistribution** an 11-feature honest model produces when the trigger is added. This nails down that the scaffold does genuine laundering, not just natural feature competition.
- The **parcel dataset build pipeline** (`scripts/build_kc_parcel_dataset.py`). Reusable for any future county-scale experiment.
- Per-trigger Moran's I measurements (`results/scaffold_*/morans_i.json`).

Full cross-trigger comparison table: `docs/comparison_honest_vs_adv_off_baseline.md`.

Outputs: `results/scaffold_{flood_zone, flood_500yr, parcel_flood, geologic_hazard, school_zone}/`.

---

## Experiment 9 — SBM-as-Explainer (Active Thread)

**Motivation.** Every previous experiment asked "can the explainer detect / resist the attack?" The honest summary: SHAP is fooled, GeoSHAP is maximally compromised, LIME partially resists. The defense game is lost at the explainer level.

The pivot: can a **completely different inference procedure** recover the protected layer from the scaffolded model's *behavior* — no explainer involved, just the model and the test points and a known protected-attribute label (which an auditor has). Treat each test point as a node, build a graph where edges connect behaviorally similar points, fit a stochastic block model that knows about the protected attribute (pEx-SBM), and see if community structure aligns with the layer.

**Pipeline.**
- N = 200 test points as graph nodes.
- Each node encoded as a **rung-3 behavioral fingerprint**: `e_i = [f_scaffold(x_i), f_scaffold(x_i+δ_1), …, f_scaffold(x_i+δ_K)]`. First coordinate is the real-input anchor (routed to the biased model), remaining K are perturbation responses (routed to the innocuous model by a strict OOD detector). K=4, σ = 0.1 × feat_std, Gaussian perts.
- 8-NN cosine similarity graph on the fingerprints.
- Plain SBM (layer-agnostic prior) vs finite-K Dirichlet-Categorical pEx-SBM (per-layer Dirichlet prior), K=4 in both.
- Layer = `pct_black_latino > 0.25` for housing; race / gender / `racePctWhite` for the Slack suite.
- Metric: Δ = NMI(z_pEx, layer) − NMI(z_plain, layer), averaged over SBM seeds. C-ratio = within-layer / across-layer co-clustering density serves as a regime indicator.

**Housing headline (Gauss+LIME, K=4, 200 nodes).**

| Method | NMI vs layer | C-ratio |
|---|---|---|
| Plain SBM | 0.287 | 2.51 |
| Finite-K pEx SBM | **0.334** | 2.51 |
| **Δ** | **+0.048** (20-seed CI [+0.017, +0.076], **p = 1.8 × 10⁻⁹**) | — |

First non-null pEx cell in the whole SBM thread. Retracts the earlier claim (`plain-sbm-poc.md` §14) that "finite-K pEx is structurally unable to differ from plain" — that claim was about attribution-vector graphs (rung 1), which are regime (b). A well-chosen encoder (rung 3) produces a middle-regime graph (C ∈ [1, 3]) where the prior has room to act.

**Probe 1 — mechanism correction.** Original draft claimed "LIME lets perts through → gate visible." Wrong. The LIME detector rejects **0%** of σ=0.1 perts as real (all routed to innocuous). The SHAP detector accepts **73%** as real (routed to biased). The fingerprint under the LIME detector is `[biased(x), innocuous(x+δ_1), …]` — one gate-exposed anchor + K gate-hidden refinements. That **routing mixture** is the signature rung-3 picks up. Single-model sanity check confirms: honest-only, biased-only, innocuous-only all null (Δ ≤ 0.011, C ≤ 1.06). The signal requires the scaffold.

**Four-dataset replication** (rung-3 + finite-K pEx, σ swept over {0.1, 0.3, 1.0}).

| Dataset | Best cell | Plain | pEx | Δ | C |
|---|---|---|---|---|---|
| Housing (regression) | Gauss + LIME, σ=0.1 | 0.287 | 0.334 | **+0.048** | 2.51 |
| Slack COMPAS | Gauss + SHAP, σ=0.1 | 0.303 | 0.340 | **+0.038** | 1.94 |
| Slack German | Gauss + LIME, σ=1.0 | 0.115 | 0.172 | **+0.057** | 1.36 |
| Slack German | Gauss + SHAP, σ=1.0 | 0.048 | 0.119 | **+0.071** | 1.05 |
| CC psi_two (XOR innocuous) | Gauss + LIME, σ=1.0 | 0.124 | 0.699 | **+0.575** | 1.43 |

**Invariant: Δ > 0 iff OOD produces a partial routing mixture** (not all-biased, not all-innocuous). Role reversal housing↔COMPAS reflects detector-manifold calibration, not explainer-specific behavior. **XOR innocuous amplifies Δ 10×** — less-smooth innocuous produces more internally-coherent refinements, opposite of a naive "pEx detects smoothness" framing.

**Mechanism — anchor-dominance erosion.** Cosine expansion:
$$\cos(e_i, e_j) = \frac{m_i m_j}{s_i s_j} + \frac{a_i a_j}{K \cdot s_i s_j} + O(K^{-1/2})$$

The anchor term is $O(1/K)$ — its weight in cosine geometry shrinks as the fingerprint grows. At small K the anchor dominates, plain SBM recovers the layer through the anchor coordinate alone. At large K the refinement-mean term (which is **layer-independent** when innocuous is layer-blind) takes over the $O(1)$ slot, plain SBM rotates onto that (non-layer) partition, and pEx-SBM holds the layer via its conditional prior. **Δ measures plain-SBM's failure rate, not signal magnitude.**

**K-sweep confirmation (CC psi_two, LIME, σ=1.0).**

| K | Plain NMI | pEx NMI | Δ |
|---|---|---|---|
| 5 | **0.333** | 0.334 | +0.001 |
| 10 | 0.187 | 0.307 | +0.120 |
| 20 | 0.082 | **0.555** | **+0.473** |
| 40 | 0.050 | 0.491 | +0.441 |
| 80 | 0.020 | 0.426 | +0.406 |

Plain collapses; pEx holds.

**Tanh-β probe — second mechanism axis.** Innocuous model = `Bernoulli(sigmoid(β · x_u))`. Small β → `m_i` near-constant → case 1 → Δ ≈ 0. Large β → `m_i` variable, layer-orthogonal → case 3 → Δ > 0. 10-seed rerun at β=5 gives Δ = +0.056, 95% CI [+0.031, +0.080] — the highest-confidence resolution in the thread outside Probe 2. Three-phase Δ(β): saturated (β ≤ 2), mechanism active (β ≈ 3–10), rank-degenerate (β ≥ 20 — both plain and pEx crash).

**H-DP extension (Fix 4).** Durante 2025's two-level Dirichlet-Process pEx-SBM on the same graph. Infers K̂ ≈ 10, insensitive to θ₀ across a 16× range. Binary-layer NMI = 0.261 — *lower* than finite-K (0.334) because arity mismatch depresses NMI. H-DP's contribution is **refinement**: 7/10 clusters ≥ 90% layer-pure, 2 clusters (39 nodes, ~20%) sit at the gate boundary with purity 0.56–0.57. Per-cluster LIME attributions separate three scaffold regimes — gate-invisible (85 nodes, `sqft_lot` absent top-3), gate-partial (89 nodes, `sqft_lot` 0.134), gate-openly-leaking (26 nodes, `sqft_lot` 0.254 ≈ 25× gate-invisible).

**§2 Bayesian outputs.** `scripts/extract_pex_outputs_protected.py` extracts the three paper-level primitives:
- Ψ block-probability matrix (2 × K). Housing: layer-disassortative, not-protected → Block 3 (0.493), protected → Block 2 (0.524).
- Pairwise co-clustering posterior on a 20-node subset. Within-layer mean 0.312 vs across-layer 0.237, ratio 1.32.
- Per-node entropy. 9/20 confident (H < 0.001), 4/20 gate-boundary (H > 0.3) — correspond to the H-DP gate-boundary clusters.

Same primitives extracted for the three Slack datasets by `extract_slack_suite_pex_outputs.py`. Layer-disassortative Ψ replicates on 3/4 datasets; gate-boundary "flip blocks" found on COMPAS, German, CC.

**Sinusoidal Lipschitz sweep — retired.** `sign(sin(β · x_u))` lives entirely in mechanism case (3) for any β, so it can't test the three-case prediction. Δ is positive but flat at n=3 seeds with overlapping CIs. Kept as a robustness result.

Entry point for the full thread: `docs/sbm_explainer/README.md` → `summary_apr22.md` → `cosine-expression.md`.

Outputs: `results/sbm_explainer/{fingerprint_sbm, compas_fingerprint, german_fingerprint, cc_fingerprint, k_scaling, tanh_sweep, tanh_seedscale, lipschitz_sweep, hdp_pex, honest_check, ood_flip, seedscale_gauss_lime, pex_outputs, slack_outputs}/`.

---

## Experiment 12 — pEx-SBM Across the Moran's I Spectrum

**Motivation.** The paper-proposal §6.2 originally promised "pEx-SBM recovers block structure across the full autocorrelation range." Experiment 12 validates this by running the rung-3 pipeline on 8 different bias-feature layers spanning I ∈ [0.09, 0.85].

**Results.**

| Feature | Moran's I | Plain NMI | pEx NMI | Δ | C-ratio | Regime |
|---|---|---|---|---|---|---|
| condition | 0.090 | 0.065 | 0.064 | −0.000 | 0.87 | diffuse |
| view | 0.242 | **0.877** | 0.877 | 0.000 | sat | saturation |
| sqft_living | 0.291 | 0.014 | 0.007 | −0.007 | 0.98 | diffuse |
| grade | 0.365 | 0.014 | 0.020 | +0.006 | 1.00 | **middle** |
| sqft_lot | 0.469 | 0.027 | 0.041 | +0.014 | 1.20 | **middle** |
| pct_latino | 0.756 | 0.011 | 0.012 | +0.000 | 0.94 | diffuse |
| pct_black_latino | 0.834 | **0.287** | **0.334** | **+0.048** | 1.32 | **middle** |
| pct_black | 0.850 | 0.012 | 0.014 | +0.002 | 0.89 | diffuse |

**5 of 8 features show Δ > 0.** The original universal §6.2 claim is **false**. Three regimes emerge:

- **Saturation (view)**: Plain NMI 0.877 (near-perfect). pEx has no room to improve.
- **Middle (grade, sqft_lot, pct_black_latino)**: C-ratio 1.0–1.3, plain NMI 0.01–0.3. **This is where pEx fires.**
- **Diffuse (condition, sqft_living, pct_latino, pct_black)**: C-ratio < 1 or ≈ 1, penalty too sparse or feature absent from baseline.

**Reframed §6.2 claim.** Not "recovers across the full autocorrelation range" but **"recovers in the middle behavioral regime, identified by C-ratio ∈ [1, 3]."** This maps to Moran's I ≈ 0.35–0.83 in King County. The claim is *auditor-actionable*: compute plain-SBM's C-ratio first; if it's in the middle band, run pEx-SBM.

**Bug disclosure.** `docs/sbm_explainer/spectrum_sweep_diagnostic.md` documents a data-lookup bug in `run_spectrum_sweep.py` (lines 521–531) that displayed `pct_black_latino` Plain NMI as 0.000 (hardcoded fallback). The Δ = +0.048 was coincidentally correct. Canonical triple: Plain = 0.287, pEx = 0.334, Δ = +0.048.

Output: `results/sbm_explainer/spectrum_sweep/{summary.json, <feature>/{fingerprints, graph, layer, psi_*, result}.npy, figures/*.png}`.

---

## Thursday April 23 meeting — three asks

See `docs/thursday_discussion.md` for full context. The three decisions Peng needs to weigh in on:

1. **Lead finite-K Δ (clean, binary-layer NMI) or H-DP refinement (richer, arity-mismatched)?** Finite-K gives one clean number (Δ = +0.048, p = 1.8e-9). H-DP identifies a 20% gate-boundary sub-population and three scaffold-attack regimes. A single clean headline number favors finite-K; a richer two-part story favors presenting both the NMI headline and the H-DP refinement together.

2. **Is the current evidence sufficient, or does it need a formal bound?** Four-dataset replication + K-sweep + middle-regime claim is the current evidence. A stronger claim would need a theoretical plain-SBM modular-recovery-probability bound as a function of (m-variability, K, layer-alignment) — substantial extra work.

3. **Tanh β-sweep — keep or drop?** Keep as mechanism confirmation #2 (alongside K-sweep). The 10-seed knee at β=5 is solid. The sinusoidal sweep is retired.

Bonus: does the K=2 NMI=0.916 sqft_lot LIME unsupervised-detection finding stand on its own as a separate result, or does it only make sense as part of the larger SBM story?

---

## Next experiments (post-meeting, priority-ordered)

| # | Item | Est. runtime | Rationale |
|---|---|---|---|
| 1 | Penalty-magnitude sweep on rung-3 Gauss+LIME cell ({0.10, 0.30, 0.50}) | half-day | Confirm Δ = +0.048 is robust to attack strength |
| 2 | H-DP gate-boundary predictive test — do the 39 gate-boundary nodes have higher held-out adversarial sensitivity? | 1–2 days | Promotes H-DP from interpretive to predictive diagnostic |
| 3 | Extend `run_worst_case.py` to LIME (subsample n=20 candidates per point) | 2–4 hr | Cross-explainer worst-case comparison |
| 4 | Increase GeoSHAP N in robustness (20 → 50) | ~10 hr | Complete 50×3×5 matrix; enables proper ensemble |
| 5 | Plain-SBM modular-recovery-probability bound | weeks | Only needed if a formal theoretical claim is required |

`docs/sbm_explainer/followup_directions.md` has the canonical queued-follow-up list.
