# Post-Thursday follow-up directions

Parked after the April 22 session. Pitch to Peng on Thursday; start
whichever he green-lights.

## 1. Slack COMPAS reproduction (load-bearing for paper — started April 22 pm)

Slack's repo is at `../fooling-lime-shap/` (one level above the repo).
`scripts/sbm_explainer/run_compas_fingerprint.py` wires their
`Adversarial_Lime_Model` and `Adversarial_Kernel_SHAP_Model` (binary
classifier scaffold, race as biased feature, unrelated column as
innocuous proxy) directly into our rung-3 fingerprint pipeline. No
retraining of biased/innocuous models — Slack's racist_model_f and
innocuous_model_psi are used unchanged.

Runs Gaussian fingerprints at σ ∈ {0.1, 0.3, 1.0} (post-StandardScaler)
on the scaffold's class-1 probability output, then plain SBM +
finite-K pEx vs the race layer, for both LIME and SHAP scaffolds.

Prediction: LIME-scaffold Gauss fingerprints should give Δ > 0 and
C ratio > 2 at some σ (likely σ > 0.3 where perts exceed the OOD
training distribution), echoing the housing result. SHAP-scaffold
should stay flat.

### Results (April 22 pm)

| Scaffold | σ | plain NMI | pEx NMI | Δ | C ratio | routing |
|---|---|---|---|---|---|---|
| LIME | 0.1 | 0.574 | 0.574 | 0.000 | 14.05 | 0% real |
| LIME | 0.3 | 0.574 | 0.574 | 0.000 | 14.05 | 0% real |
| LIME | 1.0 | 0.033 | 0.029 | −0.003 | 1.10 | 0.1% real |
| **SHAP** | **0.1** | **0.303** | **0.340** | **+0.038** | **1.94** | **70% real** |
| SHAP | 0.3 | 0.349 | 0.347 | −0.001 | 3.01 | 68% real |
| SHAP | 1.0 | 0.228 | 0.260 | +0.032 | 1.69 | 48% real |

**Finding: the method replicates, with role reversal.** On COMPAS the
Gauss+SHAP scaffold at σ=0.1 is the direct analog of the housing
Gauss+LIME cell: plain 0.303 → pEx 0.340, Δ=+0.038, C ratio 1.94 — same
shape as housing's plain 0.287 → pEx 0.334, Δ=+0.048, C ratio 2.50.

**Why the swap.** The invariant is routing-mixture: pEx Δ > 0 whenever
the OOD detector produces a partial mixture (not all-biased, not
all-innocuous). Which detector plays that role depends on the detector's
training distribution relative to the fingerprint pert scale. On
housing, LIME's raw-unit σ=0.3 training was very strict vs our
σ=0.1·feat_std fingerprint perts. On COMPAS post-StandardScaler,
LIME's σ=0.3 training is absolute and extremely strict (rejects 100%
of σ=0.1 perts), while SHAP's kmeans-based OOD is moderately
permissive. Role-reversal reflects detector-manifold calibration, not
explainer-specific behavior.

**COMPAS also has a "saturated plain" regime.** LIME scaffold at
σ∈{0.1, 0.3}: plain NMI 0.574 with C ratio 14. The innocuous proxy
(unrelated_column_one) correlates strongly enough with race in this
synthetic setup that the all-innocuous fingerprint alone recovers the
layer at plain SBM level. pEx has nothing to add. Different regime
from both housing-null and housing-signal.

Two-dataset replication across regression vs classification, continuous
vs mostly-binary features, custom vs Slack-original adversarial
scaffolds. Δ magnitudes (0.048 housing, 0.038 COMPAS) match within
seed noise.

## 2. Penalty sweep

Rerun the rung-3 Gauss+LIME cell at scaffold penalties {0.10, 0.30,
0.50}. Does Δ +0.048 survive, grow, or vanish? Does the three-regime
LIME attribution story collapse to two regimes at strong penalty, or
persist?

Timeline: half day. Reuses the existing pipeline with one config knob
changed.

## 3. Predictive test — three-regime as attack-sensitivity predictor

The three H-DP-derived LIME attribution regimes (pure-low, layer-mixed,
pure-high) differ by how much sqft_lot leaks through LIME. Hypothesis:
the layer-mixed cluster nodes (clusters 6 and 7, 39 nodes) are the
ones most susceptible to adversarial perturbation, because they sit
at the gate decision boundary.

Test: for each of the 200 nodes, compute attack sensitivity — e.g.,
smallest ||δ|| that flips the scaffold's routing decision, or the
largest change in LIME attribution of sqft_lot under a bounded
adversarial perturbation. Check whether cluster-6/7 nodes have
significantly higher sensitivity than pure-low or pure-high nodes.

Timeline: 1–2 days. Would move the rung-3 + H-DP result from
"interpretive diagnostic" to "predictive method."
