# Follow-up directions (SBM-as-explainer thread)

Last updated: 2026-04-23 morning (pre-Peng meeting). Reorganized by
status: **closed** (done), **queued** (ready to run, waiting on Peng
green-light or capacity), **speculative** (ideas, no commitment).

Pitch the queued items to Peng on Thursday; start whichever he picks.

---

## Closed

### 1. Slack COMPAS / German / CC replication — CLOSED ✓

Four-dataset replication is done. Δ magnitudes match within seed
noise:

| Dataset | Best cell | Δ | C ratio |
|---|---|---|---|
| Housing | Gauss + LIME σ=0.1 | **+0.048** | 2.51 |
| Slack COMPAS | Gauss + SHAP σ=0.1 | **+0.038** | 1.94 |
| Slack German | Gauss + LIME σ=1.0 | **+0.057** | 1.36 |
| Slack German | Gauss + SHAP σ=1.0 | **+0.071** | 1.05 |
| CC psi_two (XOR) | Gauss + LIME σ=1.0 | **+0.575** | 1.43 |

**Invariant confirmed across all four:** Δ > 0 iff the OOD detector
produces a partial routing mixture. Role reversal housing↔COMPAS
(LIME fires on housing, SHAP fires on COMPAS) is detector-manifold
calibration.

Artifacts:
- `results/sbm_explainer/compas_fingerprint/results.json`
- `results/sbm_explainer/german_fingerprint/results.json`
- `results/sbm_explainer/cc_fingerprint/results.json`
- Full narrative: `summary_apr22.md` §4a–c.

### 2. Probe 1 — OOD routing check — CLOSED ✓

Original mechanism ("LIME OOD lets perts through") was **wrong**.
Corrected via `run_ood_flip_probe.py` and `run_honest_fingerprint_check.py`:
LIME detector rejects 0% of σ=0.1 perts (all → innocuous); SHAP
detector accepts 73% (→ biased). Single-model configs are all null
(honest / biased / innocuous each give Δ ≤ 0.011). **Signal requires
the scaffold's routing mixture** — this is a scaffold-attack detector,
not a feature-geometry detector.

Artifacts: `results/sbm_explainer/{ood_flip, honest_check}/results.json`.

### 3. Probe 2 — 20-seed CI on housing Gauss+LIME — CLOSED ✓

Δ = 0.043 ± 0.018 (SEM 0.004), 95% CI [+0.017, +0.076], t = 10.68,
**p = 1.8 × 10⁻⁹**. Five-seed estimate (+0.048) reproduces.

Artifact: `results/sbm_explainer/seedscale_gauss_lime/results.json`.

### 4. Mechanism confirmation — K-sweep + Tanh-β probe — CLOSED ✓

- **K-sweep** (`run_k_scaling_sweep.py`): CC psi_two LIME σ=1.0.
  Plain NMI 0.333 (K=5) → 0.020 (K=80); pEx stays 0.33–0.56. Δ rises
  from +0.001 to +0.473 as K grows. **Anchor-dominance erosion
  confirmed.**
- **Tanh probe** (`run_tanh_sweep.py` + `run_tanh_seedscale.py`):
  β=5, n=10 gives Δ = +0.056, 95% CI [+0.031, +0.080]. Three-phase
  Δ(β): saturated (β≤2), active (3–10), rank-degenerate (β≥20).

Unified §3.1 mechanism in `cosine-expression.md` supersedes earlier
"smoothness" and "decorrelation" framings.

### 5. H-DP pEx (Fix 4) — CLOSED ✓

Durante 2025's two-level DP on the housing Gauss+LIME graph.
K̂ ≈ 10 across θ₀ ∈ {0.25, 0.5, 1.0, 2.0}. Binary-layer NMI = 0.261
(arity-depressed). Refinement payoff: 7/10 clusters layer-pure, 2
clusters (39 nodes) at gate boundary with three LIME attribution
regimes (gate-invisible / partial / openly-leaking).

Artifacts: `results/sbm_explainer/hdp_pex/`.

### 6. §2 Bayesian outputs + block characterization — CLOSED ✓

Ψ matrix + pairwise co-clustering + per-node entropy + semantic
block labels extracted for housing (`extract_pex_outputs_protected.py`,
`characterize_pex_blocks.py`) and the three Slack datasets
(`extract_slack_suite_pex_outputs.py`). Layer-disassortative Ψ on
3/4 datasets (partial on German C=1.05). Flip blocks found on 3/4.

Artifacts: `results/sbm_explainer/{pex_outputs, slack_outputs}/`.

### 7. Spectrum sweep across Moran's I — CLOSED ✓

Exp 12: 8 features spanning I ∈ [0.09, 0.85]. **5/8 show Δ > 0.**
Three regimes emerge: saturation / middle / diffuse. §6.2 reframed
from "universal" to "middle regime (C ∈ [1, 3])."

Artifacts: `results/sbm_explainer/spectrum_sweep/`.

### 8. Sinusoidal Lipschitz sweep — RETIRED ✗

`run_lipschitz_sweep.py`. Δ positive across β but n=3 CIs overlap;
E[n_flips] reparameterization didn't help. Kept as a robustness
result, removed from mechanism-test list.

---

## Queued (pitch to Peng)

### Q1. Penalty-magnitude sweep on the rung-3 Gauss+LIME cell

Rerun the main housing rung-3 cell at scaffold penalties {0.10,
0.30, 0.50}. Does Δ = +0.048 survive at weaker / stronger attacks?
Does the three-regime LIME attribution story collapse to two regimes
at strong penalty, or persist?

- Timeline: half day. Reuses existing pipeline; one config knob.
- Deliverable: table of (penalty, Δ, C-ratio) across the 3 penalties,
  plus the three-regime attribution breakdown at each.
- Importance: **robustness check** for the paper's headline number.

### Q2. Predictive test on H-DP gate-boundary clusters

Hypothesis: the 39 gate-boundary nodes (H-DP clusters 6 and 7,
purity 0.56 / 0.57) are the ones most susceptible to adversarial
perturbation, because they sit at the OOD-detector decision boundary.

Test per-node attack sensitivity two ways:
1. Smallest ||δ|| that flips the scaffold's routing (real → perturbed
   or vice versa).
2. Largest change in LIME attribution of `sqft_lot` under a bounded
   adversarial δ.

Compare gate-boundary nodes vs pure-low and pure-high clusters.

- Timeline: 1–2 days. Requires writing a small adversarial-search
  script; pipeline otherwise exists.
- Deliverable: per-cluster sensitivity distribution; if gate-boundary
  is significantly higher, H-DP moves from "interpretive diagnostic"
  to **"predictive method."**
- Importance: would be a strong second-contribution candidate for a
  NeurIPS-main paper.

### Q3. Plain-SBM modular-recovery-probability bound

Derive a bound on plain-SBM's probability of recovering the layer
as a function of (m-field variability, K, layer-alignment). Would
make the §3.1 unified mechanism into a formal theorem.

- Timeline: weeks. Requires Bayesian-nonparametric / SBM-theory
  expertise (Camerlenghi et al. 2019 distribution theory + DMMS
  threshold).
- Deliverable: a theorem stating "under conditions X, plain-SBM
  recovery probability ≤ f(m-var, K, align); under those conditions
  pEx-SBM recovers the layer with probability ≥ g(·)."
- Importance: **required** for a NeurIPS-main submission. Optional
  for workshop / TMLR.

### Q4. Extend worst-case x* (Exp 6) to LIME

Currently TreeSHAP-only (LIME requires per-candidate explain calls).
Subsample to n=20 candidates per (reference point, ε) pair; ~2–4 hr.

- Deliverable: cross-explainer worst-case comparison.
- Importance: finishes the Exp 6 picture; low-priority.

### Q5. Increase GeoSHAP N in robustness (Exp 5)

Robustness matrix currently has GeoSHAP on 20 test points / 20
neighbors (vs 50/50 for other explainers). Increasing to 50/50
gives the proper 50×3×5 matrix but takes ~10 hours on a laptop.

- Importance: cleans up the Exp 5 figure; not headline-critical.

---

## Speculative (no commitment yet)

### S1. Rung-3 variants — maximize C-ratio deliberately

Current Gauss fingerprint at σ=0.1 produces C = 2.51. A higher-dim
targeted grid sweeping multiple features jointly might push C higher
while staying inside the middle regime. Would strengthen Δ.

### S2. Encoder-as-detector paper (no pEx component)

The K=2 plain-SBM NMI = 0.916 on sqft_lot LIME (strong-penalty attack)
is the strongest unsupervised attack-detection signal in the whole
project. Could be a stand-alone paper: "plain SBM on rung-1 LIME
attribution graphs recovers the attack's protected layer at strong
penalty, no pEx needed." Shorter paper, narrower claim.

### S3. SAE-feature case study (the LLM arm of the paper proposal)

§6.3.1 of `Notes/pEx-SBM Paper Proposal.md` promises a case study
fitting pEx-SBM with transformer depth as the layer coloring on SAE
feature graphs. Requires GPU + pretrained SAE dictionary. Not
prioritized for the geo/XAI arm, but the obvious LLM extension.

---

## Peng meeting asks (Thursday April 23)

See `../thursday_discussion.md` for context. Three decisions:

1. **Lead finite-K Δ or H-DP refinement?** Finite-K is clean (one
   number, p = 1.8e-9). H-DP is richer (3 attribution regimes,
   gate-boundary sub-population) but arity-mismatched (K̂ ≈ 10 vs
   binary layer).
2. **Workshop/TMLR vs NeurIPS-main?** Workshop (four-dataset +
   K-sweep + middle-regime claim) is achievable now. Main-track
   needs Q3 (the theoretical bound).
3. **Tanh β-sweep — keep or drop?** Keep — it's the second
   confirmatory mechanism axis alongside K-sweep.

Bonus: is the K=2 NMI=0.916 finding (S2) publishable standalone?
