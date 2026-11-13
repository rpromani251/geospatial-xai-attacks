# Project state — 2026-05-22

Self-contained snapshot for planning future directions outside this
session. Distilled from `CLAUDE.md`, `summary_apr22.md`,
`day_log_apr22.md`, `followup_directions.md`, and `cosine-expression.md`.

---

## One-sentence summary

A rung-3 behavioral fingerprint encoder + finite-K Dirichlet-Categorical
pEx SBM is a Slack-style scaffold-attack detector; the mechanism is
anchor-dominance erosion (a.k.a. m-field variability + layer-misalignment),
empirically replicated on four datasets and confirmed on two independent
mechanism axes (K and Lipschitz β).

---

## Two parallel tracks

### Track A — original geospatial plan (COMPLETE)
All 6 steps done on Seattle housing:
1. Baseline XGBoost (R² = 0.89)
2. SHAP / LIME / GeoSHAP wrappers
3. Adversarial scaffold (SHAP attack succeeds, LIME partial — opposite of Slack's tabular result)
4. Spatially-aware 3-level neighborhoods
5. 3×5 robustness matrix + trustworthiness classification
6. Worst-case x* sweep (TreeSHAP-only)

Loose ends (unlikely to be revived): extend x* sweep to LIME, raise
N_EXPLAIN_GEO, add honest GeoSHAP to baseline.

### Track B — SBM-as-explainer (ACTIVE)
The current research thread. Below.

---

## Track B — what's established

### Empirical
| Dataset | Best cell | plain → pEx | Δ | C ratio |
|---|---|---|---|---|
| Housing (Gauss+LIME, σ=0.1) | rung-3 | 0.287 → 0.334 | **+0.048** (6σ) | 2.51 |
| COMPAS (Gauss+SHAP, σ=0.1) | rung-3 | 0.303 → 0.340 | +0.038 | 1.94 |
| German (Gauss+LIME, σ=1.0) | rung-3 | 0.115 → 0.172 | +0.057 | 1.36 |
| German (Gauss+SHAP, σ=1.0) | rung-3 | 0.048 → 0.119 | +0.071 | 1.05 |
| CC psi_one (SHAP σ=0.1) | rung-3 | 0.270 → 0.369 | +0.099 | 1.93 |
| **CC psi_two (LIME σ=1.0)** | rung-3 | 0.124 → 0.699 | **+0.575** | 1.43 |

**Routing-mixture invariant:** Δ > 0 iff the scaffold's OOD detector
produces a partial mixture (not all-biased, not all-innocuous). Which
explainer (LIME vs SHAP) plays the routing role is detector-manifold
dependent, not explainer-specific.

**Probe 2 (20-seed CI):** Δ = 0.043 ± 0.018, t = 10.68, p = 1.8×10⁻⁹.

### Mechanism (§3.1 of `cosine-expression.md`)
Edge cosine decomposes as
`cos(e_i, e_j) = m_i m_j/(s_i s_j) + a_i a_j/(K s_i s_j) + O(K^{-1/2})`.
pEx beats plain SBM **iff** the refinement-mean field `{m_i}` varies
across points at O(1) AND the m-partition is not coincident with the
layer.

Three cases:
- m constant → anchor-dominated → plain recovers layer → Δ ≈ 0
- m variable, layer-aligned → m-partition = layer → Δ ≈ 0
- m variable, layer-misaligned → plain rotates off layer → Δ > 0

### Two mechanism tests confirm
1. **K-sweep** (CC psi_two): plain 0.333 → 0.020 as K: 5 → 80; pEx
   holds 0.33–0.56. Δ grows +0.001 → +0.473.
2. **Tanh-Bernoulli β-sweep** (10-seed): three-phase Δ(β) picture —
   - Phase A (β ≲ 2): saturated, Δ ≈ 0
   - Phase B (β ≈ 3–10): mechanism active, β=5 Δ=+0.056 [+0.031, +0.080]
   - Phase C (β ≳ 20): fingerprint rank-degenerate, both plain & pEx crash

### H-DP extension (Fix 4, Durante et al. 2025)
- K̂ ≈ 10 inferred (θ₀-insensitive across 16× range)
- NMI 0.261 < finite-K 0.334 (arity mismatch vs binary truth)
- BUT posterior C ratio 3.48 > finite-K 2.51
- K=10 partition separates three LIME attribution regimes:
  - pure-low (n=85): sqft_lot absent → gate invisible
  - layer-mixed (n=89): sqft_lot mid-rank (0.134) → gate partial leak
  - pure-high (n=26): sqft_lot dominant (0.254) → gate openly leaking
- Two clusters (6, 7; 39 nodes) sit at the gate boundary — sub-population
  the binary layer collapses.

### Retractions / dead ends
- "regime (b) is structural" (§13d of POC) — encoder-induced, not universal
- "finite-K pEx structurally cannot help" (§14) — graph-specific
- "pEx detects smoothness" — refuted (CC XOR has largest Δ)
- "pEx detects decorrelation" — refuted (tanh high-β: low decorr, large Δ)
- 1/K vs 1/√K rate prediction — retracted
- Sinusoidal-sign Lipschitz probe — retired (parametrization collapses)

---

## Current open decision (was Peng meeting 2026-04-23)

Three questions to resolve before committing to a paper shape:
1. Lead **finite-K Δ** (binary-layer recovery) or **H-DP refinement**
   (three LIME regimes) as the headline?
2. Is four-dataset + K-sweep + tanh-sweep enough for **workshop/TMLR**,
   or develop a formal **modular-recovery-probability bound** for
   NeurIPS-main?
3. Keep β-sweep as robustness appendix, or drop?

---

## Queued follow-ups

| # | Direction | Cost | Payoff |
|---|---|---|---|
| 1 | Penalty sweep (scaffold penalty {0.10, 0.30, 0.50}) | half-day | Robustness validation; does three-regime LIME story collapse to two? |
| 2 | Predictive test on H-DP gate-boundary clusters (6,7) — do they have higher adversarial sensitivity on held-out? | 1–2 days | Promotes H-DP result from diagnostic to **predictive method** |
| 3 | Formal plain-SBM recovery-probability bound as f(m-variability, K, layer-alignment) | weeks | NeurIPS-main shape; sharper than K-sweep alone |
| 4 | Vary scaffold's OOD detector (e.g., LIME-style on SHAP perts) — does Gauss+SHAP null flip to hit? | days | Tests "encoder–scaffold = pert-manifold calibration" reduction |
| 5 | Higher-dim targeted rung-3 encoders sweeping multiple features — push C ratio without saturating | days | Could expand Phase B regime |
| 6 | Find rung-3 variant that reproduces LIME-attr saturation (NMI ≥ 0.9 at K=2) | speculative | Encoder-as-detector result with no pEx needed |

---

## Repo / artifacts pointer

- **Entry point:** `docs/sbm_explainer/README.md`
- **Narrative:** `docs/sbm_explainer/day_log_apr22.md`
- **Peng-ready writeup:** `docs/sbm_explainer/summary_apr22.md`
- **Theory:** `docs/sbm_explainer/cosine-expression.md`
- **Experiment registry:** `docs/sbm_explainer/results.md`
- **Follow-ups:** `docs/sbm_explainer/followup_directions.md`
- **Figures:** `docs/sbm_explainer/figures/{k_scaling,tanh_sweep,tanh_seedscale,lipschitz_reparam}.png`
- **Scripts:** `scripts/sbm_explainer/` (run_*, analyze_*, plot_*)
- **Results JSONs:** `results/sbm_explainer/{fingerprint_sbm,hdp_pex,compas_fingerprint,german_fingerprint,cc_fingerprint}/`
- **Reproducibility recipe:** §6 of `summary_apr22.md`

Working branch: `claude/busy-cannon-PDcAx` (clean as of this snapshot).
