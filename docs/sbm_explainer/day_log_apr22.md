# Day log — April 22 SBM-explainer thread

Written end-of-day to capture what was run, what broke, how it was
fixed, and how the mechanism story evolved. Intended as a continuity
artifact for tomorrow's Thursday meeting prep.

## Starting state (morning of April 22)

- Rung-3 fingerprint encoder finding (Δ=+0.048 on housing Gauss+LIME)
  replicated once on Slack COMPAS (Δ=+0.038, SHAP σ=0.1) the prior
  evening.
- Open questions: does the mechanism replicate on German and CC?
  Does a formal cosine-decomposition predict the observed Δ?
- Target: deliverable for Thursday April 23 meeting with Peng.

## What ran today

### 1. Slack German replication
- Script: `scripts/sbm_explainer/run_german_fingerprint.py`
- Pipeline identical to COMPAS: σ ∈ {0.1, 0.3, 1.0} Gaussian
  fingerprint, `Adversarial_Lime_Model` and `Adversarial_Kernel_SHAP_Model`
  from Slack, 3 fp × 5 SBM seeds.
- **Result:** LIME σ=1.0 Δ=+0.057 (C 1.36), SHAP σ=1.0 Δ=+0.071
  (C 1.05) — strongest SHAP effect in the set. σ ≤ 0.3 hit
  saturated-plain regime via loan-rate/gender correlation.

### 2. Slack CC replication
- Script: `scripts/sbm_explainer/run_cc_fingerprint.py`
- Both innocuous variants: `psi_one` (single binary proxy) and
  `psi_two` (XOR over two). d=102 post-preprocessing, n=200.
- **Result:** psi_one SHAP σ=0.1 Δ=+0.099 (direct housing analog).
  psi_two LIME σ=1.0 **Δ=+0.575** (plain 0.124 → pEx 0.699), five-to-
  tenx the other datasets. XOR innocuous amplifies Δ — directly
  contradicts "smoothness detector" interpretation.

### 3. Cosine decomposition (`docs/sbm_explainer/cosine-expression.md`)
- Full derivation with denominator: cos(e_i, e_j) decomposes as
  `m_i m_j/(s_i s_j) + a_i a_j/(K s_i s_j) + O(K^{-1/2})`.
- Earlier hand-wavy sketch predicted 1/K vs 1/√K decay in Δ between
  smooth and XOR regimes. **Denominator kills the distinction** —
  both regimes end up with O(1/√K) edge-level layer SNR. Prediction
  retracted before running the sweep.

### 4. K-scaling sweep across all 4 datasets
- Script: `scripts/sbm_explainer/run_k_scaling_sweep.py`
- K ∈ {5, 10, 20, 40, 80}, 3 fp × 5 SBM seeds per cell.
- **Bug during setup:** `ModuleNotFoundError: No module named 'get_data'`
  — SLACK wasn't on sys.path at import-time because the import was
  inside a setup function. Fix: moved `sys.path.insert(0, str(SLACK))`
  to module top.
- **Result:** CC psi_two plain NMI 0.333 → 0.020 as K: 5 → 80; pEx
  stays 0.33–0.56. Δ grows from +0.001 to +0.473 as plain collapses.
  §5 rate predictions **both failed** — Δ grows with K on CC and
  German, doesn't decay at all. This forced a mechanism rethink.
- Figure: `docs/sbm_explainer/figures/k_scaling.png`.

### 5. Mechanism rethink: anchor-dominance erosion (§6a of cosine-expression.md)
- At small K the fingerprint is anchor-dominated (e_i has 1 anchor
  coord, few refinements) — plain SBM recovers layer from the anchor
  alone → plain NMI high, pEx adds nothing.
- At large K the cosine normalization shrinks anchor weight to O(1/K)
  while the O(1) refinement-mean term takes over. If the
  refinement-mean field is not layer-aligned, plain rotates onto the
  unrelated partition. pEx holds via the conditional prior. **Δ is
  plain's failure rate, not signal magnitude.**
- This is the single unified mechanism. The earlier "two-regime
  decorrelation" story (from §3 of cosine-expression.md) was a
  special case.

### 6. Lipschitz sweep with sinusoidal-sign innocuous
- Script: `scripts/sbm_explainer/run_lipschitz_sweep.py`
- `f_inn_β(x) = sign(sin(β · x_u))`, β ∈ {0.25, …, 20}, K=20 fixed.
- **Result:** Δ positive but non-monotone across β, 3-seed CIs very
  wide. **β=0.25 and β=0.5 give identical output** — for standardized
  x ∈ [−3, 3] the sign(sin(βx)) threshold collapses to sign(x) at
  small β. Parametrization is bad.
- **Reparameterization attempt.** Proposed replotting on the physical
  axis E[n_flips] = β σ √(2/π)/π. User pushed back correctly:
  E[n_flips] is strictly monotone in β, so no x-axis reparam can
  turn a non-monotone Δ into monotone. Observed curve is flat + noisy
  at n=3 seeds, and the apparent "β=2/β=10 peaks vs β=1 trough" is
  inside the per-seed noise [0.26, 0.00, −0.01] vs [0.52, 0.12, 0.58].
- Figure: `docs/sbm_explainer/figures/lipschitz_reparam.png`.
  **Downgraded from mechanism test to robustness result.**

### 7. Tanh-Bernoulli probe (replacement Lipschitz sweep)
- Script: `scripts/sbm_explainer/run_tanh_sweep.py`
- `f_inn_β(x) = Bernoulli(sigmoid(β · x_u))`, global Lipschitz β/4.
- Smooth across all β; tests cases (1) → (3) of the unified mechanism
  as β grows.
- **Result:** Plain NMI monotonically degrades 0.73 → 0.39 from β=2
  to β=10; pEx holds at 0.65–0.73. Δ transitions cleanly from 0 at
  β ≤ 2 (saturated-plain via refinement degeneracy) to +0.26 at β=10
  (variable layer-orthogonal m-field).
- Figure: `docs/sbm_explainer/figures/tanh_sweep.png`.
- **This is the clean Lipschitz mechanism signature** the sinusoidal
  probe failed to produce. Upgraded (4) from robustness to second
  mechanism axis.

### 8. Theory unification (§3.1 of cosine-expression.md)
Tanh result forced a rewrite of §3. Original "decorrelation → Δ"
framing was a special case of a more general picture:

> pEx helps iff the refinement-mean field {m_i} varies across points
> at a scale dominating the O(1/K) anchor term, AND the m-partition
> is not coincident with the layer.

Three cases:
- m_i constant → anchor-dominated plain SBM recovers layer → Δ ≈ 0
- m_i variable, layer-aligned → m-partition = layer → Δ ≈ 0
- m_i variable, layer-misaligned → plain rotates off layer → Δ > 0

Everything observed reduces to which case applies:
- CC psi_two XOR: case (3) by construction (m varies on x_u1 ⊕ x_u2).
- Tanh low β: case (1) via refinement degeneracy.
- Tanh high β: case (3) via hard-step on x_u.
- German σ ≤ 0.3: case (2) via loan-rate/gender correlation.
- German σ = 1.0: case (3) partial.
- K-sweep: drives case (3) signal to dominance by diluting the
  anchor's cosine weight.

## Problems that came up and how they resolved

1. **K-sweep import error.** `get_data` not found. SLACK not on
   sys.path at import time. Fixed by module-level insert.
2. **Sinusoidal-sign probe parametrization collapses.** At small β on
   standardized data, sign(sin(βx)) = sign(x). Identified by observing
   β=0.25 and β=0.5 gave identical results. No fix — retired the
   probe in favor of tanh.
3. **Δ-reparameterization hypothesis didn't hold.** Proposed mapping
   β → n_flips to expose a monotone curve; n_flips is itself monotone
   in β, so nothing could be gained. User caught this before I
   committed a misleading figure. Resolution: reparam figure retained
   only to show that n=3-seed CIs overlap, demonstrating the non-
   monotonicity is seed noise.
4. **Buffering in background Python.** `python3 -u` needed for
   incremental log output during long-running sweeps. Without it,
   progress was invisible until completion. Checked via incremental
   `results.json` writes instead.
5. **Stale monitor notifications.** Two monitor timeouts fired during
   the session (K-sweep progress, Lipschitz progress) referencing
   tasks that had already completed. Ignored.
6. **Decorrelation-framing of mechanism turned out incomplete.** The
   original §3 framed Δ as driven by refinement decorrelation across
   δ. Tanh at high β has low decorrelation (refinements near-
   deterministic in x_u) but large Δ. Forced the unified §3.1
   reformulation (variability of m_i across i, not decorrelation
   across δ).

## What's running at end of day

All background work complete.

## 9. Tanh seed-scale + saturation (evening)

`scripts/sbm_explainer/run_tanh_seedscale.py` completed in ~15 min.
β ∈ {2, 5, 10} at 10 fp seeds, β ∈ {20, 50} at 5 fp seeds.

| β | n | plain | pEx | Δ | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 2  | 10 | 0.744 | 0.745 | +0.001 | [−0.001, +0.003] |
| **5**  | **10** | **0.615** | **0.670** | **+0.056** | **[+0.031, +0.080]** |
| **10** | **10** | **0.422** | **0.590** | **+0.168** | **[+0.050, +0.286]** |
| 20 | 5  | 0.203 | 0.429 | +0.226 | [−0.022, +0.474] |
| 50 | 5  | 0.033 | 0.222 | +0.189 | [−0.041, +0.419] |

**Three findings.**

1. **Knee resolved.** β=5 Δ = +0.056 [+0.031, +0.080], SEM = 0.011.
   Highest-confidence resolution of Δ > 0 in the entire thread
   outside Probe 2.
2. **β=10 is a bimodal phase transition.** Per-seed Δ from +0.01 to
   +0.56 — some seeds stay in case (1), others transition to case
   (3). Mean Δ still resolved above zero.
3. **Saturation surprise.** User's tanh→indicator prediction was
   that Δ would drop at β ≥ 20. Instead Δ plateaus at 0.17–0.23
   because plain *and* pEx drop together: plain collapses from 0.42
   to 0.03, pEx from 0.59 to 0.22. At β → ∞ the fingerprint becomes
   rank-degenerate (only anchor + proxy-sign carry info), which
   breaks pEx's advantage even though the anchor-dilution mechanism
   is still firing on plain. Directionally consistent with the
   prediction, manifests as "both drop" not "Δ drops."

**Three-phase Δ(β) picture** (revision to §6c of cosine-expression.md):
- Phase A (β ≲ 2): saturated-plain, Δ ≈ 0.
- Phase B (β ≈ 3–10): mechanism active, Δ rises.
- Phase C (β ≳ 20): fingerprint rank-degenerate, Δ plateaus but
  both plain and pEx crash.

Figure: `docs/sbm_explainer/figures/tanh_seedscale.png`.

**Implication for paper.** Phase B window is where the mechanism
is cleanly observable — roughly 1 < β·σ < 10 at σ=1.0. If a paper
claims "pEx detects scaffolded bias," the operating regime of
that claim is "innocuous function with moderate Lipschitz at
perturbation scale." Extreme step-function innocuous models break
the detector (both sides lose signal).

## State of the Thursday story (end of day)

**Empirical.**
- Four-dataset replication (housing, COMPAS, German, CC) with Δ > 0
  in middle-regime cells of every scaffold.
- Routing-mixture invariant: Δ > 0 iff OOD detector produces partial
  mixture.
- Saturated-plain regime appears on three datasets via different
  routes (proxy-correlation, refinement degeneracy, aligned m-field).

**Formal.**
- §2 cosine decomposition correct at edge level.
- §3.1 unified mechanism: pEx helps iff m-field varies and is
  layer-misaligned.
- Two independent tests of this mechanism:
  - K-sweep (vary fingerprint dim). Plain collapses on CC psi_two,
    pEx holds. Cleanest functional signature.
  - Tanh-sweep (vary innocuous Lipschitz). Plain collapses at β≥5,
    pEx holds. Second axis confirmation.
- Neither is a tight rate prediction — Δ grows sigmoidally with
  anchor-dilution, not as a simple power law. This is expected from
  §3.1 because Δ measures plain-SBM *failure probability*, not
  edge-level SNR.

**What's not supported.**
- Sinusoidal-sign Lipschitz rate prediction — retracted.
- "pEx detects smooth Lipschitz" — refuted (CC XOR has the largest Δ).
- "pEx detects decorrelation" — refuted (tanh high-β has low
  decorrelation and large Δ).

## Open questions for tomorrow
- Does the β=10 tanh result tighten when seed-scaled to 10? Currently
  Δ=+0.26±0.21 at n=3, so unclear if Δ>0 is resolved.
- Do β=20 and β=50 plateau at Δ≈0.25 or start dropping? User's
  "tanh-becomes-indicator" prediction says drop; unified mechanism
  says stay.
- Does the unified §3.1 mechanism survive if we try to write down a
  formal bound on plain SBM recovery probability as a function of
  (m-variability, K, layer-alignment)? If yes → NeurIPS-main-shaped.
  If no → TMLR/workshop.
