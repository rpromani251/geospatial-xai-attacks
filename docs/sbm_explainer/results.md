# Rung-3 fingerprint encoder — results

Three-rung encoder ladder from `design.md`, all on `sqft_lot` scaffold
(layer = >P75 = 9361), n = 200, K = 4, 8-NN cosine, 5 seeds, scaffold
penalty 0.20.

## Headline

| Rung | Encoder | log BF | plain NMI | finite-K pEx NMI | Δ | C ratio |
|---|---|---|---|---|---|---|
| 1 | LIME attr | +516 | 0.538 | 0.538 | +0.000 | 4.78 |
| 1 | SHAP attr | +356 | 0.095 | 0.099 | +0.004 | 1.10 |
| 2 | honest-booster leaves | +117 | 0.121 | 0.126 | +0.005 | 1.33 |
| 3 | fingerprint Gauss + LIME scaffold | **+145** | **0.287** | **0.334** | **+0.048** | **2.51** |
| 3 | fingerprint targeted + LIME scaffold | +2 | 0.063 | 0.063 | +0.000 | 1.16 |
| 3 | fingerprint Gauss + SHAP scaffold | −2 | 0.004 | 0.003 | −0.001 | 0.99 |
| 3 | fingerprint targeted + SHAP scaffold | −3 | 0.012 | 0.012 | +0.000 | 1.01 |

Reproduced from `results/sbm_explainer/fingerprint_sbm/comparison.md`.

## Three findings

### 1. Regime (b) was encoder-induced, not universal

§13d of `plain-sbm-poc.md` concluded that the SBM posterior is blind to
the layer in the transition region and the failure is
likelihood-structural. That conclusion was drawn entirely from SHAP/LIME
attribution graphs. Rung 3 gives a graph with log BF +145 (between
LIME at p=0.13 and LIME at p=0.16 on the penalty sweep, both regime b)
and a posterior C ratio of **2.51** — not even close to the ~1.01–1.04
that defines regime b. The posterior *does* carry layer structure when
the encoder gives it something coherent to block on. Regime b is a
property of attribution-vector geometry, not of SBM inference on this
problem.

### 2. finite-K pEx null was graph-specific, not structural

Six orthogonal stress tests (α, γ, noisy, subpop, held-out, penalty) all
returned Δ = 0 to seed precision — but all ran on SHAP/LIME attribution
graphs whose posterior C ratio was ≤ 1.1. The Dirichlet prior has
nothing to act on when the posterior doesn't encode the layer at all,
and nothing to nudge when the posterior is already saturated. Rung 3's
Gauss+LIME graph is in the middle regime (C ratio 2.51, MAP NMI 0.287 —
neither blind nor saturated), and the prior moves the posterior:
Δ = +0.048 ± 0.008 at 6σ.

The stronger claim from §14 — "finite-K pEx is structurally unable to
differ from plain" — is retracted. The correct statement is: **finite-K
pEx differs from plain only on graphs whose posterior has unsaturated
layer awareness. SHAP/LIME attribution graphs do not produce such
graphs; fingerprint-under-scaffold graphs can.**

### 3. The signal is the scaffold's routing mixture, not gate exposure per se

Probe 1 (`run_ood_flip_probe.py`) + honest-model sanity check
(`run_honest_fingerprint_check.py`, April 22) revises the mechanism
story. Earlier drafts of this doc attributed the Gauss+LIME non-null
to "LIME OOD lets perts through as real → BiasedModel → gate visible
in fingerprint." The routing check falsifies that:

    LIME detector classifies  0% of σ=0.1·feat_std Gauss perts as real
                                                       → routed to INNOCUOUS
    SHAP detector classifies 73% of the same perts as real
                                                       → routed to BIASED

The LIME config routes perts **away** from the gate, not toward it.

The honest/biased/innocuous sanity check then rules out the obvious
alternative hypotheses:

| Configuration | plain NMI | pEx NMI | Δ | C ratio |
|---|---|---|---|---|
| honest XGBoost | 0.007 | 0.009 | +0.002 | 1.03 |
| BiasedModel only (all inputs → biased) | 0.004 | 0.015 | +0.011 | 1.04 |
| InnocuousModel only (all inputs → innocuous) | 0.007 | 0.015 | +0.007 | 1.06 |
| **scaffold (LIME detector)** | **0.287** | **0.334** | **+0.048** | **2.50** |

All three single-model configs are flat. The signal requires the
scaffold — specifically, the **routing mixture** that a strict LIME
detector produces.

**What's actually happening.** Under the LIME detector, `f_scaffold(x_i)`
routes x_i (a real point) to `BiasedModel` and every perturbation
`x_i + δ_k` to `InnocuousModel`. Fingerprint becomes

    e_i = [ f_biased(x_i), f_innocuous(x_i + δ_1), …, f_innocuous(x_i + δ_K) ]

The first coordinate carries a clean binary gate offset (high-sqft
points get honest − 0.3, low-sqft points get honest). The remaining
K coordinates are smooth values from a regression that can't see
sqft_lot, UTM_X, UTM_Y. Cosine 8-NN groups points that share **both**
the same anchor-offset and a similar smooth profile — and that's the
layer.

Under the SHAP detector, 73% of perts route to BiasedModel. Those
perts sometimes flip across the gate because δ on sqft_lot alone can
cross P75. The fingerprint becomes noisy at every coordinate, cosine
geometry destroys, and the graph has no block structure.

**Rung-3 is specifically a scaffold-attack detector.** It's not a
generic smoothness / feature-geometry detector (honest and innocuous
configs both null), and it's not gate visibility alone (biased-only
null). The pEx Δ is positive *because and only because* the scaffold's
routing produces the characteristic "one gate-exposed anchor + K
gate-hidden refinements" fingerprint. The detector must be strict
enough to keep perts out of the biased path (LIME-style), or the
signal degrades (SHAP-style).

- **Targeted grids are low** for a separate reason: only 7 dims, all
  varying one feature, near-step response at P75, many identical
  fingerprints. Cosine 8-NN clusters are geometry-dominated, not
  scaffold-dominated.

### 4. Seed-scaling confirms the finite-K pEx Δ

Probe 2 (`run_gauss_lime_seedscale.py`, 20 seeds on the saved
Gauss+LIME graph):

    plain NMI  = 0.2942 ± 0.0181  95% CI [0.262, 0.320]
    pEx   NMI  = 0.3372 ± 0.0078  95% CI [0.326, 0.350]
    Δ (pEx-pln)= 0.0429 ± 0.0180  95% CI [+0.017, +0.076]
    one-sample t vs Δ=0: t = 10.68, p = 1.8 × 10⁻⁹

The five-seed estimate (0.048) reproduces within noise. Δ is
separable from zero at p < 10⁻⁸.

## What this means for the pEx direction

Our finite-K pEx isn't the methodological dead-end §14 called it. It's
a modest but real extractor on the right kind of graph — +0.048 NMI on
top of plain 0.287, direction correct, std separable from zero.

But +0.048 is modest, and the real story at rung 3 is the encoder: the
encoder alone (via plain SBM) goes from 0 (attribution null) to 0.287
(fingerprint graph). The finite-K pEx delta is noise-level compared to
the encoder delta.

## H-DP pEx (Durante et al. 2025, Fix 4) — executed

Same Gauss+LIME graph, replaced our finite-K Dirichlet-Categorical
prior with Durante's two-level DP hierarchy (`prior="HDP"` in
`papers/pexsbm/pex-sbm.R`). Five seeds × 4000 iter, burn-in 2000,
θ = 0.5, θ₀ = 4. Posterior co-clustering computed from all 2000 burned
samples per seed, then averaged across seeds.

### Three-method comparison

| Method | K | NMI vs binary layer | AMI | ARI | Posterior C ratio |
|---|---|---|---|---|---|
| plain SBM | 4 fixed | 0.287 ± 0.019 | — | — | 2.51 |
| finite-K Dirichlet pEx | 4 fixed | **0.334 ± 0.008** | — | — | 2.51 |
| H-DP pEx (Durante) | ~10 inferred | 0.261 ± 0.004 | 0.244 | 0.171 | **3.48** |

Finite-K pEx wins on binary-layer NMI (it's anchored at K=4, matching
the layer's arity). H-DP is partition-finer (K̂ = 10 every seed) so the
arity mismatch mechanically depresses NMI despite a **denser** posterior
co-clustering alignment (C ratio 3.48 vs 2.51).

### θ₀ sweep closes the hyperparameter loophole

`run_hdp_pex_sweep.R` + `analyze_hdp_sweep.py` across
θ₀ ∈ {0.25, 0.5, 1.0, 2.0} × 3 seeds:

| θ₀ | K̂ | NMI | AMI | ARI |
|---|---|---|---|---|
| 0.25 | 9.3 | 0.261 | 0.244 | 0.171 |
| 0.50 | 9.7 | 0.263 | 0.248 | 0.169 |
| 1.00 | 9.7 | 0.260 | 0.242 | 0.168 |
| 2.00 | 9.7 | 0.261 | 0.243 | 0.171 |

H-DP is insensitive across a 16× prior range. K̂ stays ~9-10; NMI
stays ~0.26. Not a bad-prior artifact — the sampler is genuinely
seeing K≈10 in the posterior.

### H-DP K=10 is a refinement of the K=4 story, not a competitor

Contingency of the H-DP seed-0 partition against the binary layer
(`analyze_hdp_refinement.py`):

| cluster | n | low/high | purity |
|---|---|---|---|
| 1 | 63 | 62/1 | 0.98 |
| 2 | 11 | 11/0 | 1.00 |
| 3 | 50 | 44/6 | 0.88 |
| 5 | 9 | 9/0 | 1.00 |
| **6** | **32** | **18/14** | **0.56** |
| **7** | **7** | **4/3** | **0.57** |
| 9 | 24 | 0/24 | 1.00 |
| (4, 8, 10) | 1,1,2 | — | noise |

Seven of ten clusters are ≥90% layer-pure. Two clusters (6, 7, ~20% of
nodes combined) sit at the gate boundary. Four are tiny (size ≤ 2) —
the DP burning mass on outliers.

### Per-cluster attribution profiles — the useful part

The 200 nodes have saved SHAP/LIME attributions under the adversarial
scaffold (`results/scaffold_sqft_lot/adv/`). Grouped by H-DP cluster
role:

**LIME attributions (adv), grouped:**

| group | n | top-3 mean \|attr\| |
|---|---|---|
| pure-low (≥90%) | 85 | UTM_Y (0.126), view (0.110), sqft_living (0.096) |
| pure-high (≥90%) | 26 | **sqft_lot (0.254)**, UTM_Y (0.131), view (0.116) |
| layer-mixed | 89 | sqft_lot (0.134), UTM_Y (0.117), view (0.106) |

**SHAP attributions (adv), grouped:**

| group | n | top-3 mean \|attr\| |
|---|---|---|
| pure-low (≥90%) | 85 | sqft_living (0.063), grade (0.062), age (0.061) |
| pure-high (≥90%) | 26 | sqft_living (0.088), sqft_lot (0.080), grade (0.079) |
| layer-mixed | 89 | sqft_living (0.073), grade (0.066), age (0.052) |

The LIME row is the story. The H-DP refinement separates three distinct
**attribution regimes** of the scaffold:

- **pure-low:** sqft_lot essentially absent from the top; LIME attributes
  spatial + amenity features. The scaffold gate is invisible here.
- **layer-mixed:** sqft_lot mid-rank (0.134). The gate is partially
  leaking.
- **pure-high:** sqft_lot dominates (0.254) — 2× the mixed group and
  ~25× the pure-low group. The gate is openly leaking through LIME.

SHAP is weaker: sqft_lot appears in pure-high (0.080) but not in
pure-low, consistent with the lower SHAP-level gate visibility we
already saw. The cluster-6 / cluster-7 "gate-boundary" sub-population
has a distinctly intermediate attribution profile, which is exactly
the refinement the binary layer cannot express.

Artifacts: `results/sbm_explainer/hdp_pex/cluster_attributions_{shap,lime}.csv`.

### Framing for the writeup

- **Headline stays finite-K pEx @ K=4.** It is the best recovery of the
  binary layer we designed (NMI 0.334, Δ +0.048 over plain at 6σ).
- **H-DP adds one genuine second-order finding.** The K=10 posterior is
  a refinement — three attribution regimes (absent / partial / openly
  leaking LIME gate) and ~20% gate-boundary nodes the binary layer
  collapses. Worth one slide, not the lead.

## Open questions queued for next thread

- **Closed (Probe 1).** The OOD-detector swap as originally framed is a
  no-op in our architecture (scaffold identity = detector), but the
  three-config sanity check (honest / biased / innocuous / scaffold)
  established that the signal requires the scaffold's routing mixture,
  not any single-model configuration.
- **Closed (Probe 2).** 20-seed run gives Δ = 0.043 ± 0.018, 95% CI
  [0.017, 0.076], t = 10.68, p = 1.8e-9. Five-seed estimate
  reproduces.
- **Closed — Slack Communities-and-Crime reproduction.** d=102 binary
  classifier, `racePctWhite` as protected layer, n=200. Two innocuous
  variants: `psi_one` (single binary proxy) and `psi_two` (XOR over
  two proxies). **Largest effects in the set.** psi_one SHAP σ=0.1 is a
  direct housing-analog middle-regime cell: plain 0.270 → pEx 0.369,
  Δ=+0.099, C ratio 1.93. psi_one LIME σ=1.0 Δ=+0.230. **psi_two LIME
  σ=1.0 Δ=+0.575** (plain 0.124 → pEx 0.699) — XOR innocuous amplifies
  the pEx signal 5–10× over smoother variants. Saturated-plain regime
  present again on LIME σ∈{0.1, 0.3}. Mechanism implication: pEx is
  not detecting function smoothness — less-smooth innocuous models
  produce bigger Δ, not smaller. Artifacts:
  `results/sbm_explainer/cc_fingerprint/`.
- **Closed — Slack German reproduction.** Ran the same pipeline on
  Slack's `german_experiment` scaffold (Gender layer, continuous
  `LoanRateAsPercentOfIncome` as innocuous feature, n=100 test points).
  **Third-dataset replication, both scaffolds fire at σ=1.0:** LIME
  Δ=+0.057 (C 1.36), SHAP Δ=+0.071 (C 1.05) — the two strongest Δ's in
  the set. LIME σ∈{0.1, 0.3} hits the COMPAS-like saturated-plain
  regime (NMI 0.54, C 13) via correlation between loan-rate and Gender.
  SHAP σ∈{0.1, 0.3} is null despite ~32% partial routing — partial
  routing is necessary but not sufficient; mixture geometry matters.
  Δ magnitudes across all three datasets (housing 0.048, COMPAS 0.038,
  German 0.057–0.071) occupy the same range. Artifacts:
  `results/sbm_explainer/german_fingerprint/`.
- **Closed — Slack COMPAS reproduction.** Ran the identical pipeline on
  Slack's original `Adversarial_Lime_Model` / `Adversarial_Kernel_SHAP_Model`
  with race as the protected layer, Gaussian fingerprints at σ ∈
  {0.1, 0.3, 1.0} on scaffold class-1 probability. **Replicates with
  role reversal:** COMPAS Gauss+SHAP at σ=0.1 gives plain 0.303 → pEx
  0.340, Δ=+0.038, C ratio 1.94 — matches housing's Gauss+LIME plain
  0.287 → pEx 0.334, Δ=+0.048, ratio 2.50 within seed noise. Invariant
  confirmed: Δ > 0 iff OOD detector produces a partial routing mixture
  (neither all-biased nor all-innocuous). Housing: LIME detector is
  the strict one; COMPAS: SHAP detector is. Role reversal is
  detector-manifold calibration, not explainer-specific. COMPAS also
  shows a new saturated-plain regime (LIME σ=0.1/0.3: plain NMI 0.574
  via the `unrelated_column_one` race proxy — pEx adds nothing).
  Details and table in `followup_directions.md` §1 and
  `summary_apr22.md` §4a.
- **Closed — K-scaling sweep across all 4 datasets.** K ∈ {5, 10, 20,
  40, 80}, 3 fp seeds × 5 SBM seeds. Revealed **anchor-dominance
  erosion** mechanism: CC psi_two plain NMI 0.333 (K=5) → 0.020 (K=80)
  while pEx stays 0.33–0.56; Δ grows from +0.001 to +0.473 as plain
  collapses. §5 rate-based predictions ($1/K$ vs $1/\sqrt{K}$) fail.
  Correct mechanism: at small K the 1D-ish fingerprint is anchor-
  dominated (plain alone recovers layer); at large K cosine
  normalization shrinks the anchor to $O(1/K)$ weight, plain loses the
  layer, pEx retains via conditional prior. Δ is plain's failure rate,
  not signal magnitude. Load-bearing mechanism test for the paper.
  Details in `cosine-expression.md` §6a.
- **Closed — Lipschitz (β) sweep, downgraded to robustness result.**
  `run_lipschitz_sweep.py` CC/LIME σ=1.0 K=20 fixed, parametric
  innocuous $\mathrm{sign}(\sin(\beta x_u))$, β ∈ {0.25…20}, 3 fp
  seeds. After reparameterization on $\mathbb{E}[n_{\text{flips}}]
  \propto \beta\sigma$ (strictly monotone in β), Δ is positive but
  flat across the full decorrelation range (0.06 → 5.1 expected flips)
  with all 95% CIs overlapping zero. Cannot resolve functional form at
  n=3 seeds. **Claim is "mechanism fires broadly" not "mechanism
  scales predictably."** Figure: `docs/sbm_explainer/figures/
  lipschitz_reparam.png`. Details in `cosine-expression.md` §6b–c.
- **Running — Tanh probe.** `run_tanh_sweep.py` CC/LIME σ=1.0 K=20,
  $\mathrm{Bernoulli}(\mathrm{sigmoid}(\beta x_u))$, smooth global
  Lipschitz $\beta/4$. Second robustness probe; outcome not load-
  bearing either way.
- **Penalty sweep.** Does Δ +0.043 survive at scaffold penalties 0.10 /
  0.30 / 0.50? Half-day cost, cheap robustness check.
- **Predictive test.** Do cluster-6/7 "gate-boundary" nodes (from H-DP
  K=10 refinement) have higher adversarial sensitivity on held-out
  data than pure-low or pure-high clusters? 1–2 days; moves the
  refinement from interpretive to predictive.
- Can we build a rung-3 encoder that deliberately maximizes C ratio
  (e.g., higher-dim targeted grids sweeping multiple features)?
- Do any rung-3 variants reproduce the LIME-attr saturation regime
  (NMI → 0.9+ at K=2)?

## Files

| Path | Contents |
|---|---|
| `scripts/sbm_explainer/run_activation_sbm.py` | rung 2 pipeline (honest-booster leaves) |
| `scripts/sbm_explainer/run_fingerprint_sbm.py` | rung 3 pipeline (both fingerprint variants × both scaffolds) |
| `scripts/sbm_explainer/export_hdp_inputs.py` | reorders Gauss+LIME graph for H-DP R sampler |
| `scripts/sbm_explainer/run_hdp_pex.R` | H-DP pEx on Gauss+LIME (5 seeds × 4000 iter) |
| `scripts/sbm_explainer/hdp_posterior_C.R` | posterior co-clustering from all burned samples |
| `scripts/sbm_explainer/run_hdp_pex_sweep.R` | θ₀ sweep (0.25 / 0.5 / 1.0 / 2.0) × 3 seeds |
| `scripts/sbm_explainer/analyze_hdp_pex.py` | scores H-DP z_hats vs layer |
| `scripts/sbm_explainer/analyze_hdp_sweep.py` | per-cell + aggregated sweep table |
| `scripts/sbm_explainer/analyze_hdp_refinement.py` | H-DP × layer contingency |
| `scripts/sbm_explainer/analyze_hdp_attributions.py` | per-cluster SHAP/LIME attribution profiles |
| `scripts/sbm_explainer/run_gauss_lime_seedscale.py` | Probe 2 — 20-seed CI on Gauss+LIME |
| `scripts/sbm_explainer/run_ood_flip_probe.py` | Probe 1 — routing check + detector swap |
| `scripts/sbm_explainer/run_honest_fingerprint_check.py` | mechanism sanity — honest / biased / innocuous null |
| `results/sbm_explainer/activation_sbm/` | rung 2 outputs |
| `results/sbm_explainer/fingerprint_sbm/` | rung 3 outputs (`summary.json`, `comparison.md`) |
| `results/sbm_explainer/hdp_pex/` | H-DP outputs (`z_hats.csv`, `C_post.csv`, `results.json`, `sweep_*.csv`, `cluster_attributions_*.csv`, seed RDS) |
| `papers/pexsbm/` | Durante et al. 2025 reference R sampler |
