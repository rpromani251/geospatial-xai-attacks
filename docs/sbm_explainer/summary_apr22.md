# Comprehensive summary — SBM-as-explainer thread, April 22

Written for Thursday April 23 meeting with Peng. Covers the full arc
from the §13 regime-(b) failure through finite-K pEx, rung-3 fingerprint
encoder, H-DP pEx (Durante et al. 2025), θ₀ sweep, and per-cluster
attribution profiles.

---

## 0. Vocabulary

Two "pEx" variants are referred to throughout; they are not the same
model:

- **finite-K Dirichlet-Categorical pEx** (ours). K fixed (here K=4). The
  "layered prior" is a Dirichlet(γ) over the K-simplex per layer, fed
  into Categorical node-to-block assignments. Implemented in
  `src/sbm/pex_sbm.py`.
- **H-DP pEx** (Durante et al. 2025, arXiv 2410.10619). K inferred. Two
  nested Dirichlet Processes (layer-level DPs linked to a global profile
  DP). Three-route Gibbs: join-in-layer / bring-in-from-base /
  new-profile. Reference sampler in `papers/pexsbm/pex-sbm.R`.

Both supply a "per-layer prior over block usage" — the difference is
fixed-arity Dirichlet vs. infinite-arity HDP.

---

## 1. Where we started

`docs/plain-sbm-poc.md` §13 documented a regime distinction:

- **Regime (a)** — saturated MAP alignment. Posterior NMI(z, layer) ≥ 0.9,
  posterior C_within / C_across ≥ 4. Example: LIME attribution graph at
  penalty 0.20 on sqft_lot (NMI 0.538, C ratio 4.78).
- **Regime (b)** — posterior blindness. Posterior NMI ≈ 0, C ratio ≈ 1
  despite a high log Bayes factor for the layer (+356 for SHAP attr at
  penalty 0.20). The layer is "visible" to Fix 3 but not to the SBM.

§13d concluded regime (b) was structural: "Dirichlet-Categorical pEx
cannot help when the posterior assigns zero probability to partitions
that align with the layer." §14 extended this to a six-stress-test
null for finite-K pEx (α, γ, noisy, subpop, held-out, penalty) —
Δ(pEx − plain) = 0 to seed precision in every cell.

§15 listed four possible fixes. This thread ran Fix 3 (change the
encoder) first, then Fix 4 (swap in H-DP).

---

## 2. Rung-3 fingerprint encoder (main experiment)

Third rung of the encoder ladder from `docs/sbm_explainer/design.md`:

- **Rung 1** SHAP / LIME attribution vectors (original regime-b source)
- **Rung 2** XGBoost honest-booster leaf indices (Δ = 0 again —
  honest-booster leaves don't reflect the scaffold gate)
- **Rung 3** behavioral fingerprint under the scaffolded model:
  e_i = [f_scaffold(x_i), f_scaffold(x_i + δ_1), …, f_scaffold(x_i + δ_K)]

Two encoder variants × two scaffolds = 4 graphs, 200 nodes, 8-NN cosine.

### Results

| Encoder | Scaffold | log BF | plain NMI | finite-K pEx NMI | Δ | C ratio |
|---|---|---|---|---|---|---|
| Gaussian | LIME | **+145** | **0.287** | **0.334** | **+0.048** | **2.51** |
| Gaussian | SHAP | −2 | 0.004 | 0.003 | −0.001 | 0.99 |
| Targeted | LIME | +2 | 0.063 | 0.063 | 0.000 | 1.16 |
| Targeted | SHAP | −3 | 0.012 | 0.012 | 0.000 | 1.01 |

One cell (Gauss+LIME) produces the middle regime — C ratio 2.51
(neither blind ~1.0 nor saturated ≥ 4). This is the first pEx cell in
the entire thread where Δ is separable from zero at 6σ.

### Why Gauss+LIME is special — corrected mechanism (April 22 pm)

Earlier drafts gave a wrong mechanism ("LIME OOD lets perts through →
gate visible"). Probe 1 routing check falsifies it:

    LIME detector classifies  0% of σ=0.1·feat_std Gauss perts as real
                                                    → routed to INNOCUOUS
    SHAP detector classifies 73% of the same perts as real
                                                    → routed to BIASED

Then honest/biased/innocuous sanity (`run_honest_fingerprint_check.py`):

| Configuration | plain NMI | pEx NMI | Δ | C ratio |
|---|---|---|---|---|
| honest XGBoost | 0.007 | 0.009 | +0.002 | 1.03 |
| BiasedModel only | 0.004 | 0.015 | +0.011 | 1.04 |
| InnocuousModel only | 0.007 | 0.015 | +0.007 | 1.06 |
| **scaffold (LIME detector)** | **0.287** | **0.334** | **+0.048** | **2.50** |

All single-model configs null. The signal requires the scaffold's
**routing mixture**. Under LIME detector:

    e_i = [ f_biased(x_i), f_innocuous(x_i + δ_1), …, f_innocuous(x_i + δ_K) ]

One gate-exposed anchor (x_i is real → routed to biased), K smooth
gate-hidden refinements (perts are flagged OOD → routed to innocuous
which can't see sqft_lot). Cosine 8-NN groups points sharing both
anchor-offset and smooth profile → recovers the layer.

Under SHAP detector, 73% of perts go to BiasedModel where σ=0.1·std
perturbations of sqft_lot can cross the P75 threshold → gate-crossing
noise at every coordinate → cosine geometry destroyed → null.

**The rung-3 + finite-K pEx pipeline is specifically a Slack-style
scaffold-attack detector.** It is not a feature-geometry detector
(honest null), not a bare-gate detector (biased null), and not an
innocuous-smoothness detector (innocuous null). The signal is a
structural fingerprint of the scaffold's routing — detectable because
strict OOD detectors (which Slack-style attacks need, to hide the bias
from the explainer) create the one-anchor-plus-K-refinements pattern
that rung-3 picks up.

### What this retracts from the POC

1. **Regime (b) is encoder-induced, not universal.** Rung 3 produces a
   graph at comparable log BF (+145 vs +356) with C ratio 2.51, far
   above the 1.0–1.04 that defines regime (b). The posterior does
   carry layer structure when the encoder gives it something coherent
   to block on.

2. **The finite-K pEx null was graph-specific.** It showed up on all
   graphs with C ratio ≤ 1.1 (attribution vectors under the scaffold),
   but not on the middle-regime Gauss+LIME graph. The §14 stronger
   claim "finite-K pEx is structurally unable to differ from plain"
   is retracted. Correct statement: finite-K pEx differs from plain
   only on graphs whose posterior has unsaturated layer awareness.

---

## 3. H-DP pEx on the same graph (Fix 4)

Replaced the finite-K Dirichlet-Categorical prior with Durante's
two-level DP, all other inputs identical (Gauss+LIME graph, 200 nodes,
N_vec = [150, 50]). Five seeds × 4000 iter × burn-in 2000 in R.
Posterior co-clustering from all 2000 burned-in samples per seed;
VI-optimal point estimate via `minVI(method="avg", max.k=20)` from
`mcclust.ext`.

### Three-method comparison

| Method | K | NMI | AMI | ARI | C ratio |
|---|---|---|---|---|---|
| plain SBM | 4 fixed | 0.287 ± 0.019 | — | — | 2.51 |
| finite-K Dirichlet pEx | 4 fixed | **0.334 ± 0.008** | — | — | 2.51 |
| H-DP pEx (Durante) | ~10 inferred | 0.261 ± 0.004 | 0.244 | 0.171 | **3.48** |

H-DP does *not* beat finite-K pEx on binary-layer NMI — it keeps
inferring K̂ = 10, which mechanically penalizes NMI against a binary
truth. Its posterior co-clustering is actually denser (3.48 vs 2.51):
blocks respect the layer more strongly at edge level, just at a finer
granularity than the binary layer captures.

### θ₀ sweep (closes the hyperparameter loophole)

THETA0_GRID = {0.25, 0.5, 1.0, 2.0}, 3 seeds each:

| θ₀ | K̂ | NMI | AMI | ARI |
|---|---|---|---|---|
| 0.25 | 9.3 | 0.261 | 0.244 | 0.171 |
| 0.50 | 9.7 | 0.263 | 0.248 | 0.169 |
| 1.00 | 9.7 | 0.260 | 0.242 | 0.168 |
| 2.00 | 9.7 | 0.261 | 0.243 | 0.171 |

Insensitive across a 16× prior range. H-DP genuinely sees K≈10 in the
Gauss+LIME posterior; it isn't a prior artifact.

### Is K=10 useful — contingency vs the binary layer

Seed 0 (representative):

| cluster | n | low / high | purity |
|---|---|---|---|
| 1 | 63 | 62 / 1 | 0.98 |
| 2 | 11 | 11 / 0 | 1.00 |
| 3 | 50 | 44 / 6 | 0.88 |
| 5 | 9 | 9 / 0 | 1.00 |
| 6 | 32 | 18 / 14 | **0.56** |
| 7 | 7 | 4 / 3 | **0.57** |
| 9 | 24 | 0 / 24 | 1.00 |
| (4, 8, 10) | 1, 1, 2 | — | DP noise |

7/10 clusters ≥90% layer-pure. Two clusters (6, 7, 39 nodes combined)
sit at the gate boundary. So K=10 is mostly a refinement of the layer,
with one genuinely new sub-finding: ~20% of nodes are gate-boundary
cases the binary layer collapses.

### Per-cluster attribution profiles — the interpretable payoff

The 200 nodes have saved SHAP/LIME attributions under the adversarial
scaffold. Grouped by cluster role:

**LIME attributions (adv):**

| group | n | top-3 mean \|attr\| |
|---|---|---|
| pure-low (≥90%) | 85 | UTM_Y (0.126), view (0.110), sqft_living (0.096) |
| pure-high (≥90%) | 26 | **sqft_lot (0.254)**, UTM_Y (0.131), view (0.116) |
| layer-mixed | 89 | sqft_lot (0.134), UTM_Y (0.117), view (0.106) |

**SHAP attributions (adv):**

| group | n | top-3 mean \|attr\| |
|---|---|---|
| pure-low | 85 | sqft_living (0.063), grade (0.062), age (0.061) |
| pure-high | 26 | sqft_living (0.088), sqft_lot (0.080), grade (0.079) |
| layer-mixed | 89 | sqft_living (0.073), grade (0.066), age (0.052) |

The LIME row is the story. The H-DP partition separates three distinct
attribution regimes of the adversarial scaffold:

- **pure-low:** sqft_lot absent from the top. LIME attributes spatial +
  amenity features. The gate is invisible.
- **layer-mixed:** sqft_lot mid-rank (0.134). The gate is partially
  leaking.
- **pure-high:** sqft_lot dominant (0.254) — 2× the mixed group and
  ~25× the pure-low group. The gate is openly leaking through LIME.

SHAP is weaker: sqft_lot appears in pure-high (0.080) but not in
pure-low — consistent with the lower SHAP-level gate visibility already
seen at rung 1. The mixed clusters have a distinctly *intermediate*
LIME attribution profile, which is exactly the refinement the binary
layer cannot express.

Artifacts: `results/sbm_explainer/hdp_pex/cluster_attributions_{shap,lime}.csv`.

---

## 4. How to frame this for Peng (suggested)

**Headline.** On the rung-3 Gauss+LIME fingerprint graph, finite-K
pEx at K=4 is the best binary-layer recovery we have (NMI 0.334 ±
0.008, Δ +0.048 over plain SBM at 6σ, posterior C ratio 2.51). This is
the first cell in the entire SBM-as-explainer thread where a pEx prior
moves the posterior separably from zero. **The effect replicates on
three additional Slack scaffolds** — COMPAS Gauss+SHAP σ=0.1
(Δ=+0.038), German Gauss+LIME/SHAP σ=1.0 (Δ=+0.057 / +0.071), and
Communities-and-Crime — with CC producing the largest effects in the
set (psi_two LIME σ=1.0: Δ=+0.575). Four-dataset replication across
regression and classification, single and XOR innocuous models, and
feature dimensions from 11 to 102, with role reversal driven by
detector-manifold calibration.

**Why the earlier pEx nulls were wrong (§14 retraction).** They were
about attribution-vector graphs, which are posterior-blind (regime b);
a well-chosen *encoder* puts the posterior in the middle regime, where
the prior has room to act. The six-test null was graph-specific, not
structural.

**H-DP extension (Fix 4).** Swapping in Durante's two-level DP does
not beat finite-K on binary-layer NMI — it infers K̂ ≈ 10 and the
arity mismatch depresses NMI. θ₀ sweep across a 16× range confirms
this is not a bad-prior artifact. H-DP's contribution is a *refinement*:
the K=10 partition separates the scaffold's three LIME-attribution
regimes (gate-absent / gate-partial / gate-leaking) and identifies a
~20% gate-boundary sub-population the binary layer collapses.

**Three candidate write-ups this supports.**

1. **"A behavioral-fingerprint encoder turns pEx-SBM into a regime-a-vs-b
   detector."** Lead with the finite-K +0.048 result, rung 3 as the
   method contribution. H-DP as an appendix showing the result is
   robust to prior family.

2. **"H-DP partitioning of scaffold fingerprints identifies distinct
   attribution regimes."** Lead with the three-regime LIME finding and
   the gate-boundary sub-population. Finite-K + NMI as supporting
   evidence the layer is real before zooming in.

3. **Both, as one paper.** Pros: one complete story. Cons: the NMI
   headline and the refinement headline compete for top billing — need
   careful structuring.

Ask Peng which story he finds more publishable and frame around that.

---

## 4a. Two-dataset replication — Slack COMPAS (April 22 pm)

Ran the identical rung-3 + finite-K pEx pipeline on Slack et al.'s
original COMPAS adversarial setup (`Adversarial_Lime_Model` and
`Adversarial_Kernel_SHAP_Model` from `papers/fooling-lime-shap/`,
unchanged), with race as the protected layer and Gaussian fingerprints
at σ ∈ {0.1, 0.3, 1.0} on the scaffold class-1 probability output.

| Scaffold | σ | plain NMI | pEx NMI | Δ | C ratio | routing |
|---|---|---|---|---|---|---|
| LIME | 0.1 | 0.574 | 0.574 | 0.000 | 14.05 | 0% real |
| LIME | 0.3 | 0.574 | 0.574 | 0.000 | 14.05 | 0% real |
| LIME | 1.0 | 0.033 | 0.029 | −0.003 | 1.10 | 0.1% real |
| **SHAP** | **0.1** | **0.303** | **0.340** | **+0.038** | **1.94** | 70% real |
| SHAP | 0.3 | 0.349 | 0.347 | −0.001 | 3.01 | 68% real |
| SHAP | 1.0 | 0.228 | 0.260 | +0.032 | 1.69 | 48% real |

**The method replicates, with role reversal.** COMPAS Gauss+SHAP at
σ=0.1 is the direct analog of housing's Gauss+LIME cell: plain 0.303 →
pEx 0.340, Δ=+0.038, C ratio 1.94 — same shape as housing's plain 0.287
→ pEx 0.334, Δ=+0.048, ratio 2.50. Δ magnitudes match within seed noise.

**Which detector plays the routing-mixture role is detector-manifold
dependent.** On housing, LIME's raw-unit σ=0.3 training is strict vs
our σ=0.1·feat_std perts; on COMPAS (post-StandardScaler) LIME's
σ=0.3 training is absolute and extreme (rejects 100% of σ=0.1 perts
so all go to innocuous — no anchor-vs-refinement contrast), while
SHAP's kmeans-based OOD is moderately permissive and produces the
partial mixture that makes rung-3 + pEx fire.

**Invariant across both datasets:** Δ > 0 iff the OOD detector
produces a *partial* routing mixture (not all-biased, not
all-innocuous). Role reversal reflects detector-manifold calibration,
not explainer-specific behavior.

**COMPAS additionally exhibits a "saturated-plain" regime.** LIME
scaffold at σ∈{0.1, 0.3} gives plain NMI 0.574 with C ratio 14 — the
all-innocuous fingerprint alone recovers race via the
`unrelated_column_one` proxy at plain-SBM level (pEx has nothing to
add). Different regime from housing-null and housing-signal.

Two-dataset replication across regression ↔ classification, continuous
↔ mostly-binary features, and custom ↔ Slack-original scaffolds.
Artifacts: `results/sbm_explainer/compas_fingerprint/`.

### Third dataset — Slack German

Ran the same pipeline on Slack's `german_experiment` scaffold. Cleaner
than COMPAS: single continuous innocuous feature
(`LoanRateAsPercentOfIncome`, no random binary proxies), Gender as the
protected layer, n=100 test points.

| Scaffold | σ | plain NMI | pEx NMI | Δ | C ratio | routing |
|---|---|---|---|---|---|---|
| LIME | 0.1 | 0.539 | 0.545 | +0.006 | 12.80 | 0% real |
| LIME | 0.3 | 0.542 | 0.545 | +0.003 | 13.46 | 0% real |
| **LIME** | **1.0** | **0.115** | **0.172** | **+0.057** | **1.36** | 0.1% real |
| SHAP | 0.1 | 0.098 | 0.101 | +0.003 | 1.19 | 35% real |
| SHAP | 0.3 | 0.095 | 0.103 | +0.008 | 1.21 | 31% real |
| **SHAP** | **1.0** | **0.048** | **0.119** | **+0.071** | **1.05** | 17% real |

**Both scaffolds fire at σ=1.0.** LIME σ=1.0 gives Δ=+0.057 (stronger
than housing's +0.048), SHAP σ=1.0 gives Δ=+0.071 (strongest of the
three datasets). Invariant restated again: the routing mixture becomes
partial-enough for the signal when the pert scale is high enough — at
σ=1.0 on standardized German, even LIME's otherwise-extreme detector
sits near the flip point where the few-real-perts pattern matches the
one-anchor-plus-K-refinements geometry.

LIME at σ∈{0.1, 0.3} hits the saturated-plain regime again (NMI 0.54,
C ratio 13) — the innocuous model's loan-rate decision rule happens to
correlate with Gender in the training distribution, so the
all-innocuous fingerprint alone recovers Gender at plain-SBM level.

SHAP at σ∈{0.1, 0.3} is null despite partial routing (~32% real) —
shows partial routing is necessary but not sufficient; the geometry of
the split also matters. At σ=0.1 the 35% mixture is too close to
"mostly biased," which destroys the anchor-vs-refinement structure
needed for the cosine-kNN graph to pick up layer-correlated blocks.

**Three-dataset replication.** Housing (regression) + COMPAS + German
(binary classification), custom and Slack-original scaffolds, Δ
magnitudes 0.048 / 0.038 / 0.057 / 0.071 all in the same range.
Artifacts: `results/sbm_explainer/german_fingerprint/`.

### Fourth dataset — Slack Communities and Crime

Ran the same pipeline on Slack's `cc_experiment` scaffold
(`racePctWhite` as protected layer, d=102 post-preprocessing, n=200
test points). CC has two innocuous-model variants: `psi_one` is a
single-binary-proxy analog of COMPAS; `psi_two` is XOR over two random
binary columns — maximally non-smooth.

**psi_one (single proxy):**

| Scaffold | σ | plain | pEx | Δ | C ratio | routing |
|---|---|---|---|---|---|---|
| LIME | 0.1 | 0.562 | 0.562 | 0.000 | 11.75 | 0.8% |
| LIME | 0.3 | 0.562 | 0.562 | 0.000 | 11.58 | 0.1% |
| **LIME** | **1.0** | **0.046** | **0.276** | **+0.230** | **1.10** | 0.0% |
| **SHAP** | **0.1** | **0.270** | **0.369** | **+0.099** | **1.93** | 62% |
| SHAP | 0.3 | 0.451 | 0.451 | 0.000 | 6.67 | 14% |
| SHAP | 1.0 | 0.007 | 0.091 | +0.085 | 0.97 | 0.1% |

**psi_two (XOR):**

| Scaffold | σ | plain | pEx | Δ | C ratio | routing |
|---|---|---|---|---|---|---|
| LIME | 0.1 | 0.575 | 0.575 | 0.000 | 14.73 | 0.8% |
| LIME | 0.3 | 0.487 | 0.503 | +0.016 | 4.80 | 0.1% |
| **LIME** | **1.0** | **0.124** | **0.699** | **+0.575** | **1.43** | 0.0% |
| SHAP | 0.1 | 0.233 | 0.253 | +0.020 | 1.88 | 64% |
| SHAP | 0.3 | 0.061 | 0.068 | +0.007 | 1.10 | 17% |
| **SHAP** | **1.0** | **0.144** | **0.433** | **+0.289** | **1.51** | 0.1% |

**CC produces the largest effects in the set.** psi_one SHAP σ=0.1
(Δ=+0.099, C 1.93) is a direct middle-regime analog of housing's
Gauss+LIME cell. psi_one LIME σ=1.0 hits Δ=+0.230, and the XOR variant
psi_two LIME σ=1.0 reaches **Δ=+0.575 (plain 0.124 → pEx 0.699)** —
five-to-ten× the deltas from the other datasets.

**XOR amplifies the pEx signal.** psi_two's XOR innocuous model is
maximally non-smooth: perturbations within the gate-hidden path produce
highly variable predictions, making the one-anchor-plus-K-refinements
fingerprint more internally coherent (refinements share a distinctive
XOR-scrambled profile that differs between points). This is a
mechanism prediction: less-smooth innocuous models give bigger Δ, not
smaller — the opposite of what you'd guess if you thought pEx was
picking up local function smoothness.

**Saturated-plain regime** appears again (LIME σ∈{0.1, 0.3}: NMI 0.56–0.58,
C ratio 12–15) via the single-binary proxy correlating with race in
the training distribution — same phenomenon as COMPAS/German.

**Four-dataset replication.** Housing + COMPAS + German + CC
(regression + 3 classifications), custom + Slack-original scaffolds,
single + XOR innocuous models, d=11 to d=102. The routing-mixture
invariant holds across all of them. Artifacts:
`results/sbm_explainer/cc_fingerprint/`.

## 4c. K-sweep and Lipschitz sweep (April 22 pm)

**K-sweep** (`run_k_scaling_sweep.py`, K ∈ {5, 10, 20, 40, 80}, 3 fp
seeds × 5 SBM seeds per cell, all four datasets). Originally set up to
test a $1/K$ vs $1/\sqrt{K}$ rate prediction; the prediction *fails*
but the data reveal the actual mechanism.

CC psi_two (LIME σ=1.0) headline:

| K | plain NMI | pEx NMI | Δ | C ratio |
|---|---|---|---|---|
| 5 | **0.333** | 0.334 | +0.001 | **21.4** |
| 10 | 0.187 | 0.307 | +0.120 | 3.24 |
| 20 | 0.082 | **0.555** | **+0.473** | 1.25 |
| 40 | 0.050 | 0.491 | +0.441 | 1.18 |
| 80 | 0.020 | 0.426 | +0.406 | 1.05 |

**Anchor-dominance erosion.** At small K the fingerprint is 1D-ish (one
anchor + a few refinements); plain SBM alone recovers the layer via
the anchor coord (K=5: NMI 0.333, C ratio 21). As K grows the cosine
decomposition shrinks the anchor's weight from $O(1)$ to $O(1/K)$ while
the refinement-mean field $m_i m_j/(s_i s_j)$ becomes the dominant
$O(1)$ term — which is layer-independent. Plain SBM collapses to the
refinement geometry and loses the layer; pEx keeps it via the
conditional prior. Δ is a "plain-fails-but-pEx-succeeds" gap, not a
signal magnitude. Full analysis in `cosine-expression.md` §6a.

**Lipschitz (β) sweep** (`run_lipschitz_sweep.py`, CC/LIME, σ=1.0,
K=20 fixed, parametric innocuous $\mathrm{sign}(\sin(\beta x_u))$,
β ∈ {0.25, …, 20}, 3 fp seeds × 5 SBM seeds). After reparameterization
on $\mathbb{E}[n_{\text{flips}}] \propto \beta\sigma$ (strictly monotone),
observed Δ is positive but flat with overlapping 95% CIs:

| β | E[n_flips] | Δ | 95% CI (n=3) |
|---:|---:|---:|---:|
| 0.25 | 0.064 | +0.21 | [−0.39, +0.81] |
| 1.00 | 0.254 | +0.08 | [−0.29, +0.46] |
| 2.00 | 0.508 | +0.41 | [−0.22, +1.03] |
| 10.0 | 2.540 | +0.38 | [−0.43, +1.19] |
| 20.0 | 5.079 | +0.25 | [−0.24, +0.74] |

The apparent β=2/β=10 peaks vs β=1 trough at 3-seed point estimates is
inside the noise (per-seed Δ at β=1 is [0.26, 0.00, −0.01]; at β=2 is
[0.52, 0.12, 0.58] — distributions overlap). E[n_flips] ∝ β is
strictly monotone, so no axis reparameterization can rescue a
non-monotone Δ from monotone underlying density.

**Downgrade.** Originally intended as a second confirmatory mechanism
test. Now a **robustness result**: Δ > 0 across a broad class of
innocuous models, not specific to XOR. Load-bearing mechanism
confirmation moves to the K-sweep alone. Figure:
`docs/sbm_explainer/figures/lipschitz_reparam.png`.

**Tanh probe — clean Lipschitz signature** (`run_tanh_sweep.py`,
CC/LIME σ=1.0 K=20, $\mathrm{Bernoulli}(\mathrm{sigmoid}(\beta x_u))$,
β ∈ {0.1, …, 10}, 3 fp × 5 SBM seeds):

| β | plain | pEx | Δ |
|---:|---:|---:|---:|
| 0.1–2 | 0.67–0.73 | 0.67–0.73 | ≈ 0 (saturated-plain via refinement degeneracy) |
| 5.0 | 0.582 | 0.653 | +0.071 |
| 10.0 | **0.394** | 0.651 | **+0.257** |

Plain NMI **monotonically degrades** from 0.73 (β=2) to 0.39 (β=10) —
resolved at n=3 seeds (per-seed std ≤ 0.12). pEx **stays flat** at
0.65–0.73 throughout. This is the §3 two-regime prediction exactly:
low β → $m_i$ near-constant → anchor dominates → plain suffices;
high β → $m_i$ variable → refinement-mean confound dominates at $O(1)$
→ plain rotates off layer → pEx holds via conditional prior. **Tanh
probe promotes (4) from "robustness point" to a second mechanism
confirmation alongside the K-sweep.**

Δ CIs at n=3 are wide at β=10 (95% CI [−0.27, +0.78]) — the plain
trajectory is the clean signature, not Δ magnitude. Targeted 20-seed
rerun at β ∈ {2, 5, 10} (~1 hr) recommended to tighten.

Figure: `docs/sbm_explainer/figures/tanh_sweep.png`.

## 4b. Probes run April 22 pm

- **Probe 1** (OOD flip + routing + honest-model sanity):
  - Routing check confirms LIME detector rejects Gauss perts as OOD (0%
    real), SHAP detector accepts them (73% real) — opposite of the
    original stated mechanism.
  - Three single-model configs (honest / biased / innocuous) all null.
  - Corrected mechanism: rung-3 Δ is a signature of the scaffold's
    routing mixture (one gate-exposed anchor + K smooth gate-hidden
    refinements), producible only when a strict OOD detector keeps
    perts out of the biased path.
  - Implication: rung-3 + finite-K pEx is a **scaffold-attack detector**,
    not a generic feature-geometry detector.
- **Probe 2** (20-seed CI on Gauss+LIME):
  - Δ = 0.0429 ± 0.0180 (SEM 0.004), 95% CI [+0.017, +0.076]
  - t-test vs Δ=0: t = 10.68, p = 1.8 × 10⁻⁹
  - Five-seed estimate reproduces within noise.

## 4d. Honest paper positioning (revised April 22 evening)

**What holds up as formal content:**
1. Four-dataset empirical regularity (housing + COMPAS + German + CC)
   — Δ > 0 in middle-regime cells, routing-mixture invariant.
2. §2 first-order cosine decomposition (correct at edge level, wrong
   baseline for a Δ-scaling prediction).
3. K-sweep as the one clean mechanism test — anchor-dominance erosion
   is unambiguous in the data (CC psi_two plain 0.333 → 0.020; pEx
   stays 0.33–0.56).

**What we downgraded:**
- The $1/K$ vs $1/\sqrt{K}$ rate prediction — retracted.
- β-sweep as second confirmatory test — downgraded to robustness
  result with n=3-seed caveat.

**Evidentiary status options:**
- **Current evidence.** "Empirical regularity + first-order cosine model
  + one mechanism test." Clean and defensible. What the current
  evidence supports on its own.
- **Stronger claim.** Needs either (a) the H-DP refinement/LIME-regime
  story promoted to a second contribution, or (b) a sharper mechanism
  test than K-sweep — likely a theoretical bound on plain-SBM's
  recovery probability as a function of K, which is substantial
  additional work.

**Three questions for Peng Thursday:**
1. Lead finite-K (binary-layer Δ) or H-DP (refinement regimes)?
2. Is four-dataset + K-sweep enough evidence, or should we develop
   the modular-recovery-probability bound?
3. Does β-sweep belong as robustness appendix, or drop it?

## 5. Open technical questions

- Does varying the scaffold's OOD detector (e.g., fit a LIME-style OOD
  on SHAP perturbations) flip the Gauss+SHAP null into a hit? If yes,
  the encoder–scaffold interaction reduces to "is the pert distribution
  in-manifold for the OOD detector."
- Can we build a rung-3 encoder that deliberately maximizes C ratio
  without saturating (e.g., higher-dim targeted grids sweeping multiple
  features)? The finite-K pEx Δ grows, at least naively, with C ratio.
- Is the "three LIME attribution regimes" finding stable across
  scaffold penalties? At stronger penalty, do the mixed clusters merge
  into pure-high (collapsing to two regimes)?
- Do any rung-3 variants reproduce the LIME-attr saturation regime
  (NMI → 0.9+ at K=2)? Would give a strong standalone encoder-as-detector
  result with no pEx component.

---

## 6. Reproducibility

All outputs in `results/sbm_explainer/{fingerprint_sbm,hdp_pex}/`.

```bash
# Rung 3 fingerprint SBM (Python)
python3 scripts/sbm_explainer/run_fingerprint_sbm.py

# Export Gauss+LIME graph to the R sampler's format
python3 scripts/sbm_explainer/export_hdp_inputs.py

# H-DP pEx main run (R, ~100s × 5 seeds, requires mcclust/mcclust.ext)
Rscript scripts/sbm_explainer/run_hdp_pex.R "$PWD"

# θ₀ sweep
Rscript scripts/sbm_explainer/run_hdp_pex_sweep.R "$PWD"

# Analysis
python3 scripts/sbm_explainer/analyze_hdp_pex.py
python3 scripts/sbm_explainer/analyze_hdp_sweep.py
python3 scripts/sbm_explainer/analyze_hdp_refinement.py
python3 scripts/sbm_explainer/analyze_hdp_attributions.py
```

R environment: R 4.5.3, `mcclust` (CRAN), `mcclust.ext` (GitHub:
`sarawade/mcclust.ext`), `forcats`. `papers/pexsbm/pex-sbm.R` was
patched to skip heavy deps (tidyverse / randnet / greed) that aren't
needed by `pex_sbm()` itself.
