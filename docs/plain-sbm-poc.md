# SBM and pEx-SBM: Attack-Detection from Explanation Graphs

_End-to-end documentation of the plain-SBM proof-of-concept, the
partially-exchangeable extension (pEx-SBM), and all stress tests._

Last updated: 2026-04-22 (day before Thursday meeting).

---

## ⚠️ READ FIRST — retraction notice (2026-04-22)

**This document's §§13d–14 conclude "finite-K pEx is structurally
unable to differ from plain SBM." That claim has been RETRACTED.**

The claim was drawn from experiments that all used **rung-1**
attribution-vector graphs (SHAP/LIME attributions as the node
encoding). Those graphs are **regime (b)** — posterior C-ratio
≈ 1.0–1.1, meaning the posterior carries no layer-aligned structure
for the prior to act on. The null result was graph-specific, not
structural.

A third encoder — **rung-3 behavioral fingerprints**, introduced on
2026-04-22 — produces middle-regime graphs (C ratio ≈ 2.5) on the
same housing + scaffold pipeline. On those graphs the finite-K pEx
prior **does** move the posterior: Δ = +0.048 on housing Gauss+LIME
(95% CI [+0.017, +0.076], p = 1.8 × 10⁻⁹, 20 seeds). The effect
replicates across three additional Slack datasets (COMPAS, German,
Communities & Crime) with Δ from +0.038 to +0.575.

**Corrected statement**: finite-K pEx differs from plain only on
graphs whose posterior has unsaturated layer awareness (C-ratio ∈
[1, 3]). Rung-1 attribution graphs don't produce such graphs in
general; rung-3 fingerprint-under-scaffold graphs can.

### Where to read the rung-3 story instead

- **`docs/sbm_explainer/summary_apr22.md`** — Peng-ready writeup of
  the full rung-3 thread.
- **`docs/sbm_explainer/day_log_apr22.md`** — narrative of what broke
  and how.
- **`docs/sbm_explainer/cosine-expression.md`** — the unified §3.1
  mechanism that explains *why* rung 3 works and rung 1 doesn't.
- **`docs/sbm_explainer/encoder_ladder_results.md`** (post-rename;
  formerly `results.md`) — the rung-1 / rung-2 / rung-3 comparison
  table.
- **`Notes/intuition_guide.md`** (vault) — teaching walkthrough of
  the whole SBM thread from first principles, including the
  three-rung encoder ladder.

### What in this document is still valid

- §§1–8 (plain-SBM model + collapsed Gibbs sampler + pEx-SBM model):
  correct, useful as a reference implementation.
- §13b (penalty sweep phase transition at p ≈ 0.19 on sqft_lot LIME):
  still valid, reproducible from `run_penalty_sweep.py`.
- §13c (Fix 3 Beta-Bernoulli layer-density test, log BF rising
  monotonically from −7 to +390 across the penalty sweep): still
  valid.
- §13d (regime (a) vs (b) co-clustering diagnostic): the diagnostic
  itself is correct. Its *conclusion* — that regime (b) is
  universal — is what's retracted. Rung 3 lives in the middle
  regime (neither a nor b) and pEx fires.

> **Naming convention.** "pEx" and "pEx-SBM" in this document refer
> to our **finite-K Dirichlet-Categorical** variant (K fixed,
> per-layer Dirichlet(γ), integrated out against the Beta-Bernoulli
> edge likelihood — §6). This is **not** the pEx-SBM of Durante et al.
> (2025, arXiv:2410.10619), which is the **H-DP pEx** (two-level
> Dirichlet Process, K inferred, three-route Gibbs sampler,
> `pex-sbm.R` in `papers/pexsbm/`). Every null result in §§9–14 is
> specific to our finite-K variant *on rung-1 graphs*. H-DP pEx
> behaves differently, Fix 4 in §15 is now executed (`docs/sbm_
> explainer/summary_apr22.md` §3), and rung-3 graphs are where
> finite-K pEx fires.

---

## Contents

1. Pipeline overview
2. Plain SBM — model
3. Collapsed Gibbs sampler — two conjugacies
4. Per-node update rule
5. Read-outs from a fitted SBM
6. pEx-SBM — model and sampler
7. Main battery results (8 cells × 5 seeds + K-sweep)
8. Prior-vs-likelihood decomposition (theory)
9. Stress test 1 — α-sweep (weak signal)
10. Stress test 2 — γ-sweep (prior strength)
11. Stress test 3 — noisy layer labels
12. Stress test 4 — subpopulation attack
13. Stress test 5 — held-out layer inference
13b. Stress test 6 — penalty sweep (genuine attack strength)
13c. Fix 3 — Beta-Bernoulli layer-density test (pair-level sanity check)
13d. Regime (a) vs (b) — posterior co-clustering diagnostic
14. Consolidated diagnosis
15. What would actually move pEx
16. Performance — Phase 1 refactor (21× speedup)
17. Files and reproducibility

---

## 1. Pipeline overview

Per explanation method (SHAP or LIME) we have an $(n, d)$ matrix $E$, one row
per test instance. We build a behavioral graph by thresholding pairwise
cosine similarity into a binary symmetric $k$-NN adjacency $A$. We then fit a
Bernoulli Stochastic Block Model with $K$ blocks on $A$, recover a block
partition $z$, and score how well $z$ aligns with a **layer label**
$\ell_i = \mathbf{1}[\text{bias\_feature}_i > P_{75}]$ that was *never given
to the model*. The honest and adversarial models are compared on the same
layer.

**Intuition.** If the adversarial scaffold is routing based on the bias
feature, its behavior on layer-high points should look structurally
different from its behavior on layer-low points, so the SBM should pick up a
layer-aligned block. The honest model, which has no such routing, should
not.

---

## 2. Plain SBM — model

Let $A \in \{0,1\}^{n \times n}$ be symmetric, zero-diagonal. Assign each
node a latent block $z_i \in \{1, \dots, K\}$. Generative story:

$$
\begin{aligned}
\pi                              &\sim \mathrm{Dirichlet}(\gamma, \dots, \gamma) \\
z_i \mid \pi                     &\sim \mathrm{Categorical}(\pi) \\
B_{rs}                           &\sim \mathrm{Beta}(\alpha, \beta), \quad \text{iid over } (r,s) \\
A_{ij} \mid z_i = r,\, z_j = s   &\sim \mathrm{Bernoulli}(B_{rs})
\end{aligned}
$$

$K$ is fixed. Edges are conditionally independent given the partition; all
"structure" is pushed into the $K \times K$ block-connectivity matrix $B$.

The collapsed sampler integrates $\pi$ and $B$ out in closed form and
samples only $z$.

---

## 3. Collapsed Gibbs sampler — two conjugacies

### 3a. Beta–Bernoulli: edge-block marginal

For block pair $(r,s)$, let $M_{rs}$ = observed edges, $T_{rs}$ = ordered
pair count. Integrating $B_{rs}$:

$$
p(\text{edges}_{rs}) \;=\; \int_0^1 B^{M_{rs}}(1-B)^{T_{rs}-M_{rs}} \,\mathrm{Beta}(B;\alpha,\beta)\, dB \;=\; \frac{\mathrm{B}(\alpha + M_{rs},\; \beta + T_{rs} - M_{rs})}{\mathrm{B}(\alpha, \beta)}.
$$

In log space, `_log_beta_binomial(k, n, alpha_post, beta_post)` in
`src/sbm/sbm.py`.

### 3b. Dirichlet–Categorical: label prior

Integrating $\pi$ out:

$$
p(z_i = r \mid z_{-i}) \;\propto\; n_r^{(-i)} + \gamma,
$$

the Chinese-Restaurant-like "popularity plus pseudocount."

---

## 4. Per-node update rule

Remove node $i$ from its current block and re-sample:

$$
P(z_i = r \mid z_{-i}, A) \;\propto\; (n_r^{(-i)} + \gamma) \;\cdot\; \prod_{s=1}^{K} \mathrm{BB}\!\left( e_{is};\; t_{is},\; \alpha + M_{rs}^{(-i)},\; \beta + T_{rs}^{(-i)} - M_{rs}^{(-i)} \right),
$$

with

- $e_{is}$ = edges from $i$ to current members of block $s$,
- $t_{is}$ = potential such edges ($= n_s^{(-i)}$),
- $M^{(-i)}, T^{(-i)}$ = block-pair counts with $i$ removed,
- $\mathrm{BB}(k; n, a, b) = \mathrm{B}(a+k, b+n-k) / \mathrm{B}(a, b)$.

Computed in log space (`scipy.special.gammaln`) then softmaxed.

**Warm start.** Uniform-random init hit local optima on toy data ($\mathrm{ARI}{\approx}0.4$).
`SpectralClustering` on $A$ as warm-start gives $\mathrm{ARI}=1.0$ on the
same toys. `z_init="random"` opts out.

---

## 5. Read-outs from a fitted SBM

- **Block assignment** $z$ (final).
- **Posterior-mean block-probability matrix**
  $\psi_{rs} = (M_{rs} + \alpha) / (T_{rs} + \alpha + \beta)$ — heatmap
  reveals assortative vs disassortative structure.
- **Co-clustering matrix** $C_{ij} = P(z_i = z_j)$ from the thinned
  post-burn-in trace.
- **Layer-block contingency** $N[\ell, r] = \#\{i : \ell_i = \ell,\; z_i = r\}$.
- **Layer-separation NMI** $\mathrm{NMI}(z, \ell)$ — primary scalar readout.
- **Within/across layer co-clustering means.** Gap = direct evidence the
  behavioural graph respects the layer.

---

## 6. pEx-SBM — model and sampler

With observed layer labels $\ell_i \in \{1, \dots, L\}$:

$$
\begin{aligned}
\pi^{(\ell)} &\sim \mathrm{Dirichlet}(\gamma, \dots, \gamma), \quad \ell = 1, \dots, L \\
z_i \mid \ell_i = \ell,\, \pi^{(\ell)} &\sim \mathrm{Categorical}(\pi^{(\ell)}) \\
A_{ij} \mid z_i = r,\, z_j = s &\sim \mathrm{Bernoulli}(B_{rs}) \quad \text{(unchanged)}
\end{aligned}
$$

Block identities and $B$ are shared across layers; only the block-proportion
vectors $\pi^{(\ell)}$ differ — hence "partially exchangeable." The Gibbs
update changes in one term only:

$$
p(z_i = r \mid z_{-i}, \ell_i = \ell) \;\propto\; n_{r,\ell}^{(-i)} + \gamma, \qquad n_{r,\ell}^{(-i)} = \#\{j \neq i : z_j = r,\, \ell_j = \ell\}.
$$

The Beta-Binomial edge term is identical to plain SBM.

Implementation: `fit_pex_sbm` in `src/sbm/pex_sbm.py` maintains an additional
$(L, K)$ table `N_layer_block`. Entries with $\ell_i = -1$ are treated as
unobserved and fall back to the plain Dirichlet prior (used by the held-out
stress test).

---

## 7. Main battery results

**Setup.** 8 cells (4 features × 2 explainers), 5 seeds each, fixed
$K = 4$, $k = 8$, $n_{\text{iter}} = 400$, burn-in $= 200$, thinning $= 5$.

### Layer-separation NMI (5-seed mean ± std)

| Bias feature | Moran's $I$ | $n_{\text{high}}$ | Explainer | Honest NMI | Adv NMI (plain) | Adv NMI (pEx) |
|---|---|---|---|---|---|---|
| pct_black_latino | 0.76 | 24 | SHAP | 0.064 ± 0.001 | **0.139 ± 0.001** | 0.167 ± 0.017 |
| pct_black_latino | 0.76 | 24 | LIME | 0.039 ± 0.000 | **0.271 ± 0.000** | 0.271 ± 0.000 |
| sqft_lot | 0.47 | 50 | SHAP | 0.020 ± 0.000 | **0.095 ± 0.001** | 0.099 ± 0.006 |
| sqft_lot | 0.47 | 50 | LIME | 0.012 ± 0.002 | **0.538 ± 0.001** | 0.538 ± 0.001 |
| view | 0.24 | 17 | SHAP | 0.027 ± 0.000 | 0.068 ± 0.001 | 0.068 ± 0.001 |
| view | 0.24 | 17 | LIME | 0.042 ± 0.004 | 0.062 ± 0.001 | 0.062 ± 0.001 |
| waterfront | 0.05 | 4 | SHAP | 0.002 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 |
| waterfront | 0.05 | 4 | LIME | 0.015 ± 0.005 | 0.015 ± 0.002 | 0.015 ± 0.002 |

### K-sweep on sqft_lot LIME (strongest cell, 5 seeds)

| $K$ | plain honest NMI | plain adv NMI | pEx honest NMI | pEx adv NMI |
|---|---|---|---|---|
| **2** | 0.000 | **0.916** | 0.000 | **0.916** |
| 3 | 0.013 | 0.654 | 0.013 | 0.654 |
| 4 | 0.011 | 0.538 | 0.011 | 0.538 |
| 6 | 0.032 | 0.459 | 0.034 | 0.459 |

At $K = 2$ the behavioural block structure recovers the binary layer label
almost perfectly ($\mathrm{NMI} = 0.916$), zero variance across seeds, from
edges alone — the SBM is never told which points are high-sqft_lot.

### Honest ↔ Adversarial partition agreement (same nodes, two models)

| Setup | NMI | ARI |
|---|---|---|
| pct_black_latino SHAP | 0.507 | 0.479 |
| pct_black_latino LIME | 0.607 | 0.535 |
| sqft_lot SHAP | 0.352 | 0.249 |
| sqft_lot LIME | 0.557 | 0.466 |

### Key observations

1. **Layer-separation NMI rises after the attack in every cell.** The
   adversarial explanation graph is more layer-aligned than the honest one,
   even though neither model was told the layer. Core positive signal.

2. **LIME 3–5× more sensitive than SHAP.** Plausible mechanism: LIME fits a
   *local linear* per instance; the scaffold's discrete biased↔innocuous
   switch shows up as a sharp local slope. SHAP tree explanations are
   flattened by background-sample averaging.

3. **Across-layer co-clustering collapses in the strongest cell.** sqft_lot
   LIME adv: within-layer $C = 0.400$, across-layer $C = 0.013$ (~30× ratio).
   Honest baselines show ratios of 1.0–1.2×.

4. **Moran's $I$ is not the right moderator.** Ranking of adv NMI over
   features: sqft_lot ($I{=}0.47$) $>$ pct_black_latino ($I{=}0.76$) $>$ view
   ($I{=}0.24$) $\gg$ waterfront ($I{=}0.05$). Real moderators: (a) whether
   $n_{\text{high}}$ clears the block-size floor, (b) whether the feature is
   in the baseline model at all (structural vs demographic regime).

5. **pEx matches plain in all 8 cells.** Within $\pm 0.005$, 5-seed means.
   Motivating the decomposition below.

---

## 8. Prior-vs-likelihood decomposition

pEx and plain share the Beta–Bernoulli edge likelihood — same $A$, same $K$,
same $(\alpha, \beta)$. Only the block prior differs:

$$
\underbrace{p(z_i = r \mid z_{-i})}_{\text{plain}} \propto n_r^{(-i)} + \gamma,
\qquad
\underbrace{p(z_i = r \mid z_{-i}, \ell_i)}_{\text{pEx}} \propto n_{r, \ell_i}^{(-i)} + \gamma.
$$

Any observable gap is therefore a gap *between priors*. Log-posterior odds
between candidate partitions $z^\star$ (layer-aligned) and $z'$:

$$
\log \frac{p(z^\star \mid A)}{p(z' \mid A)} = \underbrace{\log \frac{p(A \mid z^\star)}{p(A \mid z')}}_{\log \Lambda \text{ (shared)}} + \underbrace{\log \frac{p(z^\star)}{p(z')}}_{\log \Pi \text{ (differs)}}
$$

### Three regimes

| Regime | $\log \Lambda$ | Behaviour |
|---|---|---|
| **Saturated** | $\gg \log \Pi$ | Posterior $\approx$ MLE of likelihood; prior negligible. Both methods recover. |
| **Absent** | $\approx 0$ | Posterior is prior, but neither prior puts preference between equally-populated competing partitions — both collapse to noise. |
| **Intermediate** | $\sim \log \Pi$ | Prior breaks the tie. pEx should beat plain *if* the mechanism is active. |

### Where the cells sit

| Cell | adv NMI | Regime |
|---|---|---|
| sqft_lot LIME (K=2) | 0.916 | saturated |
| sqft_lot LIME (K=4) | 0.538 | saturated (partial recovery, edges still dominate) |
| pct_BL LIME (K=4) | 0.271 | saturated |
| view LIME (K=4) | 0.062 | absent ($n_{\text{high}}{=}17$) |
| waterfront LIME (K=4) | 0.015 | absent ($n_{\text{high}}{=}4$) |

Every cell is saturated or absent — so the battery null is predicted by the
decomposition alone. Stress tests 1–5 probe the intermediate regime and
adjacent pathologies.

### General principle

> A Bayesian method whose only contribution is a prior modification can
> produce an observable effect only when the likelihood is neither saturated
> nor absent.

Applies to informative priors, regularization-as-prior, hierarchical
pooling. Effect size is $O(\log \Pi / \log \Lambda)$; both extremes hide it.

---

## 9. Stress test 1 — α-sweep (weak signal)

**Script.** `scripts/run_weak_signal_stress.py`. Results:
`results/sbm_poc/stress_weak_signal/summary.json`.

**Design.** Interpolate attack strength:

$$
X(\alpha) \;=\; \alpha\, X_{\text{adv}} + (1 - \alpha)\, X_{\text{honest}}, \quad \alpha \in \{1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.05\}.
$$

Track a log-likelihood-ratio proxy $\widehat{\log\Lambda}$ derived from the
edge-density gap between within-layer and across-layer pairs under a
two-block Beta-Binomial:

$$
\widehat{\log\Lambda} \;\approx\; \tfrac{1}{2}\, n_{\text{edges}} \log\!\frac{p_w (1 - p_a)}{p_a (1 - p_w)}.
$$

| α | $\widehat{\log\Lambda}$ | plain NMI | pEx NMI | Δ |
|---|---|---|---|---|
| 1.00 | 3864 | 0.538 ± 0.001 | 0.538 ± 0.001 | 0.000 |
| 0.70 | 3366 | 0.398 ± 0.000 | 0.398 ± 0.000 | 0.000 |
| 0.50 | 1667 | 0.014 ± 0.001 | 0.014 ± 0.001 | 0.000 |
| 0.35 | 783 | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.000 |
| 0.25 | 325 | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.000 |
| 0.15 | 71 | 0.038 ± 0.001 | 0.038 ± 0.001 | 0.000 |
| 0.05 | −10 | 0.012 ± 0.002 | 0.012 ± 0.002 | 0.000 |

**Shows.** pEx = plain to machine precision at every $\alpha$, including the
intermediate regime where plain collapses $0.538 \to 0.014 \to 0.038$.
Likelihood ranges four orders of magnitude; pEx never pulls ahead. The
intermediate regime did not rescue pEx.

---

## 10. Stress test 2 — γ-sweep (prior strength)

**Script.** `scripts/run_gamma_sweep.py`. Results:
`results/sbm_poc/stress_gamma/summary.json`.

**Design.** Hold likelihood fixed (full attack), vary Dirichlet concentration:

$$
\gamma \in \{0.1, 1, 5, 20, 100\}, \quad 3 \text{ seeds each}.
$$

| γ | plain NMI | pEx NMI | Δ |
|---|---|---|---|
| 0.1 | 0.53805 ± 0.00049 | 0.53805 ± 0.00049 | 0.0 |
| 1.0 | 0.53805 ± 0.00049 | 0.53805 ± 0.00049 | 0.0 |
| 5.0 | 0.53805 ± 0.00049 | 0.53805 ± 0.00049 | 0.0 |
| 20.0 | 0.53805 ± 0.00049 | 0.53805 ± 0.00049 | 0.0 |
| 100.0 | 0.53805 ± 0.00049 | 0.53805 ± 0.00049 | 0.0 |

**Shows.** Partitions are bit-for-bit identical between plain and pEx at
every $\gamma$, and invariant to $\gamma$ itself — mean NMI matches to 16
digits. Scaling the Dirichlet prior 1000× does nothing.

### Magnitude analysis

Per-node log-prior difference between pEx and plain is bounded by

$$
|\log(n_{r,\ell} + \gamma) - \log(n_r + \gamma)| \;\leq\; \log\!\frac{n + \gamma}{\gamma} \;\approx\; \log(n_\ell^{\max} + 1) \;\sim\; 4\ \text{nats per node}
$$

at $n_{\text{high}} = 50$, $\gamma = 1$. The per-node Beta–Binomial edge term
sums $K$ contributions each of order $O(e_{is})$; at ~16 edges per node in
the 8-NN graph with $p_w \approx 0.08$ vs $p_a \approx 0.002$, log-odds per
edge $\approx \log(40) \approx 3.7$, giving **~30+ nats per candidate gap.**

At $\gamma = 100$, $K = 4$ the prior gap is $\log((n+\gamma)/\gamma)
\approx 1.1$ nats — *smaller*, not larger. The Dirichlet prior cannot reach
the edge term's scale under this graph regime.

### Diagnosis (strong)

The softmax of edge-dominated log-probs collapses to an argmax. Plain and
pEx share the edge term and therefore make the **same move at every node,
every step**, regardless of the prior.

> In the strong-attack regime, the collapsed Gibbs sampler is not sampling.
> It is argmax-ing the edge likelihood. Any method whose contribution is
> only through the Dirichlet–Categorical prior is structurally invisible —
> not just dominated.

---

## 11. Stress test 3 — noisy layer labels

**Script.** `scripts/run_noisy_layer_stress.py`. Results:
`results/sbm_poc/stress_noisy_layer/summary.json`.

**Design.** Flip a fraction $p$ of layer labels (sqft_lot LIME, full attack,
$K{=}4$, 5 seeds). Measure pEx NMI vs the **true** layer.

| $p$ | pEx NMI | Δ vs plain |
|---|---|---|
| 0.00 | 0.538 ± 0.0005 | 0.0 |
| 0.05 | 0.538 ± 0.0005 | 0.0 |
| 0.10 | 0.538 ± 0.0004 | −0.0002 |
| 0.20 | 0.538 ± 0.0005 | 0.0 |
| 0.30 | 0.538 ± 0.0005 | 0.0 |
| 0.50 | 0.538 ± 0.0004 | −0.0002 |

**Shows.** pEx is flat at 0.538 for every $p$, even $p = 0.5$ (pure
random labels). Follows immediately from the γ-sweep diagnosis: since the
prior doesn't enter the decision, corrupting the labels the prior sees has
no effect. "pEx is robust to label noise" only because pEx is robust to its
own prior.

---

## 12. Stress test 4 — subpopulation attack

**Script.** `scripts/run_subpop_stress.py`. Results:
`results/sbm_poc/stress_subpop/summary.json`.

**Design.** Attacker biases only fraction $p$ of layer-high nodes (rest get
honest explanations). Layer-low always honest. Layer labels true for all
high nodes, including those whose explanation was left honest.
$K{=}4$, 5 seeds.

| $p$ | plain NMI | pEx NMI | Δ |
|---|---|---|---|
| 0.00 | 0.012 ± 0.002 | 0.013 ± 0.002 | +0.001 |
| 0.25 | 0.034 ± 0.008 | 0.036 ± 0.011 | +0.002 |
| 0.50 | 0.075 ± 0.008 | 0.079 ± 0.012 | +0.004 |
| 0.75 | 0.215 ± 0.057 | 0.222 ± 0.069 | +0.007 |
| 1.00 | 0.441 ± 0.000 | 0.441 ± 0.000 | +0.000 |

**Shows.** NMI rises smoothly with attack coverage (no threshold effect —
attack-detection is dose-responsive). pEx gives a positive bias of
0.001–0.007 at intermediate $p$, but always **within seed std** — not
significant. The per-layer prior does not rescue unattacked layer-high
nodes from their "honest-looking" edge neighbourhoods.

Note that even at $p = 0.75$, seed std is 0.06 (huge) — the sampler is now
in a genuinely multi-modal regime where different seeds find different
layer-aligned partitions.

---

## 13. Stress test 5 — held-out layer inference

**Script.** `scripts/run_heldout_stress.py`. Results:
`results/sbm_poc/stress_heldout/summary.json`.

**Design.** Fraction $f$ of layer labels hidden at fit time (layer $= -1$;
pEx falls back to plain Dirichlet for those nodes). Stratified per layer.
NMI evaluated on the held-out subset only — the *generalisation* task. Plain
SBM never uses labels and serves as the "no-info" baseline. $K{=}4$, 5 seeds.

| $f$ | plain held NMI | pEx held NMI | Δ |
|---|---|---|---|
| 0.10 | 0.485 | 0.485 | 0.000 |
| 0.25 | 0.510 | 0.510 | 0.000 |
| 0.50 | 0.541 | 0.541 | 0.000 |
| 0.75 | 0.555 | 0.555 | 0.000 |

**Shows.** This was the one experiment pEx could have won by construction
— plain-SBM has no layer channel at all. It still ties plain to seed
precision on every held-out fraction. With 25–90% of labels observed, pEx
places held-out nodes into the same blocks plain does without any label
info. When the prior is the *only* differentiator it still doesn't move
the sampler.

---

## 13b. Stress test 6 — penalty sweep (genuine attack strength)

**Script.** `scripts/run_penalty_sweep.py`. Results:
`results/sbm_poc/stress_penalty/{summary.json, penalty_plot.png}`.

**Motivation.** The α-sweep interpolated post-hoc between honest and
adversarial *explanation vectors* — a synthetic mixture, not a genuinely
weaker attacker. Here we regenerate adversarial explanations from scratch
at each scaffold penalty $p$ (retraining the biased model), producing the
true "natural weak-attack" trajectory.

**Design.** For each $p$, regenerate `generate_scaffold_explanations.py`
→ build 8-NN similarity graph for LIME and SHAP → fit plain + pEx at
$K{=}4$, 5 seeds. An initial grid $\{0.05, 0.20, 0.35, \ldots, 0.95\}$
showed a discontinuous jump from plain adv NMI = 0.038 at $p{=}0.05$ to
0.398 at $p{=}0.20$, so a transition-window grid
$\{0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20\}$ was run to
probe the middle.

**LIME (sqft_lot, $K{=}4$):**

| $p$ | plain adv | pEx adv | Δ |
|---|---|---|---|
| 0.01 | 0.012 | 0.012 | +0.0006 |
| 0.04 | 0.036 | 0.037 | +0.0010 |
| 0.05 | 0.038 | 0.039 | +0.0012 |
| 0.07 | 0.014 | 0.014 | +0.0001 |
| 0.10 | 0.035 | 0.035 | +0.0000 |
| 0.13 | 0.010 | 0.010 | +0.0007 |
| 0.16 | 0.017 | 0.017 | +0.0000 |
| 0.19 | 0.398 | 0.398 | +0.0000 |
| 0.20 | 0.398 | 0.398 | +0.0000 |

**SHAP (sqft_lot, $K{=}4$):** plain adv NMI flat at 0.042–0.055 across
all $p$; max |Δ| = 0.0066 at $p{=}0.20$, all others within ±0.0021. SHAP
cliff lies above 0.20.

**Shows.**

1. **Phase transition, not a gradient.** LIME detectability is a step
   function: noise floor for $p \leq 0.16$, saturated plateau for $p \geq
   0.19$. No intermediate NMI values appear at grid resolution $\Delta p =
   0.03$. The "testable middle" where the prior could matter is a width-<
   0.03 knife-edge, not a regime.
2. **pEx null survives the transition.** Even the cliff point ($p{=}0.19$,
   plain NMI = 0.398) gives Δ = 0.000. No regime — subnoise, cliff, or
   saturated — produces a pEx advantage.
3. **Confirms the α-sweep diagnosis under a genuine attack.** The
   synthetic-mixture null wasn't an artefact of the interpolation; it
   holds when the attacker is genuinely weaker.

---

## 13c. Fix 3 — Beta-Bernoulli layer-density test (pair-level sanity check)

**Script.** `scripts/run_layer_density_test.py`. Results:
`results/sbm_poc/layer_density/{summary.json, comparison.md}`.

**Purpose.** A prerequisite check independent of any block decomposition:
does the 8-NN similarity graph have any layer-aligned edge structure at
all? If yes, SBMs have a signal lower bound; if no, the pEx null isn't
the SBM's fault.

**Model.** Three Beta-Bernoulli cells on pair type:
$$A_{ij} \mid \ell_i, \ell_j \sim \text{Bernoulli}(\theta_{\ell_i \ell_j}), \qquad \theta \sim \text{Beta}(1,1).$$
Closed-form posteriors $\theta_{\ell \ell'} \mid A \sim \text{Beta}(1 + M_{\ell \ell'},\; 1 + T_{\ell \ell'} - M_{\ell \ell'})$
where $M$ and $T$ are edge and trial counts per pair type. Log Bayes
factor compares $\mathcal{M}_1$ (three $\theta$'s) vs $\mathcal{M}_0$
(one shared $\theta$):
$$\log\text{BF} = \sum_{k \in \{00,11,01\}} \text{Beta}(1 + M_k, 1 + T_k - M_k) - \text{Beta}(1 + M_\text{tot}, 1 + T_\text{tot} - M_\text{tot}).$$

**Relation to the SBM.** The SBM (plain or pEx) fits a partition $z$ and
measures layer-alignment of $z$ via NMI. Fix 3 bypasses the partition
entirely: it measures layer-alignment of the *edges* directly. Same
graph, same layer. If Fix 3 returns a large log BF but the SBM returns a
low NMI, the SBM's partition readout is discarding signal that the graph
contains.

**Main battery (adversarial graphs, 8 cells).**

| Feature | Expl. | log BF | $\theta_{11}$ | $\theta_{01}$ | ratio | plain NMI | pEx NMI |
|---|---|---|---|---|---|---|---|
| pct_black_latino | SHAP | **+207.8** | 0.324 | 0.011 | 28.5 | 0.139 | 0.167 |
| pct_black_latino | LIME | **+108.5** | 0.266 | 0.024 | 11.3 | 0.271 | 0.271 |
| sqft_lot | SHAP | **+356.3** | 0.178 | 0.010 | 18.3 | 0.095 | 0.099 |
| sqft_lot | LIME | **+516.0** | 0.199 | 0.003 | 78.5 | 0.538 | 0.538 |
| view | SHAP | +154.3 | 0.449 | 0.014 | 32.5 | 0.068 | 0.068 |
| view | LIME | +100.7 | 0.377 | 0.021 | 17.8 | 0.062 | 0.062 |
| waterfront | SHAP | +0.33 | 0.375 | 0.037 | 10.2 | 0.014 | 0.014 |
| waterfront | LIME | −3.69 | 0.250 | 0.045 | 5.6 | 0.015 | 0.015 |

Honest graphs all have $|\log\text{BF}| \leq 5$ — confirms the layer
signal is attack-induced, not an artefact of the layer definition.
Waterfront has no evidence either way ($n_\text{high}=4$, only 6 possible
high–high pairs) — the null there is structural, not a failure.

**Penalty sweep (sqft_lot, adversarial).**

| $p$ | Expl. | log BF | $\theta_{11}$ | $\theta_{01}$ | ratio | plain NMI |
|---|---|---|---|---|---|---|
| 0.01 | LIME | −6.8 | 0.068 | 0.053 | 1.3 | 0.012 |
| 0.04 | LIME | −6.5 | 0.068 | 0.053 | 1.3 | 0.036 |
| 0.05 | LIME | −5.4 | 0.071 | 0.052 | 1.4 | 0.038 |
| 0.07 | LIME | +3.9 | 0.079 | 0.046 | 1.7 | 0.014 |
| 0.10 | LIME | **+50.2** | 0.105 | 0.035 | 3.0 | 0.035 |
| 0.13 | LIME | **+130.2** | 0.132 | 0.025 | 5.2 | 0.010 |
| 0.16 | LIME | **+259.2** | 0.161 | 0.015 | 10.7 | 0.017 |
| 0.19 | LIME | **+390.0** | 0.183 | 0.008 | 24.1 | 0.398 |
| 0.20 | LIME | **+433.0** | 0.188 | 0.006 | 33.6 | 0.398 |

SHAP penalty-sweep log BFs rise in parallel (−4.4 at $p{=}0.01$ to
+172.6 at $p{=}0.20$) while plain SHAP NMI stays flat ~0.045.

**Shows.**

1. **Log BF is smooth and monotone; SBM NMI is a phase transition.** On
   LIME, log BF climbs from $-7$ at $p{=}0.01$ to $+390$ at $p{=}0.19$
   through every intermediate value: $+4, +50, +130, +259$. At each of
   these the edges encode layer at $\log\text{BF} \gg 5$ (strong
   evidence) but the plain-SBM partition returns noise NMI. The signal
   is there continuously; the SBM *throws it away* at the readout.
2. **This reframes the pEx null.** Across all stress tests the question
   was "why doesn't pEx beat plain?" The honest answer is now:
   **neither plain nor pEx exploit the available evidence**. A pair-level
   detector sees +130 nats at $p{=}0.13$ where the SBM sees nothing. Any
   prior-level pEx modification inherits this limitation because the
   bottleneck isn't the prior, it's the partition readout.
3. **The phase transition sits in the clustering step, not the graph.**
   At $p{=}0.16$ the high–high edge density is already $10\times$ the
   across density (ratio 10.7, log BF +259). The SBM still returns NMI
   0.017 because block-reassignment is dominated by within-block edge
   mass, not cross-block contrast. Once the gap widens enough for
   spectral/argmax dynamics to flip the partition, NMI jumps from 0.02
   to 0.40 in one increment.
4. **SHAP is in a permanent "subdetect" regime.** Log BF goes from −4 to
   +172 across the penalty sweep, but plain SHAP NMI never climbs above
   0.055. SHAP has layer signal but the SBM doesn't cross threshold
   within the tested range. Confirms the diagnosis isn't LIME-specific.
5. **Honest baselines clean.** All 8 honest cells have $|\log\text{BF}| <
   31$ (view's 30 reflects tiny $T_{\text{hh}}=136$; the ratio-5.6
   effect disappears relative to adv's 32.5). No spurious layer signal
   before attack.

**Initial interpretation — partially retracted by §13d.** The earlier
draft claimed "the prior isn't the bottleneck, the readout is." That
framing conflates two failure modes:

- **(a) MAP collapse.** Posterior $P(z \mid A)$ has mass on layer-aligned
  configurations but the MAP readout picks a different one. In that
  case the full co-clustering matrix $C_{ij} = P(z_i = z_j \mid A)$
  carries signal the MAP discards; a posterior-level auditor recovers
  it.
- **(b) Likelihood blindness.** The posterior itself doesn't encode the
  layer — no discrete K-block partition concentrates the pair-level
  evidence. No readout of the SBM posterior can recover the signal
  because the posterior doesn't carry it.

§13d below runs the diagnostic and shows the transition region is
regime (b). The correct takeaway is narrower:

- **Pair-level layer-density test is itself a usable auditor.** It
  separates saturated from absent cells at log BF ~100s vs ~0 and
  tracks attack strength continuously in closed form. Still stands.
- **A block-level auditor via posterior-level readout does not rescue
  plain SBM** in the transition region — see §13d.

---

## 13d. Regime (a) vs (b) — posterior co-clustering diagnostic

**Script.** `scripts/run_coclustering_diagnostic.py`. Results:
`results/sbm_poc/coclustering_diagnostic/{summary.json, table.md}`.

**Setup.** Plain SBM, $K{=}4$, 5 seeds, posterior co-clustering matrix
$C_{ij} = \frac{1}{S} \sum_s \mathbb{1}[z_i^{(s)} = z_j^{(s)}]$ averaged
across seeds and post-burn-in draws. We compare:

- $\bar C_\text{within} = $ mean $C$ over pairs with same layer
- $\bar C_\text{across} = $ mean $C$ over pairs with different layers

In regime (a) the gap $\bar C_\text{within} - \bar C_\text{across}$
precedes the MAP NMI transition: posterior has layer structure before
the modal partition does. In regime (b) the gap moves in lockstep with
MAP NMI — the posterior itself is blind to the layer.

**Penalty sweep on adversarial sqft_lot.**

| $p$ | Expl. | MAP NMI | $\bar C_\text{within}$ | $\bar C_\text{across}$ | ratio | log BF |
|---|---|---|---|---|---|---|
| 0.01 | LIME | 0.012 | 0.250 | 0.245 | 1.02 | −6.8 |
| 0.04 | LIME | 0.036 | 0.310 | 0.298 | 1.04 | −6.5 |
| 0.05 | LIME | 0.038 | 0.310 | 0.297 | 1.04 | −5.4 |
| 0.07 | LIME | 0.014 | 0.251 | 0.244 | 1.03 | +3.8 |
| 0.10 | LIME | 0.035 | 0.270 | 0.288 | **0.94** | +50.2 |
| 0.13 | LIME | 0.010 | 0.250 | 0.246 | 1.02 | **+130.2** |
| 0.16 | LIME | 0.017 | 0.254 | 0.252 | 1.01 | **+259.2** |
| 0.19 | LIME | 0.398 | **0.365** | **0.076** | **4.78** | +390.0 |
| 0.20 | LIME | 0.398 | **0.365** | **0.076** | **4.78** | +433.0 |
| 0.10 | SHAP | 0.048 | 0.291 | 0.281 | 1.04 | +18.1 |
| 0.13 | SHAP | 0.047 | 0.289 | 0.283 | 1.02 | +48.8 |
| 0.16 | SHAP | 0.046 | 0.290 | 0.287 | 1.01 | +86.8 |
| 0.19 | SHAP | 0.055 | 0.279 | 0.255 | 1.09 | +158.1 |
| 0.20 | SHAP | 0.055 | 0.277 | 0.253 | 1.10 | +172.5 |

**Shows.**

1. **Regime (b).** On LIME the posterior ratio is flat at 1.01–1.04
   throughout $p \in [0.01, 0.16]$ — even at $p{=}0.16$ where log BF =
   +259 (the edges carry ~$e^{259}$-fold evidence for layer-dependent
   density). Then at $p{=}0.19$ the ratio **jumps** to 4.78
   simultaneously with MAP NMI rising from 0.017 to 0.398. The
   posterior mirrors the MAP phase transition; it does not precede it.
2. **The gap is essentially zero — or negative — in the transition
   region.** At $p{=}0.10$ LIME the within-layer co-clustering is
   *lower* than across-layer (ratio 0.94, gap −0.019). The SBM isn't
   just missing the layer; it's finding block structure mildly
   anti-correlated with it, which is exactly what "distributed
   pair-level evidence that fails to concentrate into K blocks" looks
   like.
3. **SHAP stays in regime (b) across the entire grid.** Log BF reaches
   +172 at $p{=}0.20$ but $\bar C$ ratio never exceeds 1.10. SHAP adv
   graphs are strictly weaker at driving block formation than LIME,
   despite comparable pair-level evidence.
4. **MAP NMI and posterior $\bar C$ ratio are redundant readouts of the
   same underlying quantity: whether the posterior has identified a
   layer-coherent partition.** Neither is a bottleneck relative to the
   other; they both report what the posterior carries.

**Interpretation.** The SBM is a poor summarizer of this graph in the
transition region, not a well-informed posterior being poorly read out.
Pair-level evidence distributed across many nodes does not concentrate
into K discrete blocks under the Beta-Bernoulli + Dirichlet likelihood.
The partition representation itself is lossy here — even a perfect
oracle readout can't extract signal the posterior never placed.

**Consequence for the roadmap.** Readout-level fixes (per-block
$\hat\theta_{\ell \ell'}$, marginal-likelihood ratios) on the existing
plain-SBM posterior will *not* rescue the transition-region null. The
interventions that could work all change the model, not the summary:

- **Layer-conditional edge likelihood** $B_{rs}^{(\ell)}$ — injects
  layer into the likelihood, so the posterior is *allowed* to encode
  it.
- **Weighted-edge SBM on continuous similarities** — binarizing 8-NN
  may discard magnitude information that carries the transition-region
  signal. Fix 3 showed the binary graph still has evidence, but the
  distribution across pairs is what the Beta-Bernoulli can't
  concentrate.
- **Degree correction (DC-SBM).** If high-sqft_lot nodes have
  systematically different similarity-graph degrees, plain SBM may be
  blocking by degree rather than by the layer-aligned connectivity
  pattern.

Fix 4 (H-DP) remains worth attempting but is no longer obviously
sufficient: H-DP still assumes a fixed Beta-Bernoulli likelihood, so
the distributed-evidence problem may persist. Layer-conditional
$B_{rs}^{(\ell)}$ is the more directly targeted intervention.

---

## 14. Consolidated diagnosis

Six SBM stress tests across five orthogonal axes — attack strength
interpolation ($\alpha$), prior strength ($\gamma$), label quality ($p$),
attack coverage (subpop), label visibility ($f$), and genuine attack
strength (penalty) — all return $\Delta(\text{pEx} - \text{plain}) = 0$
within seed precision.

The null is **not** "pEx is as good as plain."

The null (for our finite-K Dirichlet-Categorical pEx) is: on these six
tests, **finite-K pEx cannot differ from plain** because the collapsed
Gibbs softmax is argmax-ed by edge evidence at every node, making the
~4-nat Dirichlet term structurally invisible, not merely dominated.

**Update (2026-04-22, rung-3 result).** This is now known to be graph-
specific, not universal. On the Gauss+LIME fingerprint graph (rung 3 of
the encoder ladder — `docs/sbm_explainer/`), at log BF +145 and MAP NMI
0.287 with posterior C ratio 2.51, finite-K pEx gives
$\Delta = +0.048 \pm 0.008$ — the first non-null pEx effect in the
campaign. The prior is invisible *on graphs where the posterior is
itself layer-blind or layer-saturated*; given a middle-regime graph, it
acts. SHAP/LIME attribution graphs produce only blind-or-saturated
posteriors; fingerprint-under-scaffold graphs produce middle-regime
ones. See `docs/sbm_explainer/encoder_ladder_results.md` for the full rung comparison.

H-DP pEx (Durante et al. 2025 — §15 Fix 4) is a different model, not a
tuning of ours, and its per-node update structure (three sampling
routes, K inferred) may give a larger effect than our +0.048 on the
same graph. This is still an open question.

This is a methodological finding, not a failed implementation — and it
points cleanly at what *would* work.

**Orthogonal lower bound (Fix 3, §13c).** The 8-NN graph contains
hundreds of nats of layer-aligned edge evidence that grows
monotonically with attack strength while the SBM's MAP NMI stays at
noise until a phase transition at $p \in (0.16, 0.19)$.

**Regime diagnostic (§13d).** The posterior co-clustering ratio
$\bar C_\text{within} / \bar C_\text{across}$ tracks MAP NMI exactly —
flat at ~1.0 through the transition region, jumping to 4.78 at
$p{=}0.19$ simultaneously with NMI. This is regime (b): the posterior
itself is blind to the layer in the transition region, not just the
MAP readout. Readout-level fixes on the existing plain-SBM fits will
not rescue the null. The interventions that could — layer-conditional
edge likelihood $B_{rs}^{(\ell)}$, weighted-edge / DC-SBM — all change
the model, not the summary statistic.

---

## 15. What would actually move pEx

**Note on terminology.** This section proposes fixes for our finite-K
Dirichlet-Categorical pEx. Fix 4 (H-DP pEx, Durante et al. 2025) is a
different model entirely — see below.

Four genuinely different interventions:

1. **Layer-conditional edge likelihood.** Replace Beta–Bernoulli $B_{rs}$
   with $B_{rs}^{(\ell)}$ — one block-connectivity matrix per layer. Now
   the likelihood itself sees the layer and contributes layer-dependent
   per-node nats. Requires revising the Beta-Binomial marginal and the
   $M, T$ count updates to be layer-indexed $(K, K, L)$.

2. **Posterior-regularisation form.** Add a layer-agreement penalty as a
   soft constraint on $z$:

   $$
   \log p(z \mid A, \ell) \;\propto\; \log p(A \mid z) + \log p(z) + \lambda \sum_i \text{agree}(z_i, \ell_i)
   $$

   with $\lambda$ free of the $O(\log n)$ bound that hamstrings the
   Dirichlet term. Effectively a non-Dirichlet prior with tunable strength.

3. **Non-collapsed sampler with explicit $B$ updates.** Under variational
   or HMC posterior, the sampler can explore partitions with lower edge
   likelihood but higher prior support. Mixing worsens; prior-responsiveness
   is restored.

A "crank up $\gamma$" fix does not work — γ-sweep empirically invariant.

4. **H-DP pEx (Durante et al. 2025, Fix 4).** Replace the finite-K
   Dirichlet-Categorical prior with a two-level Dirichlet Process
   hierarchy: a base measure $\tilde P_0 \sim \text{DP}(\theta_0, P_0)$
   over global profiles, per-layer measures $\tilde P_j \sim
   \text{DP}(\theta, \tilde P_0)$. $K$ is inferred, not fixed. Per-node
   Gibbs has three routes (join existing profile in layer / bring in
   global profile not yet in layer / instantiate new profile) rather
   than a single reweighted softmax term. Route B is the important one:
   "does layer $j$ have a profile it has never seen before?" acts at
   the sampling-decision level, not as a log-count nudge bounded by 4
   nats. This is the intervention proposed by the paper the spec doc
   cites; our finite-K implementation is a simpler cousin, not the
   thing tested there. R reference: `papers/pexsbm/pex-sbm.R`
   (Durante et al.'s own sampler). Queued as the highest-priority
   post-Thursday extension given the rung-3 opening that shows the
   prior is not universally invisible.

---

## 16. Performance — Phase 1 refactor

**Before:** ~40 s per fit at $n = 200$, $K = 4$, $n_{\text{iter}} = 400$.
After the 5-stress-test campaign scaled to 5 seeds each, total wall time
was becoming limiting.

**Changes** (`src/sbm/sbm.py` + `src/sbm/pex_sbm.py`):

- Maintain an $(n, K)$ edge-to-block matrix $E$ incrementally:
  $E_{is} = \sum_j A_{ij}\, \mathbf{1}[z_j = s]$.
  Update on move $i: c \to r_{\text{new}}$ as $E_{:,c} \mathrel{-}= A_{:,i}; \; E_{:,r_{\text{new}}} \mathrel{+}= A_{:,i}$ (cost $O(n)$ per move, done once per accepted move rather than per candidate).
- Vectorise $M, T$ updates using symmetric row/column subtraction; the
  within-block diagonal doubling ($M_{cc} \mathrel{-}= 2 e_{ic}$) is handled
  automatically by applying the update to both row and column.
- Compute per-candidate posterior parameters $\alpha_{\text{post}}$,
  $\beta_{\text{post}}$ as full $(K, K)$ matrices with diagonal halving, then
  vectorise `gammaln` across the whole grid in a single call — replacing the
  $K^2$ Python loop with a few array ops.
- Deduplicate pEx's inlined remove/add logic via shared
  `_remove_from_block`, `_add_to_block`, `_log_edges_all_r` helpers in
  `sbm.py`. pEx now only adds the $(L, K)$ table update.

**Measured.** 40 s → ~1.9 s per fit on sqft_lot LIME K=4 ($n = 200$, 400
iter, 3 seeds benchmarked). **≈ 21× speedup**, target was 5–8×. API
unchanged (`SBM`, `PExSBM` dataclasses, `fit_sbm`, `fit_pex_sbm`
signatures identical).

**Correctness.** NMI distribution matches pre-refactor battery values
(5-seed mean 0.53806, known value 0.5380500792166456). Trajectories aren't
bit-identical because vectorised `gammaln` sums in a different order than
the scalar loop, perturbing tie-breaking in `rng.choice` — normal for
any Gibbs sampler.

**Not yet done.** Phase 2 (Numba JIT on the per-node loop, block
membership list) would push another 3–5×. Deferred — current speed is
adequate for the meeting.

---

## 17. Files and reproducibility

### Source

| File | Contents |
|---|---|
| `src/sbm/similarity.py` | `cosine_similarity_matrix`, `build_knn_similarity_graph` |
| `src/sbm/sbm.py` | `fit_sbm` (collapsed Gibbs), `SBM` dataclass, shared helpers `_init_state`, `_log_edges_all_r`, `_remove_from_block`, `_add_to_block` |
| `src/sbm/pex_sbm.py` | `fit_pex_sbm`, `PExSBM` dataclass; reuses shared helpers; supports layer $= -1$ (held-out) |
| `src/sbm/compare.py` | $\psi$ matrix, NMI/ARI, layer-block contingency, within/across co-clustering |
| `src/sbm/sbm_old.py`, `src/sbm/pex_sbm_old.py` | Pre-refactor reference implementations (backup) |

### Scripts

| Script | Produces |
|---|---|
| `scripts/generate_scaffold_explanations.py` | honest + adv SHAP + LIME matrices for any `--bias-feature` |
| `scripts/run_sbm_poc.py` | single-cell end-to-end: explanations → graph → SBM → heatmap + JSON |
| `scripts/run_sbm_battery.py` | 8-cell battery + K-sweep, 5 seeds, plain + pEx |
| `scripts/plot_sbm_heatmap.py` | layer-sorted honest vs adversarial co-clustering heatmap (headline figure) |
| `scripts/run_weak_signal_stress.py` | α-sweep (stress test 1) |
| `scripts/run_gamma_sweep.py` | γ-sweep (stress test 2) |
| `scripts/run_noisy_layer_stress.py` | label-noise sweep (stress test 3) |
| `scripts/run_subpop_stress.py` | subpopulation-coverage sweep (stress test 4) |
| `scripts/run_heldout_stress.py` | held-out layer inference (stress test 5) |
| `scripts/run_penalty_sweep.py` | genuine-attack penalty sweep (stress test 6) |
| `scripts/run_layer_density_test.py` | Fix 3: pair-level Beta-Bernoulli layer-density test |
| `scripts/run_coclustering_diagnostic.py` | Regime (a) vs (b) — posterior co-clustering gap across penalty sweep |

### Results

| Directory | Contents |
|---|---|
| `results/sbm_poc/<feature>_<explainer>/` | per-cell PoC (heatmaps, summary JSON) |
| `results/sbm_poc/battery/` | 8-cell + K-sweep summary + headline heatmap |
| `results/sbm_poc/stress_weak_signal/` | α-sweep |
| `results/sbm_poc/stress_gamma/` | γ-sweep |
| `results/sbm_poc/stress_noisy_layer/` | noisy-label sweep |
| `results/sbm_poc/stress_subpop/` | subpopulation sweep |
| `results/sbm_poc/stress_heldout/` | held-out inference |

### Caveats

- **Single feature for stress tests.** All 5 stress tests use sqft_lot LIME
  K=4 (the strongest cell). The decomposition argument predicts the null
  transfers, but we have not confirmed empirically on pct_black_latino.
- **Binary adjacency.** Thresholded $k$-NN discards edge magnitudes. A
  weighted-SBM variant would keep strengths.
- **Layer $= \mathbf{1}[\text{bias\_feature} > P_{75}]$.** Coarse. For
  sqft_lot this is the group the scaffold penalises (P75 threshold by
  construction); for pct_black_latino it is a proxy.
- **Spectral warm-start biases toward large-block solutions.** Random init
  hits local optima; spectral init finds the edge-dominated mode — which is
  what we want for attack detection but means the sampler may not be
  exploring far from that mode.
