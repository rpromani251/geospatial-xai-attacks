# Thursday Discussion Prep — April 23, 2026

## Headline numbers

| Result | Cell | Value |
|---|---|---|
| **Full layer recovery, no supervision** | sqft_lot LIME, K=2 | **NMI = 0.916** |
| Partial recovery at K=4 | sqft_lot LIME | NMI = 0.538 |
| Cross-feature attack sensitivity | LIME vs SHAP | LIME 3–5× larger NMI |
| Moran's I as moderator | feature sweep | non-monotone; not the right axis |
| pEx vs plain SBM (full-attack) | 8 cells + K-sweep | Δ NMI within ±0.005 |

---

## Three questions for Peng

### 1. Is the K=2 NMI=0.916 result the right headline for a write-up?

**Context:** On sqft_lot LIME, plain SBM with K=2 recovers the layer partition at NMI=0.916 from explanation-similarity edges alone — no supervision, no access to the attack's internals. This is the strongest "unsupervised attack-detection from explanations" signal we have.

**Question:** Is this publishable on its own as an auditing-methods finding, or does it need the full 4-feature + stress-test story to land?

**Why it matters:** Shapes whether we frame the project as (a) a detector methodology paper with the scaffold as case study, or (b) an attacks+detectors paper where detection is one of several threads.

---

### 2. Is the "Moran's I is not the right moderator" finding worth foregrounding?

**Context:** Original hypothesis: higher spatial autocorrelation → attack is easier to hide. Full sweep contradicts this: attack feasibility (OOD accuracy) is I-invariant at ~99%, and residual attribution on the biased feature *grows* monotonically with I. SBM sensitivity also does not track I (sqft_lot I=0.47 dominates pct_black_latino I=0.76 on LIME).

**Question:** Do we reframe the paper around "feature role in the baseline model > spatial autocorrelation" as the real moderator, or treat I as a confounder to control for rather than a variable of interest?

**Why it matters:** If we pivot, the contribution becomes "attack detectability depends on the structural role of the biased feature, not its geography." That's a different (and arguably more transferable) claim.

---

### 3. Does the α-sweep result + γ diagnosis reframe the pEx null correctly?

**Context (the decomposition):** pEx and plain SBM share the Beta–Bernoulli edge likelihood; they differ only in the Dirichlet–Categorical block prior. Any gap between them is a gap between priors, normalized by likelihood strength.

**α-sweep result (new today):** Interpolated $X(\alpha) = \alpha X_{\text{adv}} + (1-\alpha) X_{\text{honest}}$, swept $\alpha \in [0.05, 1.0]$, traced $\widehat{\log\Lambda}$ from 3864 down to −10 (four orders of magnitude). **pEx = plain to machine precision at every α**, including the intermediate regime where plain NMI collapses from 0.54 → 0.014 → 0.038.

**Diagnosis:** At γ=1, K=4, n=200, the per-node log-prior gap between pEx and plain is bounded by log(n_high + 1) ≈ 4 nats. The per-node Beta–Binomial edge term is tens to hundreds of nats. The prior is dwarfed regardless of likelihood strength. **The real axis is γ, not α.**

**γ-sweep result (completed today):** Fixed full attack, varied γ ∈ {0.1, 1, 5, 20, 100}. **pEx ≡ plain bit-for-bit at every γ** — mean NMI = 0.53805 to 16 digits at all 5 γ values. Plain is also invariant to γ (as expected — its single-Dirichlet term doesn't encode the layer). The effect isn't "prior too weak" but "Gibbs softmax collapsed to argmax by edge evidence — prior is structurally invisible."

**Four additional stress tests (completed):**

- **Noisy layer labels** (flip fraction $p \in \{0, 0.05, 0.1, 0.2, 0.3, 0.5\}$): pEx NMI flat at 0.538 across all $p$, max |Δ vs plain| = 2e-4.
- **Subpopulation attack** (bias fraction $p$ of layer-high only, 5 seeds): pEx gives a positive bias of +0.001 to +0.007 at intermediate $p$, but always within seed std.
- **Held-out layer inference** (fraction $f \in \{0.1, 0.25, 0.5, 0.75\}$ labels hidden, pEx sees layer $= -1$ for held-out nodes): pEx NMI on held-out subset ties plain to seed precision at every $f$ — even though plain has no layer channel at all.
- **Penalty sweep** (genuine weaker attacks, not α-interpolation): regenerated scaffolds at $p \in \{0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20\}$ after finding a discontinuous jump 0.038 → 0.398 in LIME NMI between $p{=}0.05$ and $p{=}0.20$. Transition-window grid resolved: **phase transition, not a gradient** — LIME detectability stays at noise (~0.01–0.04) for $p \leq 0.16$, cliffs to 0.398 for $p \geq 0.19$. SHAP cliff lies above 0.20 (flat ~0.045 across grid). pEx Δ null at every point including the cliff itself (max |Δ| = 0.007, all within seed std). There is no "mid-regime" for the prior to matter in.

**Fix 3 orthogonal check (Beta-Bernoulli layer-density test, completed):** Before blaming pEx, verify the graph contains layer-aligned edges at all. Closed-form Beta-Bernoulli posterior on three pair types (low–low, high–high, across) with $\text{Beta}(1,1)$ prior; log Bayes factor $\mathcal{M}_1$ (three $\theta$'s) vs $\mathcal{M}_0$ (one shared $\theta$). Result is decisive: on the sqft_lot LIME adv graph log BF = +516 (ratio $\theta_{11}/\theta_{01} = 78.5$) — the edges encode layer overwhelmingly. Critically, across the penalty sweep log BF rises **smoothly and monotonically** from −7 at $p{=}0.01$ to +390 at $p{=}0.19$, passing through +50, +130, +259 at intermediate penalties where plain-SBM NMI stays at noise (0.010–0.035). The signal is continuous in the graph; the SBM's NMI shows a phase transition.

**Regime (a) vs (b) diagnostic (completed):** Is the SBM's phase transition a MAP-collapse artifact (regime a — posterior has layer structure, argmax throws it away) or does the posterior itself lack layer awareness (regime b — signal never enters the summary)? Computed posterior co-clustering $\bar C_\text{within}$ vs $\bar C_\text{across}$ across the penalty sweep. Result: **regime (b)**. LIME posterior ratio is flat at 1.01–1.04 through $p \in [0.01, 0.16]$ — at $p{=}0.16$ with log BF = +259, the posterior ratio is 1.01, essentially nothing. Then at $p{=}0.19$ the ratio jumps to 4.78 simultaneously with MAP NMI rising from 0.017 to 0.398. At $p{=}0.10$ LIME the gap is *negative* (ratio 0.94). The SBM posterior doesn't carry the layer structure in the transition region; it is a poor summarizer of distributed pair-level evidence, not a well-informed posterior being badly read out. This **retracts the "readout is the bottleneck" framing** and puts model-level fixes (layer-conditional $B_{rs}^{(\ell)}$, weighted-edge / DC-SBM) back in primary position over readout-level fixes.

**Combined conclusion across all six stress tests (α, γ, noisy, subpop, held-out, penalty) plus Fix 3:** pEx is **structurally unable to differ from plain** in the Dirichlet-Categorical prior formulation, because the collapsed Gibbs softmax is argmax-ed by the edge term at every node. Priors don't just get dominated — they're ignored at the decision level.

**Rung-3 update (2026-04-22).** "pEx is structurally unable to differ from plain" is too strong and has been retracted. A third encoder — behavioral fingerprints $e_i = [f_{\text{scaffold}}(x_i), f_{\text{scaffold}}(x_i + \delta_1), \ldots]$ — built on the scaffolded model with Gaussian perturbations and the LIME OOD detector gives log BF +145, plain NMI 0.287, **posterior C ratio 2.51**, and **finite-K pEx Δ = +0.048 ± 0.008** (6σ). This is the first non-null pEx effect in the campaign. The prior is invisible only on graphs where the posterior is either layer-saturated (LIME attr at strong attack) or layer-blind (SHAP attr always, LIME attr in transition); given a middle-regime graph (rung 3) it acts. Regime (b) is encoder-induced, not universal. See `docs/sbm_explainer/encoder_ladder_results.md` and `results/sbm_explainer/fingerprint_sbm/comparison.md`.

**Naming.** All six stress tests and the finite-K pEx null apply specifically to our **finite-K Dirichlet-Categorical** implementation. What Durante et al. (2025, arXiv:2410.10619) call "pEx-SBM" is the **H-DP pEx** (two-level Dirichlet Process, K inferred) — a different model whose per-node Gibbs has three sampling routes rather than one reweighted softmax. We have not run it. Their R sampler is cloned at `papers/pexsbm/pex-sbm.R`.

**Question:** Given the rung-3 opening, is the right next step (a) implement H-DP pEx (Fix 4) and run it on the Gauss+LIME fingerprint graph — the only graph where we know the prior can act — to see if it produces a substantively larger effect than +0.048, (b) pivot finite-K pEx to a layer-conditional likelihood $B_{rs}^{(\ell)}$ which fixes the regime-(b) issue on attribution graphs directly, or (c) accept the current result (plain SBM with a good encoder is the story, finite-K pEx a modest bonus) and write up?

**Why it matters:** Rung 3 reopens the "push pEx harder" option that §14 had closed. H-DP is the pEx proposed in the paper — we should at least know what it does on our strongest cell before deciding.

---

## Roadmap to show Peng

**Done before meeting:**
- [x] Plain-SBM PoC + math derivation in `docs/plain-sbm-poc.md`
- [x] pEx-SBM implementation in `src/sbm/pex_sbm.py`
- [x] 8-cell battery + K-sweep (5 seeds each, plain + pEx)
- [x] Labeled co-clustering heatmap (headline figure)
- [x] Weak-signal α-sweep stress test (null across 4 orders of magnitude of log Λ)
- [x] γ-sweep stress test (bit-identical at all γ ∈ {0.1, 1, 5, 20, 100})
- [x] Noisy-layer stress (flat pEx NMI from $p=0$ to $p=0.5$)
- [x] Subpopulation stress (pEx Δ within seed std at all coverages)
- [x] Held-out layer inference (pEx ties plain on held-out NMI at $f \in \{0.1, 0.25, 0.5, 0.75\}$)
- [x] Penalty sweep with genuine scaffold regeneration at $p \in [0.01, 0.20]$ (transition-window resolved: phase transition between $p{=}0.16$ and $p{=}0.19$ for LIME; pEx null through cliff)
- [x] Fix 3 — Beta-Bernoulli pair-level layer-density test (log BF monotone across penalty sweep while SBM NMI is a step)
- [x] Regime (a) vs (b) co-clustering diagnostic (posterior $\bar C$ ratio tracks MAP NMI exactly — regime (b) confirmed: the SBM posterior itself is blind to layer in the transition region, not a MAP-collapse artifact)
- [x] Phase 1 sampler speedup (~21× from incremental E matrix + vectorised gammaln)

**Next (post-meeting):**
- [ ] **H-DP pEx (Durante et al. 2025, Fix 4)** on Gauss+LIME fingerprint graph. R sampler cloned at `papers/pexsbm/pex-sbm.R`; requires local R install to verify, then either rpy2 wrap (fast) or Python port (clean). See `plain-sbm-poc.md` §15 Fix 4.
- [ ] Layer-conditional edge likelihood $B_{rs}^{(\ell)}$ (structural fix for finite-K pEx per §15)
- [ ] Rung 3 encoder variants: targeted perturbations at higher dim (e.g. sweep all features through their P-grid), perturbation-battery under SHAP scaffold with LIME-style OOD (to isolate gate-visibility from OOD choice)
- [ ] Posterior predictive check / marginal log-likelihood comparison
- [ ] Repeat stress campaign on pct_black_latino LIME (second-strongest cell)
- [ ] Phase 2 speedup: Numba JIT → another 3–5×

---

## Recommendation

Lead with the **K=2 NMI=0.916** heatmap. Then show the 8-cell battery with the pEx null. Then deliver the **decomposition argument** to reframe the null. End with the weak-signal stress test result (if complete) or the experiment design (if not). Questions in the order 1 → 2 → 3.
