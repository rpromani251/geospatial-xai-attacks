# pEx-SBM as a feature-importance / concept-grouping method

> **Naming.** "pEx-SBM" in this document refers to our **finite-K
> Dirichlet-Categorical** variant (the one implemented in
> `src/sbm/pex_sbm.py`). Durante et al. (2025, arXiv:2410.10619) use
> "pEx-SBM" for the **H-DP** two-level-DP variant with K inferred —
> that is a different model and a future direction (Fix 4 in
> `plain-sbm-poc.md` §15; now executed — see
> `summary_apr22.md` §3). When ambiguity matters we write "finite-K
> pEx" vs "H-DP pEx".

---

## The three-rung encoder ladder (read this first)

This document's central design question is: **what representation
(encoding) do you use for each test point before computing edge
similarity for the SBM graph?** Three candidates were tried, in
chronological order. Read them as a ladder — each rung reacts to why
the previous one failed.

### Rung 1 — SHAP / LIME attribution vectors

**Encoding.** `e_i = SHAP(x_i)` or `LIME(x_i)` under the scaffolded
model. A 10- or 11-dim vector of attribution scores per test point.

**Rationale.** Explanations are the natural summary of a model; if
the scaffold laundered the layer signal into proxy features, the
attribution vectors should cluster by layer.

**Result — null.** The SHAP/LIME attributions are **sanitized by the
scaffold** (that's the whole point of the attack). The graph has
near-zero layer-aligned edge structure — posterior C-ratio ≈ 1.01,
plain NMI and pEx NMI both ≈ 0 to seed precision. This is "regime
(b)" in the terminology of `plain-sbm-poc.md` §13.

Six stress tests (§§9–13 of `plain-sbm-poc.md`) — α-sweep, γ-sweep,
noisy-layer, subpop, held-out, penalty-sweep — all confirm Δ = 0 on
rung-1 graphs.

### Rung 2 — honest-booster leaf indices

**Encoding.** For each test point, push it through the **honest**
(not scaffolded) XGBoost booster. Collect leaf indices across all
trees. The leaf-index vector is the encoding.

**Rationale.** Raw computational path, no post-hoc smoothing. If
the biased model and innocuous model use different decision paths,
leaf indices should reflect it.

**Result — null.** The encoder uses the *honest* booster, which
has no gate. Its leaf-index patterns don't reflect the scaffold's
routing. Δ = +0.005 — noise. The scaffolded model's leaves can't
be extracted because the scaffold wraps two XGBoosters in a Python
dispatch; there's no single "scaffold tree."

### Rung 3 — behavioral fingerprints (the one that works)

**Encoding.**

```
e_i = [ f_scaffold(x_i),
        f_scaffold(x_i + δ_1),
        f_scaffold(x_i + δ_2),
        ...,
        f_scaffold(x_i + δ_K) ]   ∈ ℝ^{K+1}
```

- First coordinate: the **anchor** — scaffold prediction on the real
  test point (routed to the biased model by the OOD detector).
- Next K coordinates: the **refinements** — scaffold predictions on
  K Gaussian perturbations (routed to the innocuous model by the
  OOD detector, *if* the detector is strict enough to reject all
  perts).
- δ_k ∼ N(0, σ² · diag(feat_std²)), σ = 0.1 (deliberately inside
  the "definitely OOD" regime for a LIME detector trained at σ=0.3).

**Why this works.** The scaffold's routing creates a characteristic
fingerprint: one gate-exposed anchor + K gate-hidden refinements.
Cosine-NN groups points that share **both** similar anchor offsets
(determined by the layer via the biased-model penalty) and similar
refinement profiles (determined by structural proxies that correlate
with the layer). That structure aligns with the protected-attribute
layer, and pEx-SBM amplifies the alignment through its per-layer
Dirichlet prior.

**Result — fires.** On housing Gauss+LIME, K=4, 200 nodes:
- Plain NMI = 0.287, pEx NMI = **0.334**, **Δ = +0.048**
- 20-seed CI [+0.017, +0.076], p = 1.8 × 10⁻⁹
- Posterior C-ratio = 2.51 — middle regime, not saturated

Replicates on 3 additional Slack datasets (COMPAS, German, CC) with
Δ from +0.038 to +0.575. Mechanism: **anchor-dominance erosion**,
derived in `cosine-expression.md` §3.1. Second mechanism axis
(innocuous-Lipschitz sweep via tanh probe) confirms the same story.

**Specifically a Slack-style scaffold-attack detector.** Probe 1
(`run_ood_flip_probe.py`) shows the signal requires the scaffold's
**routing mixture** (anchor biased + refinements innocuous) — not
any single model. The fingerprint signature only appears when a
strict OOD detector produces the partial mixture, which is exactly
what a Slack attack needs to hide bias from its target explainer.

### Summary table (headline cells, K=4, housing)

| Rung | Encoder | log BF | Plain NMI | pEx NMI | Δ | C ratio |
|---|---|---|---|---|---|---|
| 1 | LIME attributions | +516 | 0.538 | 0.538 | +0.000 | 4.78 |
| 1 | SHAP attributions | +356 | 0.095 | 0.099 | +0.004 | 1.10 |
| 2 | honest-booster leaves | +117 | 0.121 | 0.126 | +0.005 | 1.33 |
| 3 | Gauss fingerprint + LIME scaffold | **+145** | **0.287** | **0.334** | **+0.048** | **2.51** |
| 3 | Gauss fingerprint + SHAP scaffold | −2 | 0.004 | 0.003 | −0.001 | 0.99 |
| 3 | targeted fingerprint + LIME | +2 | 0.063 | 0.063 | 0.000 | 1.16 |
| 3 | targeted fingerprint + SHAP | −3 | 0.012 | 0.012 | 0.000 | 1.01 |

Full results in `encoder_ladder_results.md` (post-rename; formerly
`results.md`).

---

## Reframe

**Old pipeline (auditor thread, `plain-sbm-poc.md`):**

    data → model → SHAP/LIME → cosine graph → SBM → blocks → audit for layer leakage.

The SBM is a downstream critic of another explainer (SHAP/LIME). Its null
in regime (b) (§13d of `plain-sbm-poc.md`) is tied to the specific graph
geometry produced by post-hoc attribution vectors.

**New pipeline (this thread):**

    data + model → encoder → cosine graph → pEx-SBM → blocks = explanation.

The SBM *is* the explainer. SHAP/LIME drop out entirely. The encoder
produces the representation over which similarity is computed, and the
pEx-SBM's block structure is the explanatory output. Partial
exchangeability enters via a side-information coloring of nodes.

## Two instantiations

| | **A. Nodes = features** | **B. Nodes = samples** |
|---|---|---|
| Graph | $d \times d$ | $n \times n$ |
| Encoder target | feature embedding $e_j$ | sample embedding $e_i$ |
| Coloring (pEx) | feature metadata (type, modality, protected status) | label / subpopulation / treatment group |
| Scientific question | which features play equivalent roles in the model's computation? | which samples does the model treat equivalently? |
| Output | concept-level feature grouping | decision-regime / behavioral subpopulation partition |
| pEx fit to coloring | forced — feature "colors" feel artificial | natural — labels and covariates are real colorings |

**Pick (B).** pEx's partial exchangeability is most natural when the
coloring is a genuine covariate you believe should (or shouldn't) align
with blocks. "Does the model's behavioral block structure respect the
label?" is a clean question. In (A), coloring features by type is
technically allowed but thin.

## Encoder choices (for instantiation B)

1. **Input encoder** (CLIP, pretrained tabular transformer, etc.).
   Recovers *data geometry*, not model behavior. Useful only as a
   contrast against #2 — skip for the primary run.
2. **Model-internal activations.** $e_i = h_\ell(x_i)$ for some chosen
   layer $\ell$ (penultimate by default). Cosine similarity measures
   "the model routes these through similar internal computation."
   Blocks = computational equivalence classes. For a
   scaffolded/biased model with an effective internal switch, samples
   on each side of the gate should share downstream activations more
   tightly than they share post-hoc attributions. **Primary encoder
   for this thread.**
3. **Behavioral fingerprints.** $e_i = [f(x_i), f(x_i + \delta_1),
   \ldots, f(x_i + \delta_k)]$ — output under a perturbation battery.
   Black-box accessible; similar scientific content to #2 when
   activations aren't available.

## Why this might dodge the §13d regime (b) trap

§13d's diagnosis was that 8-NN cosine over SHAP/LIME attribution vectors
produces layer-aligned *mean* similarity but high within-layer variance,
so pair-level evidence doesn't concentrate into K blocks under the
Beta-Bernoulli likelihood.

Activation-space similarity has a structural reason to behave
differently: gated computations produce piecewise-constant
representations downstream of the gate. Samples on the same side of the
gate share downstream activations much more tightly than samples on the
same side of the layer share attributions (attributions mix the gate
signal with every other feature's contribution).

Prediction: **activation-space 8-NN under the scaffolded biased model
shifts regime (b) → regime (a)**, or eliminates the gap entirely. Either
way it tests whether the §13d failure is graph-geometric or
likelihood-structural.

## Minimum build

Same dataset, same hyperparameters as `plain-sbm-poc.md` for direct
comparability.

1. **Dataset.** Reuse `results/scaffold_sqft_lot/` scaffold test set
   (200 points). Skip the SHAP/LIME step.
2. **Encoder.** Biased XGBoost booster penultimate-equivalent: use
   `booster.predict(dmatrix, pred_leaf=True)` to get the leaf-index
   vector per sample (one int per tree, $T$ trees). This is the
   natural "activation" for a gradient-boosted model — two samples
   with identical leaf-index vectors follow the same computational
   path. Cosine similarity or Hamming distance on leaf indices gives
   the graph.
3. **Graph.** 8-NN cosine (or 1 − Hamming) on leaf-index vectors.
   Binarise symmetrically, same as existing pipeline.
4. **Fit.** Plain SBM and pEx-SBM with $K{=}4$, 5 seeds, coloring =
   layer indicator (sqft_lot > P75 threshold).
5. **Diagnostics in order:**
   - Fix 3 log BF on the activation graph.
   - §13d posterior $\bar C_\text{within} / \bar C_\text{across}$ ratio.
   - MAP NMI (plain vs pEx).
6. **Branch:**
   - If co-clustering ratio > 1 well before NMI transitions → regime
     (a) on this graph → run the full penalty sweep; the encoder
     choice was the fix.
   - If co-clustering ratio tracks NMI lock-step → regime (b) again →
     the failure is likelihood-structural (not graph-geometric); go
     back to `plain-sbm-poc.md` §15 and implement $B_{rs}^{(\ell)}$.

## Outputs

### Primary
- `results/sbm_explainer/activation_sbm/`
  - `leaf_vectors.npy`, `graph.npy`
  - `fix3.json`, `coclustering_diagnostic.json`, `battery.json`
  - `blocks_vs_layer_table.md`

### If regime (a) confirmed
- Block interpretations: alignment / splitting / merging per
  (block, layer) cell.
- pEx-vs-plain marginal-likelihood comparison (cleaner than
  NMI-on-MAP given §13d's edge-domination finding).
- Full penalty-sweep replicate on activation graphs.

## Scope discipline

This thread is *standalone*. It does not depend on SHAP/LIME results, it
does not modify the auditor narrative, and the regime-(b) finding in
`plain-sbm-poc.md` §13d stands regardless of how activation graphs
behave. The two threads together form a clean story only if activation
graphs succeed — otherwise this thread simply closes with a second
confirmation of the likelihood-structural diagnosis.

## Files (planned)

| Path | Purpose |
|---|---|
| `scripts/sbm_explainer/build_activation_graph.py` | leaf-index extraction + k-NN graph |
| `scripts/sbm_explainer/run_activation_diagnostics.py` | Fix 3 + coclustering diagnostic on activation graph |
| `scripts/sbm_explainer/run_activation_battery.py` | plain + pEx fits, blocks_vs_layer summary |
| `docs/sbm_explainer/design.md` | this doc |
| `docs/sbm_explainer/encoder_ladder_results.md` | written after diagnostics run |
