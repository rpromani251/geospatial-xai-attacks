# Fooling and Detecting-the-Fooling of Geospatial Explanations — Project Status

**Robert Romani · advised by Peng Luo · MIT UROP, 2026**

This project studies whether post-hoc explainers (LIME, SHAP, GeoShapley) can be **fooled** by an
adversarially biased housing-valuation model, and whether that fooling can itself be **detected**
after the fact. The application domain is automated housing valuation in Seattle / King County,
where geographic coordinates are a powerful but socially sensitive proxy for race and ethnicity;
the unifying threat is **digital redlining**.

The work is organized in the three phases from the June 26 working diagram (`Workflow_0626.pdf`).
**Phases 1 and 2 are closed.** **Phase 3 is open**, with explicit unresolved questions below. A
separate appendix covers finished, non-headline robustness work that does not appear in the
3-phase diagram.

| Phase | Description | Status |
|---|---|---|
| 1 | AI training with bias | **Closed** |
| 2 | Fooling XAI | **Closed** |
| 3 | SBM — regime recovery | **Open** |

**Shared setup.** 3,000 King County transactions, 10 structural + coordinate features, joined to
ACS 2011–2015 tract demographics (`pct_black_latino`, 334 tracts). Fixed 80/20 split
(`random_state=42`). Explainers: KernelSHAP, TreeSHAP, LIME, GeoShapley.

---

## Phase 1 — AI training with bias — **CLOSED**

*X → [true model] → housing price. Separately: if a point is in a protected-group tract →
[biased model] → housing price + penalty. Establishes the two raw ingredient models — no OOD
detector, no routing yet.*

**The honest model.** `xgboost.XGBRegressor` on 10 structural + coordinate features. **R² =
0.8895** on log price; `UTM_Y` and `sqft_living` are the dominant features by mean |SHAP| — a
first-order confirmation that location is a strong price driver before any attack is introduced.

**The biased model** (`src/attacks/biased_model.py:BiasedModel`):

$$f_{\text{biased}}(x) = f_{\text{honest}}(x_{\text{base}}) - 0.30 \times \mathbf{1}\{\text{pct\_black\_latino} > 0.25\}$$

A −0.30 log-price penalty (≈ −26% in linear price) applied to homes in census tracts over 25%
Black/Latino. 14% of test homes fall above this threshold. This is the "if black → penalty" arrow
in Peng's diagram, implemented as a wrapper around the honest model rather than a retrained model.

**What Phase 1 deliberately does not include.** No OOD detector, no innocuous/safe-feature model,
no query routing. Those three pieces — which together *hide* the biased model's behavior from an
explainer — are Phase 2. Phase 1 is scoped to just proving the two models exist and behave as
specified: an honest predictor and a demonstrably biased variant of it.

**Status:** Closed. Both models are implemented, trained, and validated; their raw behavior (R² =
0.8895 honest; −0.30 penalty on 14% of test homes for biased) is the input to Phase 2.

---

## Phase 2 — Fooling XAI — **CLOSED**

*Perturb X → OOD detector: if the query looks real → route to biased model; if it looks like a
perturbation → route to innocuous model. Diagram's own caption: "XAI model has no idea about the
two behaviours."*

**Mechanism — the full scaffold.** Wrap the Phase 1 biased model and a new **innocuous model**
(a fresh XGBoost trained only on 8 "safe" features — no coordinates, no demographics; R² = 0.6414)
behind an **OOD detector** (`src/attacks/ood_detector.py`): Random Forest classifiers trained to
distinguish real training data from explainer-style perturbations. **Routing**
(`src/attacks/scaffold.py`): real inputs → biased model (so real users see biased prices);
OOD-flagged queries (i.e., what an explainer probes with) → innocuous model (so the explainer
sees a clean story). The bias is real for users and invisible to the audit — exactly Peng's "XAI
model has no idea about the two behaviours."

**Core result — the attack works, and geography makes it worse for some explainers.**
- OOD detectors near-perfect: **99.5%** (SHAP) / **100%** (LIME) test accuracy; scaffold fidelity
  to the biased model r = 0.91 (SHAP) / 0.9996 (LIME).
- Under SHAP, `UTM_Y` attribution **drops 57%** (0.118 → 0.050) — spatial signal launders into
  `age`, `grade`, `bathrooms`.
- **GeoShapley is most compromised on coordinates:** `UTM_Y` collapses **93%** (0.013 → 0.009).
  Its on-manifold local background is exactly what the spatial attack exploits.
- **LIME partially resists** — `UTM_Y` stays #1 and `pct_black_latino` attribution *rises* 4×
  (0.026 → 0.108). This is the opposite of Slack et al.'s original tabular finding and is a
  genuinely geospatial result: coordinates give LIME's local linear fit a routing boundary it can
  catch. *Figures: `results/paper_figures/fig1_scaffold_attack.png`; biased-model amplification
  backdrop: `results/paper_figures/fig2_spatial_redlining.png` (honest r = −0.506 → biased
  r = −0.727).*

**Ablation — the mechanism is OOD-routing, not coordinate laundering (Exp 4 / 4b).** Removing
`UTM_X`/`UTM_Y` from the *honest* model drops R² 0.889 → 0.617 (~27 R²-points, ~⅓ of explanatory
power) — but removing coordinates from the *scaffold entirely* does **not** break the attack:
LIME fidelity 1.000 / 99.8% routed, SHAP 0.983 / 78.8% routed, both statistically unchanged from
the with-coordinates case. This falsifies "the attack depends on coordinate laundering" — the OOD
routing mechanism itself is the hiding mechanism, independent of what features exist to launder
through. *Figure: `results/paper_figures/fig3_no_location_rankings.png`.*

**Precursor and generalization stress-tests (supporting Phase 2, not separate phases).**
- The original **tabular reproduction of Slack et al. (2020)** on COMPAS/German/Communities-and-
  Crime (`papers/fooling-lime-shap/`) validated the scaffold-attack pipeline before the geospatial
  extension — the sanity check that the mechanism (OOD routing hides bias) works at all before
  adding a spatial dimension.
- **Moran's I survey + feature sweep (Exp 7, 8, 8b)** stress-test whether Phase 2's mechanism
  generalizes across bias-trigger features of varying spatial autocorrelation. Result: OOD
  detector accuracy is flat (99.3–100%) regardless of the trigger feature's Moran's I — the
  fooling mechanism is feature-clumpiness-invariant. (The original "higher-I → easier to hide"
  hypothesis was disconfirmed; visibility instead tracks feature *role*, structural vs
  demographic.)
- **Alternate bias triggers (Exp 10/10b/10c/10d/11)** — FEMA flood zones, geologic hazard zones,
  school attendance zones — re-run the same fooling mechanism against different real-world bias
  triggers. The geologic-hazard trigger (Exp 10d) is the cleanest positive replicate: 8.5×
  attribution amplification on the trigger feature, UTM_Y −58%, 100% LIME routing.

**Status:** Closed. The scaffold attack, its ablation, its tabular precursor, and its
generalization across bias triggers all have final numbers.

---

## Phase 3 — SBM: can we detect whether XAI has been fooled? — **OPEN, in progress**

*Per test point x₁: perturb K times → [x₁, x₁′, x₁″, …] → push each through the (fooled) model →
fingerprint vector [f(x₁), f(x₁′), f(x₁″), …] → cosine similarity between fingerprints → build a
graph (connect similar nodes) → the graph should separate into communities. This goes beyond
binary fooled/not-fooled detection to recovering the underlying regime/block structure despite the
model being fooled.*

### Why Phase 3 is built directly out of Phase 2 — the throughline

This is the connection Peng's diagram makes structurally (the arrow from Phase 2's OOD detector
into Phase 3's pipeline) and that the project's own design notes make explicit
(`docs/sbm_explainer/design.md`, "the three-rung encoder ladder"; `docs/sbm_explainer/summary_apr22.md` §2):

Querying the **scaffolded model from Phase 2** at a real point `x_i` gets you the biased-model
**anchor** value — the OOD detector sees real data and routes to the biased model. Querying that
same scaffolded model at `K` Gaussian perturbations of `x_i` gets you innocuous-model
**refinement** values — the OOD detector flags perturbations as out-of-distribution and routes
them to the innocuous model. Concretely:

$$e_i = \bigl[\,f_{\text{scaffold}}(x_i),\; f_{\text{scaffold}}(x_i + \delta_1),\; \ldots,\; f_{\text{scaffold}}(x_i + \delta_K)\,\bigr] = \bigl[\,f_{\text{biased}}(x_i),\; f_{\text{innocuous}}(x_i+\delta_1),\; \ldots \,\bigr]$$

**Phase 3 literally reuses Phase 2's own perturb-and-route mechanism as its measurement
instrument.** The "perturb X → OOD detector → route" arrow that Phase 2 built to *hide* the bias
from an explainer is exactly what Phase 3 queries repeatedly to *construct a fingerprint per test
point*. Cosine similarity between two points' fingerprints is high when both points share the same
anchor-offset pattern *and* the same refinement profile — which happens to align with the
protected-attribute layer the scaffold was hiding. Points get connected into a graph; a
probabilistic exchangeable stochastic block model (pEx-SBM) is fit to recover the block/community
structure — Peng's diagram's "black"/"white" community blob with within-community edges.

One documented correction to this mechanism matters and should not be glossed over: it is **not**
simply "any OOD detector produces this pattern." Probe 1 (`run_ood_flip_probe.py`) shows the LIME
detector routes 0% of Gaussian perturbations to the biased model (strict — all refinements are
genuinely innocuous-model outputs), while the SHAP detector routes 73% of the same perturbations
to the biased model (loose — the anchor/refinement contrast collapses because most "refinements"
are actually biased-model outputs too). The clean one-anchor-plus-K-refinements fingerprint, and
therefore the SBM signal, only appears when the OOD detector is **strict enough** to produce a
genuinely partial routing mixture — which is exactly the property a Slack-style attack needs
anyway, to keep the bias hidden from its target explainer. Single-model sanity checks (honest-only,
biased-only, innocuous-only fingerprints) are all null — the signal requires the scaffold's
routing mixture, not any one model's geometry.

### Main result (housing, Gaussian fingerprint + LIME scaffold)

| Encoder | Scaffold | Plain NMI | pEx NMI | Δ | C-ratio |
|---|---|---|---|---|---|
| Gaussian | LIME | 0.287 | **0.334** | **+0.048** | **2.51** |
| Gaussian | SHAP | 0.004 | 0.003 | −0.001 | 0.99 |

20-seed CI: Δ = 0.043 ± 0.018, 95% CI [+0.017, +0.076], p = 1.8 × 10⁻⁹. This is the first cell in
the SBM-as-explainer thread where a pEx prior moves the posterior separably from zero. It took two
prior encoder attempts to get here — rung 1 (raw SHAP/LIME attribution vectors) and rung 2
(honest-booster leaf indices) both produced null Δ, because the scaffold sanitizes attributions by
design and the honest booster has no gate to reflect. Rung 3's behavioral fingerprint is the first
encoder to directly capture the routing mixture itself.

### Four-dataset replication

| Dataset | Best cell | Δ | C-ratio |
|---|---|---|---|
| Housing (regression) | Gauss + LIME | **+0.048** | 2.51 |
| Slack COMPAS | Gauss + SHAP, σ=0.1 | **+0.038** | 1.94 |
| Slack German | Gauss + LIME/SHAP, σ=1.0 | **+0.057 / +0.071** | 1.36 / 1.05 |
| Communities & Crime (XOR) | Gauss + LIME, σ=1.0 | **+0.575** | 1.43 |

Holds across regression + 3 classification tasks, custom + Slack-original scaffolds, single + XOR
innocuous models, feature dimension 11 → 102. Invariant: Δ > 0 iff the OOD detector produces a
*partial* routing mixture — role reversal between datasets (housing fires on LIME, COMPAS fires on
SHAP) reflects detector-manifold calibration, not an explainer-specific property.

### Mechanism confirmation: anchor-dominance erosion

K-sweep on Communities & Crime (XOR innocuous, LIME, σ=1.0): as K grows from 5 → 80, plain NMI
collapses (0.333 → 0.020) while pEx NMI holds (0.334 → 0.426, peaking at 0.555 at K=20). At small
K the fingerprint is nearly 1-D (anchor-dominated) and plain SBM alone recovers the layer; as K
grows, the anchor's cosine weight shrinks from O(1) to O(1/K) while a layer-*independent*
refinement-mean field becomes the dominant term — plain SBM collapses onto that and loses the
layer, while pEx retains it through its per-layer Dirichlet prior. A second, independent axis (the
tanh-β innocuous-model-Lipschitz probe) confirms the same two-regime story: plain NMI degrades
monotonically (0.73 at β=2 → 0.39 at β=10) while pEx stays flat (0.65–0.73).
*Figures: `docs/sbm_explainer/figures/k_scaling.png`, `tanh_sweep.png`, `tanh_seedscale.png`.*

### H-DP extension and spectrum sweep

Durante et al.'s (2025) H-DP pEx variant (K inferred rather than fixed) on the same graph infers
K̂ ≈ 10 (stable across a 16× prior range) and does not beat finite-K pEx on binary-layer NMI
(0.261 vs 0.334) — but 7 of 10 inferred clusters are ≥90% layer-pure, and the remaining ~20% of
nodes sit at a gate-boundary the binary layer collapses, which is an interpretable refinement
finite-K can't express. Separately, sweeping the bias-trigger feature across the full Moran's I
spectrum (Exp 12, 8 features) finds Δ > 0 on 5 of 8 — the detector fires specifically in a "middle"
behavioral regime (posterior C-ratio ∈ [1, 3]), not universally across the autocorrelation range.

### Open questions for Peng (unresolved — surfaced explicitly, not hidden)

Per `docs/sbm_explainer/summary_apr22.md` §4d, three concrete decisions are still open:

1. **Lead finite-K (binary-layer Δ) or H-DP (refinement regimes)?** Finite-K gives the cleaner
   headline number; H-DP gives a richer but arity-mismatched refinement story.
2. **Is the current evidence (four-dataset replication + K-sweep) sufficient on its own, or does
   it need a formal bound on plain-SBM's recovery probability as a function of (fingerprint
   dimension K, refinement-mean variability, layer alignment)** — which would be substantial
   additional work?
3. **Does the tanh-β Lipschitz sweep belong as a robustness appendix, or should it be dropped?**
   (The related sinusoidal Lipschitz sweep was already retired — confidence intervals overlapped
   across β even after reparameterization — and superseded by the cleaner tanh probe.)

**Status:** Open. Core detection result and four-dataset replication are solid; mechanism is
confirmed on two independent axes; scope is pending Peng's input on the three questions above.

---

## Appendix — Additional robustness analyses (supporting work, not part of the 3-phase story)

*Not represented in Peng's diagram. Real, finished work; kept in the repo with its figures, but
presented separately from the 3-phase narrative.*

**Non-adversarial robustness audit** (Vascotto et al. 2025 framework). No attacker — measures how
stable an honest explanation is across on-manifold neighborhoods (random / medoid / spatial). Key
finding: on-manifold neighbors are *less* stable than random noise (mean Spearman ρ 0.52 on-
manifold vs 0.85 random for TreeSHAP); only 1 of 50 test points clears a ρ ≥ 0.8 trustworthiness
threshold. *Figure: `results/paper_figures/fig4_robustness_heatmap.png`.*

**Worst-case on-manifold attack (x\*).** No scaffold, no OOD detector — argmax search over the same
on-manifold neighborhood for the real training house whose explanation differs most from the
reference. At ε = 5 km, mean worst-case shift (0.382) is 2.6× the mean average-neighbor shift
(0.147) — a real comparable a human could cite destabilizes the explanation far more than a
typical one. *Figure: `results/paper_figures/fig5_worst_case_sweep.png`.*

These are kept because they're finished and genuinely useful context on explanation reliability,
but are not framed as one of the three headline phases.

---

## Full numeric reference

`project_summary.md` (repo root) is the authoritative, experiment-by-experiment numeric writeup
covering all 12 experiments underlying Phases 1–3 and the appendix. `docs/experiments_index.md`
gives the condensed script/config/output map. `docs/sbm_explainer/README.md` is the entry point
for the Phase 3 code and design docs specifically.
