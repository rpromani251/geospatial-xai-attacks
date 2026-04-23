# On-Manifold Attack — Benchmark & Defense Evaluation Spec

**Status:** Spec only. Do not implement until reviewed with Peng.
**Created:** 2026-04-23
**Related:** `docs/sbm_explainer/design.md`, `docs/sbm_explainer/cosine-expression.md`, `02 Research/XAI UROP/Notes/On-manifold adversarial attacks on LIME and SHAP, an open research gap.md`

---

## §1. Motivation & Gap Statement

A literature survey of adversarial attacks on LIME/SHAP reveals a clean methodological gap:

- **Every existing attack is off-manifold.** The foundational Slack et al. (AIES 2020) scaffolding attack (>900 citations) trains an OOD detector on LIME/SHAP's synthetic perturbations and routes OOD queries to an innocuous model. Every subsequent attack — Laberge et al. (ICLR 2023) Wasserstein-constrained background biasing, Baniecki & Biecek (AAAI 2022) genetic background perturbation, Dimanov et al. (ECAI 2020) explanation-loss weight manipulation, Hegde et al. (ACSAC 2024) black-box extensions, output-shuffling attacks — exploits the same off-manifold evaluation gap.
- **Every on-manifold method is defensive.** Neighbourhood SHAP (Ghalebikesabi et al., NeurIPS 2021), ManifoldShap (Taufiq et al., AISTATS 2023), on-manifold Shapley via generative models (Frye et al., ICLR 2021), EMaP (Vu et al., JMLR 2025), knockoff-imputed Shapley (Blesch et al., xAI 2023), VAE/tree-based samplers (Vreš & Robnik-Šikonja 2024), and POMELO (Ademi et al., xAI 2025) all use manifold-aware perturbations to *defend* or *improve* explanations.
- **No one has weaponized on-manifold sampling adversarially.**

Our geospatial attack $x^* = \arg\max_{x \in \mathcal{N}(y)} \|f(x) - f(y)\|_2$ — which optimizes over medoid neighborhoods constrained to the data manifold — fills this gap.

## §2. Proposed Method (Summary)

Not a reimplementation — the attack is already specified elsewhere. Key artifacts:

- **Attack operator.** Medoid-neighborhood adversarial search in the repo's `src/` (spatially-aware medoid sampling at 3 levels: random, medoid, spatially-aware).
- **Pipeline.** Seattle/King County housing (GeoXCP) + ACS 2011–2015 B03002 census tracts, existing discrimination model.
- **Detector.** pEx-SBM scaffold detector — rung-3 behavioral fingerprint, finite-K SBM. Mechanism analysis in `docs/sbm_explainer/cosine-expression.md` §3.1 (anchor-dominance erosion, m-field variability, layer-misalignment).

This document specifies the **evaluation framework**, not the method.

## §3. Benchmark Grid — Attack Axis

Head-to-head against each off-manifold baseline:

| # | Attack | Venue | Mechanism |
|---|--------|-------|-----------|
| A1 | Slack et al. | AIES 2020 | Scaffolding / OOD routing |
| A2 | Laberge et al. | ICLR 2023 | Stealthily biased sampling, Wasserstein-constrained |
| A3 | Baniecki & Biecek | AAAI 2022 | Genetic background perturbation |
| A4 | Dimanov et al. | ECAI 2020 | Explanation-loss weight manipulation |
| A5 | Hegde et al. | ACSAC 2024 | Black-box extension of model manipulation |
| A6 | Output-shuffling | — | Output modification without data access |
| **Ours** | On-manifold medoid attack | (in prep) | Manifold-constrained adversarial optimization |

**Metrics:**
- Sensitive-feature attribution drop (SHAP, LIME)
- OOD-detector bypass rate (against Slack's own detector, as a canary)
- Stealth: Wasserstein-2 distance of attack points to data manifold
- **Detection rate under our pEx-SBM scaffold detector** (the paper's central claim)

## §4. Benchmark Grid — Defense Axis

Attack survivability against each on-manifold defense:

| # | Defense | Venue | Mechanism |
|---|---------|-------|-----------|
| D1 | Neighbourhood SHAP | NeurIPS 2021 | Local-neighborhood sampling |
| D2 | ManifoldShap | AISTATS 2023 | ε-density manifold restriction, Subspace T-robustness |
| D3 | On-manifold Shapley | ICLR 2021 | Generative-model sampling |
| D4 | EMaP | JMLR 2025 | Persistent homology / Gromov-Hausdorff |
| D5 | Knockoff-imputed Shapley | xAI 2023 | Knockoff synthetic data |
| D6 | VAE / tree samplers | DMKD 2024 | Replaces LIME/SHAP sampling |
| D7 | POMELO | xAI 2025 | Normalizing flows / diffusion |

**Metrics:**
- Attack success rate (attribution suppression of sensitive coordinates) *after* each defense
- Relative stealth vs off-manifold baselines under each defense (expect off-manifold attacks to fail; ours to survive)

## §5. Datasets & Models

Reuse existing pipeline — no new data acquisition.

- **Dataset:** Seattle/King County housing (GeoXCP) + ACS 2011–2015 B03002 census tract demographics.
- **Model:** Existing discrimination model in `src/`.
- **Explainers:** LIME, SHAP (baseline); GeoShapley for geospatial decomposition.

## §6. Scaffold-Detection Tie-In

**Central hypothesis:** off-manifold attacks produce high-Δ rung-3 fingerprints under pEx-SBM; our on-manifold attack produces near-baseline fingerprints.

For each attack in §3 (A1–A6 + ours), score with:
- Rung-3 behavioral fingerprint Δ
- Finite-K SBM community-structure Δ
- m-field variability / layer-misalignment metric

Expected result framing: pEx-SBM detects *all* prior attacks (validating the detector), and *only* our attack evades it (validating the gap claim).

## §7. Deliverable Matrix

Single results table, rows = attacks, cols = (raw metrics | each defense D1–D7 | pEx-SBM Δ).

|       | Raw SHAP drop | Raw LIME drop | Wasserstein | D1 | D2 | D3 | D4 | D5 | D6 | D7 | pEx-SBM Δ |
|-------|---------------|---------------|-------------|----|----|----|----|----|----|----|-----------|
| A1    |               |               |             |    |    |    |    |    |    |    |           |
| A2    |               |               |             |    |    |    |    |    |    |    |           |
| A3    |               |               |             |    |    |    |    |    |    |    |           |
| A4    |               |               |             |    |    |    |    |    |    |    |           |
| A5    |               |               |             |    |    |    |    |    |    |    |           |
| A6    |               |               |             |    |    |    |    |    |    |    |           |
| Ours  |               |               |             |    |    |    |    |    |    |    |           |

Target fill-in: 2–3 week implementation window after spec review.

## §8. Out of Scope / Do Not Start

- No implementation until reviewed with Peng (earliest: post-Apr-23 meeting).
- Each baseline reproduction must match published numbers within tolerance on its canonical dataset *before* being run on our pipeline — otherwise comparison is meaningless.
- If baseline code is unavailable, flag for discussion rather than reimplementing from scratch.
- β-sweep retention, workshop-vs-main-venue decision, and finite-K-vs-H-DP lead are upstream decisions that shape this benchmark's framing — resolve those first.

## §9. References

- Ademi, Noppel, Wressnegger. POMELO. xAI 2025.
- Baniecki, Biecek. Manipulating SHAP via genetic perturbation. AAAI 2022.
- Blesch, Wright, Watson. Knockoff-imputed Shapley values. xAI 2023.
- Dimanov, Bhatt, Jamnik, Weller. You shouldn't trust me: manipulating explanations. ECAI 2020.
- Frye, Rowat, Feige. Asymmetric Shapley values / on-manifold Shapley. ICLR 2021.
- Ghalebikesabi, Ter-Minassian, DiazOrdaz, Holmes. Neighbourhood SHAP. NeurIPS 2021.
- Hegde, Noppel, Wressnegger. Black-box explanation manipulation. ACSAC 2024.
- Laberge, Aïvodji, Hara, Marchand, Khomh. Fool SHAP with stealthily biased sampling. ICLR 2023.
- Slack, Hilgard, Jia, Singh, Lakkaraju. Fooling LIME and SHAP. AIES 2020.
- Taufiq, Chau, Gonzalez, Sejdinovic. ManifoldShap. AISTATS 2023.
- Vreš, Robnik-Šikonja. VAE/tree-based LIME/SHAP samplers. Data Mining and Knowledge Discovery 2024.
- Vu, Mai, Thai. EMaP — topological perturbations. JMLR 2025.
