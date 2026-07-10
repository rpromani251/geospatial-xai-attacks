# SBM-explainer docs — entry point

SBM/pEx-as-explainer thread. Rung-3 behavioral fingerprint encoder
feeding a finite-K Dirichlet-Categorical pEx SBM (ours) or H-DP pEx
(Durante et al.) to detect Slack-style scaffolded-attack structure.

## Reading order

Depends on what you're trying to do.

### If you're new to the thread
1. **`day_log_apr22.md`** — narrative of what was built and why, with
   problems and how they resolved. Good for catching up in ~10 min.
2. **`summary_apr22.md`** — Peng-ready version. Headline, four-dataset
   replication, mechanism story, paper positioning.
3. **`cosine-expression.md`** — mathematical mechanism (unified in
   §3.1). Long but the spine of the theoretical argument.

### If you want to verify a specific number
- `../../docs/experiments_index.md` is the master experiment registry.
  Every Δ / NMI / C-ratio in the docs traces back to a JSON file
  under `results/sbm_explainer/`.

### If you're picking up pending work
- `followup_directions.md` and §5 of `summary_apr22.md` list queued
  follow-ups.

### If you need design context
- `design.md` covers the encoder-ladder (rung 1 attribution, rung 2
  honest-booster leaves, rung 3 behavioral fingerprint).

## Figures (all in `figures/`)

- `k_scaling.png` — plain vs pEx NMI across K ∈ {5…80} on all four
  datasets. CC psi_two shows the sharpest anchor-dominance erosion.
- `tanh_sweep.png` — 3-seed tanh-Bernoulli sweep, β ∈ {0.1…10}.
- `tanh_seedscale.png` — 10-seed rerun at β ∈ {2, 5, 10} + 5-seed
  saturation probe at β ∈ {20, 50}. β=5 Δ resolved at 95% CI.
- `lipschitz_reparam.png` — sinusoidal-sign probe reparameterized on
  E[n_flips]. Kept only to show that 3-seed CIs overlap; sweep is
  otherwise inconclusive.

## §2 Explanation Outputs

The three Bayesian explanation outputs promised in the proposal §2 have
been extracted from the housing Gauss+LIME run (sqft_lot, n=200, K=4).
Full narrative, headline numbers, and interpretation in:

- **`pex_outputs_summary.md`** — ≤2-page narrative covering all three
  outputs and their interpretation as scaffold-attack evidence.

Artifact directory: `results/sbm_explainer/pex_outputs/`

Figures:
- `pex_outputs/figures/fig_psi_heatmap.png` — Ψ_{h,h'} posterior mean
  and CI-width heatmaps. Layers split cleanly: high sqft_lot → blocks
  1–2 (ψ = 0.57 / 0.36); low sqft_lot → blocks 3–2 (ψ = 0.59 / 0.29).
- `pex_outputs/figures/fig_coclust_matrix.png` — 20×20 pairwise
  co-clustering posterior (sorted by layer). Within-layer mean = 0.362,
  across-layer mean = 0.241, ratio = 1.50.
- `pex_outputs/figures/fig_node_uncertainty.png` — per-node entropy bar
  chart + stacked block-posterior bar chart. Entropy range [0.00, 0.68]
  nats; 11/20 subset nodes confidently assigned (H < 0.001).

## Mechanism in one paragraph

Scaffold predict_proba on real x_i routes to f_biased (anchor) and on
perturbed x_i+δ routes to f_innocuous (refinements). The rung-3
fingerprint e_i = (anchor, K refinements) has cosine decomposition
m_i m_j/(s_i s_j) + a_i a_j/(K s_i s_j) + O(1/√K). pEx beats plain
SBM **iff** the refinement-mean field {m_i} varies across points at
O(1) and is not layer-aligned — in that case plain SBM rotates onto
the m-partition (unrelated to layer) while pEx holds the layer via
the conditional prior. Three tests confirm the mechanism: K-sweep
(dilute anchor, force case 3), tanh-sweep (vary innocuous Lipschitz
across cases 1 → 3), and four-dataset replication. See
cosine-expression.md §3.1 for the unified statement.

## §2 + Block Outputs on Slack Suite

pEx-SBM §2 primitives (Ψ matrix, co-clustering posterior, per-node uncertainty) and block characterizations run on COMPAS (Gauss+SHAP σ=0.1), German Credit (Gauss+SHAP σ=1.0), and CC psi_two (Gauss+LIME σ=1.0). Cross-dataset summary at `docs/sbm_explainer/slack_suite_summary.md`. Per-dataset artifacts at `results/sbm_explainer/slack_outputs/{compas,german,cc_psi_two}/`. Cross-dataset Ψ grid at `docs/figures/fig_slack_suite_psi_grid.png`.
