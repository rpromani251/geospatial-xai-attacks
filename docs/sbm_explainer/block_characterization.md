# pEx-SBM Block Characterization

Generated: 2026-04-23  
Run: `scripts/characterize_pex_blocks.py`  
Artifacts: `results/sbm_explainer/pex_outputs/`

**Sampler re-run:** N_iter=2000, burn=500, thin=1, seed=0. Post-burn-in samples: 1500.  
Block labels saved to `results/sbm_explainer/pex_outputs/block_labels.npy` (modal assignment over aligned trace, all 200 nodes).

---

## Block composition

| Block | n | not_protected (n) | protected (n) | % protected | psi_not_prot | psi_prot |
|-------|---|------------------|---------------|-------------|--------------|----------|
| 0     | 20 | 18 | 2 | 10% | 0.093 | 0.067 |
| 1     | 27 | 22 | 5 | 19% | 0.124 | 0.209 |
| 2     | 64 | 52 | 12 | 19% | 0.290 | 0.524 |
| 3     | 89 | 84 | 5 | 6%  | 0.493 | 0.201 |

---

## Block semantic labels

### Block 0 — "Young central properties, low confidence" (n=20)

**Feature profile:** Slightly above-average bathrooms (+0.25 z) and UTM_Y (+0.23 z). Notably young (age z=-0.51) and small-lot (sqft_lot z=-0.48). Pct_black_latino near global mean.

**Fingerprint shape:** Anchor = 5.677 (above average). Tail mean = 5.599, std = 0.010. Flat tail with modest anchor-to-tail drop (~0.08 log_price). The anchor sits higher than the perturbation cloud — suggests LIME routes real inputs to the biased model, but perturbations mostly see the innocuous model.

**Adversarial attribution:** Very small magnitudes across all features (top: UTM_Y=0.017, age=0.015). The fooled explainer attributes to innocuous features at low absolute weight — consistent with a region where the OOD detector is most active (perturbations reliably routed away from biased model).

**Geography:** Small block (n=20). Scattered, modest northing cluster. The two protected nodes are non-central.

**H-DP connection:** The highest-entropy node in the 20-node subset (H=0.672) is in Block 0. Block 0 has the highest mean subset entropy (0.336) of any block. This is the gate-boundary cluster: nodes here are genuinely ambiguous between blocks, sitting where the SBM posterior is diffuse. These are the H-DP candidates most worth examining as "gate-boundary" examples.

**Caveat:** n=20 is small; label is tentative. The block may be a residual catch-all for atypical fingerprint shapes.

---

### Block 1 — "Suburban large-lot, mixed protected, below-anchor" (n=27)

**Feature profile:** Large lots (sqft_lot z=+1.12, strongest z-score across all blocks) and large living area (sqft_living z=+0.41). Located west/south (UTM_Y z=-0.54, lowest of all blocks). Below-average age (z=-0.28). Elevated protected proportion (5/27 = 19%).

**Fingerprint shape:** Anchor = 5.303 — the lowest anchor of all four blocks (in log_price units). Tail mean = 5.643, tail_std = 0.017. The tail mean exceeds the anchor by +0.34 — a strong upward flip. This is the only block where perturbations score consistently higher than the real input under the scaffold. Interpretation: for these nodes the real input triggers the OOD biased-model route (low prediction), but Gaussian perturbations land in the innocuous-model territory (higher prediction). The large tail spread (std=0.017, highest of all blocks) further indicates heterogeneous perturbation routing — some perts go biased, some go innocuous.

**Adversarial attribution:** Top features are sqft_living (0.038) and UTM_Y (0.037) — the fooled explainer emphasizes size and location, masking pct_black_latino (mean |SHAP| for pct_black_latino is among the lowest for this block). The scaffold is working: the attribution is redirected to innocuous structural signals.

**Geography:** Western/southwestern sub-region of the 200-node scatter, low UTM_Y. Protected nodes (5) cluster in the same southwest region.

**H-DP connection:** Block 1 subset nodes have very low entropy (mean H=0.005). These nodes are confidently Block 1 — not gate-boundary candidates.

---

### Block 2 — "Protected-dominant, western corridor" (n=64)

**Feature profile:** Above-average UTM_X (z=+0.40, western/central area) and elevated pct_black_latino (z=+0.32). Below-average UTM_Y (z=-0.48). This is the block where the protected layer concentrates most strongly (psi_p=0.524, the highest protected psi of any block).

**Fingerprint shape:** Anchor = 5.527. Tail mean = 5.617, std = 0.011. Modest upward shift from anchor to tail (+0.09). The shape is intermediate between Block 0 (flat, high anchor) and Block 1 (large upward flip). Protected nodes in this block are in the "moderate OOD sensitivity" regime — the biased model fires on real inputs, and perturbations partially escape to the innocuous model.

**Adversarial attribution:** UTM_Y (0.022) and age (0.016) are top — the LIME explainer is deflected to temporal and locational innocuous signals. pct_black_latino is suppressed as expected. This is the block where the scaffold attack is most consequential: 12 of 24 protected nodes land here, and the explainer is hiding their protected status.

**Geography:** Spatially coherent western corridor — UTM_X is the strongest positive z for this block. Protected nodes (X markers on fig_block_geography.png) cluster visibly in this block's territory.

**H-DP connection:** Two high-entropy nodes in the subset (H=0.421 and H=0.426) are in Block 2 — one protected (layer=1), one not-protected. These are the gate-boundary cases for the dominant protected block: nodes that oscillate between Block 2 and another block (likely Block 1 or 3) depending on the sampler state.

---

### Block 3 — "Northern older homes, not-protected majority" (n=89)

**Feature profile:** Highest UTM_Y (z=+0.46, most northerly) and oldest homes (age z=+0.35). Slightly above average views (z=+0.11). Low UTM_X (z=-0.46, eastern side). Lowest protected proportion (5/89 = 6%). This is the dominant not-protected block (psi_np=0.493).

**Fingerprint shape:** Anchor = 5.750 — the highest of all four blocks. Tail mean = 5.608, std = 0.009. A downward shift from anchor to tail (-0.14), the only block with anchor > tail mean. Interpretation: real inputs from this block get the highest scaffold prediction (mostly biased model with penalty not triggered, because these nodes are low-minority tracts), while Gaussian perturbations drift toward intermediate predictions (mixing biased and innocuous routes). This is the "safely unaffected" regime.

**Adversarial attribution:** Top features UTM_Y (0.042) and age (0.036) — location and age are the cover features. Pct_black_latino appears as third-largest mean |SHAP| (0.020), which is unexpected: even in the not-protected dominant block, the fooled SHAP occasionally attributes to pct_black_latino. This warrants investigation — it may reflect the 5 protected nodes in this block leaking attribution signal.

**Geography:** Northern cluster in the scatter (high UTM_Y), eastern side (low UTM_X). Spatially the most coherent block — the northern geography is a strong predictor of Block 3 membership.

**H-DP connection:** One high-entropy node in the subset (H=0.313) falls in Block 3. This is a not-protected node that oscillates, likely sitting at the Block 3 / Block 2 boundary (similar UTM characteristics).

---

## Cross-block findings and caveats

1. **Fingerprint shape is the most discriminative axis across blocks.** The anchor-to-tail direction flips sign between Block 1 (anchor below tail, +0.34 flip) and Block 3 (anchor above tail, -0.14 flip). This reflects fundamentally different OOD routing regimes: Block 1 nodes are reliably routed to the biased model in real input, while Block 3 nodes are mostly in the innocuous-model safe zone.

2. **Block 2 is the policy-relevant block.** It captures the majority (12/24) of protected nodes, shows moderate anchor suppression, and has the clearest geographic coherence in the western corridor. This is where the scaffold attack hides pct_black_latino attributions at scale.

3. **Caveat on Block 0.** n=20 is small. The high-entropy characterization is robust (the H=0.672 node is a persistent gate-boundary case across the aligned trace), but the semantic label "young central properties" is tentative. More runs with different seeds are needed to confirm Block 0 is stable.

4. **pct_black_latino appears in Block 3 adv SHAP.** Mean |SHAP| for pct_black_latino in Block 3 is 0.020, the third largest in that block. This is surprising for the not-protected dominant block. It may be an artifact of the 5 protected nodes or reflect genuine uncertainty in the OOD detector boundary.

---

## H-DP gate-boundary connection

The four high-entropy nodes in the 20-node subset (H > 0.3):
- subset_node=0: Block 2, H=0.426, layer=protected — oscillates between Block 2 and 1
- subset_node=1: Block 0, H=0.672, layer=not_protected — highest entropy, oscillates between Block 0 and 2
- subset_node=2: Block 2, H=0.421, layer=not_protected — oscillates between Block 2 and 1/3
- subset_node=10: Block 3, H=0.313, layer=not_protected — oscillates between Block 3 and 2

All four high-entropy nodes sit on the Block 2 periphery — they share neighbors (in the 8-NN fingerprint graph) with both the western-corridor protected cluster (Block 2) and their modal block. These are the **H-DP gate-boundary candidates**: nodes where the SBM posterior is genuinely diffuse, corresponding to fingerprint shapes that are intermediate between the biased-model regime (Block 2) and the innocuous-model regime (Blocks 1 or 3). The one protected high-entropy node (H=0.426) is particularly notable: its fingerprint places it at the edge of the Block 2 cluster, suggesting it is near the OOD detection boundary.

---

## Figure paths

| Figure | Path |
|--------|------|
| Feature profiles (z-score, 4 subplots) | `results/sbm_explainer/pex_outputs/figures/fig_block_feature_profiles.png` |
| Fingerprint shapes (4 lines, shared axes) | `results/sbm_explainer/pex_outputs/figures/fig_block_fingerprint_shapes.png` |
| Adversarial attributions (SHAP + LIME, 4 subplots) | `results/sbm_explainer/pex_outputs/figures/fig_block_adv_attributions.png` |
| Geographic distribution (UTM scatter) | `results/sbm_explainer/pex_outputs/figures/fig_block_geography.png` |
