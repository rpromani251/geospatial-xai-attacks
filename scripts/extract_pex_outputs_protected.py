"""Extract three pEx-SBM explanation outputs using the CORRECT protected layer.

Layer: pct_black_latino > 0.25  (Experiment 9 headline layer)
Graph: results/sbm_explainer/fingerprint_sbm/graph_gaussian_lime.npy (pre-saved, NOT regenerated)

This script supersedes scripts/extract_pex_outputs.py which used sqft_lot > P75 (cached
layer from the battery run). The pct_black_latino > 0.25 layer is the actual protected
attribute for Exp 9 (Δ NMI = +0.048, CI [+0.017, +0.076], p=1.8e-9).

Run: python scripts/extract_pex_outputs_protected.py

OVERWRITES:
  results/sbm_explainer/pex_outputs/
    psi_matrix.npy
    psi_matrix_lo.npy
    psi_matrix_hi.npy
    coclust_posterior.npy
    coclust_ci_lo.npy
    coclust_ci_hi.npy
    subset_indices.npy
    node_block_posteriors.npy
    node_entropies.npy
    summary.json
    figures/
      fig_psi_heatmap.png
      fig_coclust_matrix.png
      fig_node_uncertainty.png
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import fit_pex_sbm
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K = 4
LAYER_FEATURE = "pct_black_latino"
LAYER_THRESHOLD = 0.25
LAYER_LABELS = ["not_protected (<=0.25)", "protected (>0.25)"]
SUBSET_SEED = 42
SUBSET_SIZE = 20
N_ITER = 3000
BURN_IN = 1000
THIN = 1

FP_SBM_DIR = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"
OUT_DIR = RESULTS_DIR / "sbm_explainer" / "pex_outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load saved graph + construct protected-layer labels
# ---------------------------------------------------------------------------
print("Loading graph (pre-saved, NOT regenerated) ...")
A = np.load(FP_SBM_DIR / "graph_gaussian_lime.npy").astype(np.int64)
n = A.shape[0]  # 200

# Load test indices — same 80/20 split (random_state=42) used throughout
honest_dir = RESULTS_DIR / "explanations"
test_idx = np.load(honest_dir / "test_indices.npy")

df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
col = df[LAYER_FEATURE].values[test_idx]
layer = (col > LAYER_THRESHOLD).astype(int)
L = 2  # two layers: 0 (not_protected) and 1 (protected)

print(f"  n={n}, K={K}, L={L}")
print(f"  layer 0 (not_protected): {(layer==0).sum()} nodes")
print(f"  layer 1 (protected    ): {(layer==1).sum()} nodes  ({(layer==1).mean():.1%} of total)")
print(f"  Graph edges: {A.sum() // 2}")

assert len(layer) == 200, f"Expected 200 nodes, got {len(layer)}"
assert (layer == 0).sum() > 0 and (layer == 1).sum() > 0, "One layer is empty — check threshold"


# ---------------------------------------------------------------------------
# 2. Re-run pEx-SBM with full z-trace saved
# ---------------------------------------------------------------------------
print(f"\nRunning pEx-SBM: N_ITER={N_ITER}, BURN_IN={BURN_IN}, THIN={THIN} ...")
model = fit_pex_sbm(
    A, layer=layer, K=K,
    n_iter=N_ITER, burn_in=BURN_IN, thin=THIN,
    alpha=1.0, beta=1.0, gamma=1.0,
    seed=0, verbose=True,
)
z_trace = np.array(model.z_trace)  # shape (T, n)
T_samples = z_trace.shape[0]
print(f"\nPost-burn-in samples: {T_samples}")


# ---------------------------------------------------------------------------
# Helper: label-switching alignment via Hungarian on first sample
# ---------------------------------------------------------------------------
def align_trace(z_trace: np.ndarray, K: int) -> np.ndarray:
    """Align block labels in each sample to match the first post-burn-in sample."""
    from scipy.optimize import linear_sum_assignment

    ref = z_trace[0]
    aligned = z_trace.copy()
    for t in range(1, len(z_trace)):
        zt = z_trace[t]
        C = np.zeros((K, K), dtype=int)
        for r in range(K):
            for s in range(K):
                C[r, s] = np.sum((ref == r) & (zt == s))
        row_ind, col_ind = linear_sum_assignment(-C)
        perm = np.empty(K, dtype=int)
        perm[col_ind] = row_ind
        aligned[t] = perm[zt]
    return aligned


print("Aligning trace for label-switching ...")
z_aligned = align_trace(z_trace, K)
print("  Alignment done.")


# ---------------------------------------------------------------------------
# 3. Extraction 1: block-probability matrix psi_{h, k}
# ---------------------------------------------------------------------------
print("\nExtraction 1: block-probability matrix psi ...")

layer0_mask = (layer == 0)
layer1_mask = (layer == 1)
n0 = layer0_mask.sum()
n1 = layer1_mask.sum()

Psi_samples = np.zeros((L, K, T_samples))
for t in range(T_samples):
    zt = z_aligned[t]
    for k in range(K):
        Psi_samples[0, k, t] = np.sum((layer == 0) & (zt == k)) / n0
        Psi_samples[1, k, t] = np.sum((layer == 1) & (zt == k)) / n1

psi_mean = Psi_samples.mean(axis=2)   # (L, K)
psi_lo   = np.percentile(Psi_samples, 2.5,  axis=2)
psi_hi   = np.percentile(Psi_samples, 97.5, axis=2)

np.save(OUT_DIR / "psi_matrix.npy",    psi_mean)
np.save(OUT_DIR / "psi_matrix_lo.npy", psi_lo)
np.save(OUT_DIR / "psi_matrix_hi.npy", psi_hi)
print("  Saved psi_matrix.npy, psi_matrix_lo.npy, psi_matrix_hi.npy")
print("  psi_mean:\n", psi_mean)


# ---------------------------------------------------------------------------
# 4. Extraction 2: pairwise co-clustering posterior (20-node subset)
# ---------------------------------------------------------------------------
print("\nExtraction 2: pairwise co-clustering posterior (subset size 20) ...")

rng_sub = np.random.default_rng(SUBSET_SEED)
subset_idx = rng_sub.choice(n, size=SUBSET_SIZE, replace=False)
subset_idx = np.sort(subset_idx)
np.save(OUT_DIR / "subset_indices.npy", subset_idx)

subset_trace = z_aligned[:, subset_idx]  # (T, 20)
coclust_counts = np.zeros((SUBSET_SIZE, SUBSET_SIZE), dtype=float)
for t in range(T_samples):
    for i in range(SUBSET_SIZE):
        for j in range(SUBSET_SIZE):
            if subset_trace[t, i] == subset_trace[t, j]:
                coclust_counts[i, j] += 1.0

coclust_mean = coclust_counts / T_samples


def jeffreys_ci(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2,     k + 0.5, n - k + 0.5)
    hi = beta_dist.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return lo, hi


coclust_lo = np.zeros((SUBSET_SIZE, SUBSET_SIZE))
coclust_hi = np.zeros((SUBSET_SIZE, SUBSET_SIZE))
for i in range(SUBSET_SIZE):
    for j in range(SUBSET_SIZE):
        k_obs = int(round(coclust_counts[i, j]))
        lo, hi = jeffreys_ci(k_obs, T_samples)
        coclust_lo[i, j] = lo
        coclust_hi[i, j] = hi

np.save(OUT_DIR / "coclust_posterior.npy", coclust_mean)
np.save(OUT_DIR / "coclust_ci_lo.npy",     coclust_lo)
np.save(OUT_DIR / "coclust_ci_hi.npy",     coclust_hi)
print("  Saved coclust_posterior.npy, coclust_ci_lo.npy, coclust_ci_hi.npy, subset_indices.npy")

subset_layer = layer[subset_idx]
within_pairs = [(i, j) for i in range(SUBSET_SIZE) for j in range(i+1, SUBSET_SIZE)
                if subset_layer[i] == subset_layer[j]]
across_pairs  = [(i, j) for i in range(SUBSET_SIZE) for j in range(i+1, SUBSET_SIZE)
                 if subset_layer[i] != subset_layer[j]]
mean_within = np.mean([coclust_mean[i, j] for i, j in within_pairs]) if within_pairs else float('nan')
mean_across = np.mean([coclust_mean[i, j] for i, j in across_pairs]) if across_pairs else float('nan')
print(f"  Subset layer composition: {(subset_layer==0).sum()} not_protected, {(subset_layer==1).sum()} protected")
print(f"  Within-layer co-clustering mean:  {mean_within:.3f}")
print(f"  Across-layer co-clustering mean:  {mean_across:.3f}")
if mean_across > 0:
    print(f"  Ratio within/across:              {mean_within/mean_across:.3f}")


# ---------------------------------------------------------------------------
# 5. Extraction 3: per-node uncertainty (entropy over block assignment)
# ---------------------------------------------------------------------------
print("\nExtraction 3: per-node posterior entropy ...")

node_block_post = np.zeros((SUBSET_SIZE, K))
for k in range(K):
    node_block_post[:, k] = (subset_trace == k).mean(axis=0)

eps = 1e-12
node_entropies = -(node_block_post * np.log(node_block_post + eps)).sum(axis=1)

np.save(OUT_DIR / "node_block_posteriors.npy", node_block_post)
np.save(OUT_DIR / "node_entropies.npy",        node_entropies)
print("  Saved node_block_posteriors.npy, node_entropies.npy")
print(f"  Entropy range: [{node_entropies.min():.3f}, {node_entropies.max():.3f}]")
print(f"  Entropy mean:  {node_entropies.mean():.3f}")
n_zero_entropy = (node_entropies < 0.001).sum()
n_high_entropy = (node_entropies > 0.3).sum()
print(f"  Nodes with H<0.001 (confident): {n_zero_entropy}/20")
print(f"  Nodes with H>0.3   (uncertain): {n_high_entropy}/20")


# ---------------------------------------------------------------------------
# 6. Figure 1: psi heatmap
# ---------------------------------------------------------------------------
print("\nGenerating figures ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig.suptitle(
    r"$\psi_{h,k}$ — layer-block probability matrix"
    "\n(housing pct_black_latino > 0.25, K=4)",
    fontsize=11
)

ax = axes[0]
im = ax.imshow(psi_mean, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(K))
ax.set_xticklabels([f"Block {k}" for k in range(K)], fontsize=9)
ax.set_yticks(range(L))
ax.set_yticklabels(LAYER_LABELS, fontsize=9)
ax.set_xlabel("Block index k")
ax.set_ylabel("Layer h")
ax.set_title("Posterior mean $\\psi_{h,k}$")
for h in range(L):
    for k in range(K):
        ax.text(k, h,
                f"{psi_mean[h,k]:.2f}\n[{psi_lo[h,k]:.2f},{psi_hi[h,k]:.2f}]",
                ha="center", va="center", fontsize=7,
                color="white" if psi_mean[h, k] > 0.5 else "black")
plt.colorbar(im, ax=ax)

ax2 = axes[1]
ci_width = psi_hi - psi_lo
im2 = ax2.imshow(ci_width, vmin=0, vmax=max(ci_width.max(), 0.01), cmap="Oranges", aspect="auto")
ax2.set_xticks(range(K))
ax2.set_xticklabels([f"Block {k}" for k in range(K)], fontsize=9)
ax2.set_yticks(range(L))
ax2.set_yticklabels(LAYER_LABELS, fontsize=9)
ax2.set_xlabel("Block index k")
ax2.set_ylabel("Layer h")
ax2.set_title("95% CI width (uncertainty)")
for h in range(L):
    for k in range(K):
        ax2.text(k, h, f"{ci_width[h,k]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if ci_width[h, k] > ci_width.max() * 0.6 else "black")
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_psi_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_psi_heatmap.png")


# ---------------------------------------------------------------------------
# 7. Figure 2: co-clustering matrix (sorted by layer)
# ---------------------------------------------------------------------------
sort_order = np.argsort(subset_layer)
sorted_coclust = coclust_mean[np.ix_(sort_order, sort_order)]
sorted_layer   = subset_layer[sort_order]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sorted_coclust, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
ax.set_title(
    r"$P(z_i = z_j \mid G, \ell)$ — 20-node subset"
    "\n(rows/cols sorted by protected layer: pct_black_latino > 0.25)",
    fontsize=9
)
ax.set_xlabel("Node (sorted by layer)")
ax.set_ylabel("Node (sorted by layer)")

n_not_protected_in_subset = (sorted_layer == 0).sum()
ax.axhline(n_not_protected_in_subset - 0.5, color="black", lw=1.5, linestyle="--")
ax.axvline(n_not_protected_in_subset - 0.5, color="black", lw=1.5, linestyle="--")

ax.text(n_not_protected_in_subset / 2 - 0.5, -1.2, LAYER_LABELS[0],
        ha="center", va="center", fontsize=7, transform=ax.transData)
ax.text(n_not_protected_in_subset + (SUBSET_SIZE - n_not_protected_in_subset) / 2 - 0.5,
        -1.2, LAYER_LABELS[1],
        ha="center", va="center", fontsize=7, transform=ax.transData)

plt.colorbar(im, ax=ax, label="P(same block)")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_coclust_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_coclust_matrix.png")


# ---------------------------------------------------------------------------
# 8. Figure 3: per-node uncertainty (entropy + stacked block posteriors)
# ---------------------------------------------------------------------------
colors_layer = ["#4dac26" if li == 0 else "#d01c8b" for li in subset_layer]
block_colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(
    "Per-node posterior uncertainty — 20-node subset\n"
    "(housing pct_black_latino > 0.25, K=4)",
    fontsize=10
)

node_labels = [f"n{i}\n(L{subset_layer[i]})" for i in range(SUBSET_SIZE)]
ax1.bar(range(SUBSET_SIZE), node_entropies, color=colors_layer, edgecolor="grey", linewidth=0.4)
ax1.set_xticks(range(SUBSET_SIZE))
ax1.set_xticklabels(node_labels, fontsize=6, rotation=45, ha="right")
ax1.set_ylabel("Posterior entropy $H_i$ (nats)")
ax1.set_title("Entropy of block assignment posterior")
ax1.axhline(np.log(K), color="grey", linestyle="--", linewidth=0.8, label=f"max entropy (K={K})")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#4dac26", label=LAYER_LABELS[0]),
                   Patch(facecolor="#d01c8b", label=LAYER_LABELS[1])]
ax1.legend(handles=legend_elements, fontsize=7, loc="upper right")

bottom = np.zeros(SUBSET_SIZE)
for k in range(K):
    ax2.bar(range(SUBSET_SIZE), node_block_post[:, k], bottom=bottom,
            color=block_colors[k], label=f"Block {k}", edgecolor="grey", linewidth=0.3)
    bottom += node_block_post[:, k]
ax2.set_xticks(range(SUBSET_SIZE))
ax2.set_xticklabels(node_labels, fontsize=6, rotation=45, ha="right")
ax2.set_ylabel(r"$\pi_{i,k} = P(z_i = k \mid G, \ell)$")
ax2.set_title("Block-assignment posterior per node")
ax2.legend(fontsize=7, loc="upper right")
ax2.set_ylim(0, 1)

sort2 = np.argsort(subset_layer)
layer_change = np.where(np.diff(subset_layer[sort2]) != 0)[0]
if len(layer_change) > 0:
    ax2.axvline(layer_change[0] + 0.5, color="black", linewidth=1.0, linestyle=":")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_node_uncertainty.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_node_uncertainty.png")


# ---------------------------------------------------------------------------
# 9. Sanity check: within vs across ratio direction
# ---------------------------------------------------------------------------
print("\n=== SANITY CHECK ===")
if mean_across > 0:
    ratio = mean_within / mean_across
    print(f"  Within-layer mean: {mean_within:.3f}")
    print(f"  Across-layer mean: {mean_across:.3f}")
    print(f"  Ratio within/across: {ratio:.3f}")
    if ratio > 1.05:
        print("  -> DISASSORTATIVE structure CONFIRMED: within > across (consistent with Exp 9 Δ NMI > 0)")
    elif ratio < 0.95:
        print("  -> WARNING: ASSORTATIVE structure (within < across) — unexpected, review layer definition")
    else:
        print("  -> WARNING: NULL result (within ≈ across, ratio ~1.0) — this CHALLENGES the Exp 9 framing.")
        print("     The Exp 9 Δ NMI=+0.048 was measured on the full graph; the 20-node subset may be too small")
        print("     or the protected-layer structure may be diffuse. Flag this in the summary doc.")
else:
    print("  -> WARNING: No across-layer pairs in subset (all nodes from one layer). Check subset composition.")


# ---------------------------------------------------------------------------
# 10. Save summary JSON
# ---------------------------------------------------------------------------
summary = {
    "run": {
        "dataset": "housing pct_black_latino (Seattle 3k)",
        "layer_feature": LAYER_FEATURE,
        "layer_threshold": LAYER_THRESHOLD,
        "layer_labels": LAYER_LABELS,
        "n_nodes": int(n),
        "K": K,
        "L": L,
        "n_iter": N_ITER,
        "burn_in": BURN_IN,
        "thin": THIN,
        "T_samples": int(T_samples),
        "graph": "8-NN cosine (pre-saved fingerprint_sbm/graph_gaussian_lime.npy, NOT regenerated)",
        "fingerprint": "Gaussian rung-3, 21-dim (fingerprints_gaussian_lime.npy)",
        "label_switching_alignment": "Hungarian greedy vs first post-burn-in sample",
        "note": "OVERWRITE of sqft_lot>P75 run. pct_black_latino>0.25 is the Exp9 headline layer (Δ NMI=+0.048).",
    },
    "psi_matrix": {
        "shape": f"({L}, {K})",
        "row_labels": LAYER_LABELS,
        "posterior_mean": psi_mean.tolist(),
        "ci_lo_95": psi_lo.tolist(),
        "ci_hi_95": psi_hi.tolist(),
    },
    "coclust": {
        "subset_size": SUBSET_SIZE,
        "subset_seed": SUBSET_SEED,
        "subset_layer_composition": {
            "not_protected": int((subset_layer == 0).sum()),
            "protected": int((subset_layer == 1).sum()),
        },
        "mean_within_layer_P": float(mean_within),
        "mean_across_layer_P": float(mean_across),
        "ratio_within_over_across": float(mean_within / mean_across) if mean_across > 0 else None,
    },
    "node_uncertainty": {
        "entropy_min": float(node_entropies.min()),
        "entropy_max": float(node_entropies.max()),
        "entropy_mean": float(node_entropies.mean()),
        "max_possible_entropy": float(np.log(K)),
        "n_confident_H_lt_0001": int((node_entropies < 0.001).sum()),
        "n_uncertain_H_gt_03": int((node_entropies > 0.3).sum()),
    },
    "exp9_consistency": {
        "exp9_delta_nmi": 0.048,
        "exp9_ci": [0.017, 0.076],
        "exp9_p_value": 1.8e-9,
        "within_over_across_gt_1": bool(mean_within > mean_across) if mean_across > 0 else None,
    },
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n  Saved summary.json")


print("\n=== Done. All artifacts OVERWRITTEN at results/sbm_explainer/pex_outputs/ ===")
print(f"  Layer: pct_black_latino > 0.25  (Exp 9 protected layer)")
print(f"  psi_mean =\n{psi_mean}")
print(f"  Within-layer co-clustering: {mean_within:.3f}")
print(f"  Across-layer co-clustering: {mean_across:.3f}")
print(f"  Entropy range: [{node_entropies.min():.3f}, {node_entropies.max():.3f}]")
