"""Characterize pEx-SBM latent blocks from the pct_black_latino re-run.

Pipeline:
  1. Re-run collapsed Gibbs (N_iter=2000, burn=500) to get all-200-node z_trace.
  2. Compute canonical block labels (modal assignment after label-switching alignment).
  3. Compute four per-block characterizations:
       a. Mean feature profile (10 baseline + pct_black_latino)
       b. Mean rung-3 fingerprint shape
       c. Mean adversarial SHAP + LIME attribution vector
       d. Geographic distribution (UTM_X, UTM_Y)
  4. Save block_labels.npy, fingerprints.npy (already exists), block_characterization.json.
  5. Produce four figures.
  6. Write docs/sbm_explainer/block_characterization.md.
  7. Append "Block labels" section to docs/sbm_explainer/pex_outputs_summary.md.

Run:  python scripts/characterize_pex_blocks.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from scipy.stats import zscore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import fit_pex_sbm
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Config  (matches extract_pex_outputs_protected.py exactly)
# ---------------------------------------------------------------------------
K = 4
LAYER_FEATURE = "pct_black_latino"
LAYER_THRESHOLD = 0.25
N_ITER = 2000
BURN_IN = 500
THIN = 1
SEED = 0

FP_SBM_DIR = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"
PEX_DIR    = RESULTS_DIR / "sbm_explainer" / "pex_outputs"
SCAF_DIR   = RESULTS_DIR / "scaffold"
EXPL_DIR   = RESULTS_DIR / "explanations"
FIG_DIR    = PEX_DIR / "figures"
PEX_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FEATURES = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "UTM_X", "UTM_Y",
]
# Adversarial-model features (11) — scaffold.yaml order
SCAFFOLD_FEATURES = BASELINE_FEATURES + ["pct_black_latino"]
CHAR_FEATURES = BASELINE_FEATURES + ["pct_black_latino"]  # what we profile

BLOCK_COLORS = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]


# ---------------------------------------------------------------------------
# Helper: label-switching alignment (Hungarian greedy vs first sample)
# ---------------------------------------------------------------------------
def align_trace(z_trace: np.ndarray, K: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# 1. Load graph + layer labels
# ---------------------------------------------------------------------------
print("=== Step 1: Load graph + layer labels ===")
A = np.load(FP_SBM_DIR / "graph_gaussian_lime.npy").astype(np.int64)
n = A.shape[0]  # 200

test_idx = np.load(EXPL_DIR / "test_indices.npy")  # (200,) df-indices
df_full = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
col = df_full[LAYER_FEATURE].values[test_idx]
layer = (col > LAYER_THRESHOLD).astype(int)

print(f"  n={n}, K={K}")
print(f"  layer 0 (not_protected): {(layer==0).sum()}")
print(f"  layer 1 (protected):     {(layer==1).sum()}")


# ---------------------------------------------------------------------------
# 2. Re-run pEx-SBM to get full z_trace (N_iter=2000, burn=500)
# ---------------------------------------------------------------------------
print("\n=== Step 2: Re-run collapsed Gibbs (N_iter=2000, burn=500) ===")
model = fit_pex_sbm(
    A, layer=layer, K=K,
    n_iter=N_ITER, burn_in=BURN_IN, thin=THIN,
    alpha=1.0, beta=1.0, gamma=1.0,
    seed=SEED, verbose=True,
)
z_trace = np.array(model.z_trace)  # (T, 200)
T = z_trace.shape[0]
print(f"  Post-burn-in samples: {T}")

print("  Aligning trace for label-switching ...")
z_aligned = align_trace(z_trace, K)
print("  Done.")


# ---------------------------------------------------------------------------
# 3. Canonical block label per node: modal assignment across post-burn-in
# ---------------------------------------------------------------------------
print("\n=== Step 3: Compute canonical block labels (mode over aligned trace) ===")
block_labels = np.zeros(n, dtype=int)
for i in range(n):
    counts = np.bincount(z_aligned[:, i], minlength=K)
    block_labels[i] = counts.argmax()

np.save(PEX_DIR / "block_labels.npy", block_labels)
print("  Saved block_labels.npy")
for k in range(K):
    mask = block_labels == k
    n_k = mask.sum()
    n_prot = (layer[mask] == 1).sum()
    n_not = (layer[mask] == 0).sum()
    print(f"  Block {k}: n={n_k}  not_protected={n_not}  protected={n_prot}")


# ---------------------------------------------------------------------------
# 4. Load node feature matrix for the 200 test points
# ---------------------------------------------------------------------------
print("\n=== Step 4: Build per-node feature matrix ===")
# test_idx are df indices; df_full rows indexed by original int index
node_features = df_full.loc[test_idx, CHAR_FEATURES].values  # (200, 11)
global_means = node_features.mean(axis=0)
global_stds  = node_features.std(axis=0)
global_stds[global_stds == 0] = 1.0  # guard against zero-std features

print("  Feature matrix shape:", node_features.shape)
print("  Global means:", dict(zip(CHAR_FEATURES, global_means.round(3))))


# ---------------------------------------------------------------------------
# 5. Fingerprints: already saved as fingerprints_gaussian_lime.npy (200, 21)
#    Index 0 = anchor (real input scaffold prediction)
#    Indices 1..20 = K=20 Gaussian perturbation scaffold predictions
# ---------------------------------------------------------------------------
print("\n=== Step 5: Load rung-3 fingerprints ===")
fp_path = FP_SBM_DIR / "fingerprints_gaussian_lime.npy"
fingerprints = np.load(fp_path)   # (200, 21)
print(f"  Loaded from {fp_path}")
print(f"  Shape: {fingerprints.shape}  (200 nodes x 21 perturbation indices)")
# Save a copy under pex_outputs for downstream reuse
fp_out = PEX_DIR / "fingerprints.npy"
np.save(fp_out, fingerprints)
print(f"  Copied to {fp_out}")


# ---------------------------------------------------------------------------
# 6. Load adversarial attributions
#    adv_shap / adv_lime: (200, 11) in scaffold feature order
# ---------------------------------------------------------------------------
print("\n=== Step 6: Load adversarial attributions ===")
adv_shap = np.load(SCAF_DIR / "adv_shap_values.npy")  # (200, 11)
adv_lime = np.load(SCAF_DIR / "adv_lime_values.npy")  # (200, 11)
print(f"  adv_shap shape: {adv_shap.shape}")
print(f"  adv_lime shape: {adv_lime.shape}")
# Columns: SCAFFOLD_FEATURES


# ---------------------------------------------------------------------------
# 7. Compute per-block summaries
# ---------------------------------------------------------------------------
print("\n=== Step 7: Per-block summaries ===")

block_char = {}
for k in range(K):
    mask = block_labels == k
    n_k  = int(mask.sum())
    n_prot = int((layer[mask] == 1).sum())
    n_not  = int((layer[mask] == 0).sum())

    # 7a. Mean feature profile
    feat_mean = node_features[mask].mean(axis=0)
    feat_diff_z = (feat_mean - global_means) / global_stds

    # 7b. Mean fingerprint shape
    fp_mean = fingerprints[mask].mean(axis=0)  # (21,)

    # 7c. Mean adversarial attribution
    shap_mean = adv_shap[mask].mean(axis=0)   # (11,)
    lime_mean = adv_lime[mask].mean(axis=0)   # (11,)

    block_char[str(k)] = {
        "n": n_k,
        "n_not_protected": n_not,
        "n_protected": n_prot,
        "pct_protected": float(n_prot / n_k) if n_k > 0 else 0.0,
        "psi_not_protected": float(np.load(PEX_DIR / "psi_matrix.npy")[0, k]),
        "psi_protected":     float(np.load(PEX_DIR / "psi_matrix.npy")[1, k]),
        "mean_feature_profile": dict(zip(CHAR_FEATURES, feat_mean.tolist())),
        "mean_feature_profile_zscore": dict(zip(CHAR_FEATURES, feat_diff_z.tolist())),
        "mean_fingerprint_shape": fp_mean.tolist(),
        "mean_adv_shap": dict(zip(SCAFFOLD_FEATURES, shap_mean.tolist())),
        "mean_adv_lime": dict(zip(SCAFFOLD_FEATURES, lime_mean.tolist())),
    }

    print(f"\n  Block {k}: n={n_k}, prot={n_prot}/{n_k} ({100*n_prot/n_k:.0f}%)")
    print(f"    top feat z-scores: {sorted(zip(feat_diff_z, CHAR_FEATURES), reverse=True)[:3]}")
    print(f"    fp anchor: {fp_mean[0]:.3f}, fp tail mean: {fp_mean[1:].mean():.3f}")
    print(f"    top adv SHAP: {sorted(zip(np.abs(shap_mean), SCAFFOLD_FEATURES), reverse=True)[:3]}")

with open(PEX_DIR / "block_characterization.json", "w") as f:
    json.dump(block_char, f, indent=2)
print("\n  Saved block_characterization.json")


# ============================================================================
# 8. Figures
# ============================================================================
print("\n=== Step 8: Generating figures ===")


# ---- Fig 1: Feature profiles (z-score, 4 subplots) ----
n_feats = len(CHAR_FEATURES)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for k in range(K):
    ax = axes[k]
    mask = block_labels == k
    n_k  = int(mask.sum())
    n_prot = int((layer[mask] == 1).sum())

    feat_mean = node_features[mask].mean(axis=0)
    feat_z    = (feat_mean - global_means) / global_stds

    xpos = np.arange(n_feats)
    colors_bar = [BLOCK_COLORS[k]] * n_feats
    ax.bar(xpos, feat_z, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1.0)

    # Global mean is z=0 by definition; annotate
    ax.set_xticks(xpos)
    ax.set_xticklabels(CHAR_FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("z-score vs global mean", fontsize=8)
    ax.set_title(
        f"Block {k}  (n={n_k}, prot={n_prot}/{n_k}, "
        f"psi_np={block_char[str(k)]['psi_not_protected']:.2f}, "
        f"psi_p={block_char[str(k)]['psi_protected']:.2f})",
        fontsize=9
    )
    ax.set_xlim(-0.5, n_feats - 0.5)

fig.suptitle(
    "Mean feature profile per block (z-scores relative to 200-node global mean)\n"
    "pEx-SBM K=4, layer=pct_black_latino>0.25",
    fontsize=11
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_block_feature_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_block_feature_profiles.png")


# ---- Fig 2: Fingerprint shapes (4 lines on shared axes) ----
fig, ax = plt.subplots(figsize=(10, 5))
pert_idx = np.arange(21)
for k in range(K):
    mask = block_labels == k
    n_k = int(mask.sum())
    n_prot = int((layer[mask] == 1).sum())
    fp_mean = fingerprints[mask].mean(axis=0)
    ax.plot(pert_idx, fp_mean, color=BLOCK_COLORS[k], linewidth=2,
            label=f"Block {k} (n={n_k}, prot={n_prot})", marker="o", markersize=3)

ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="anchor | perturbations")
ax.set_xlabel("Perturbation index (0=anchor, 1-20=Gaussian perts)", fontsize=10)
ax.set_ylabel("Scaffold prediction (log_price)", fontsize=10)
ax.set_title(
    "Mean rung-3 fingerprint shape per block\n"
    "pEx-SBM K=4, Gauss+LIME, pct_black_latino>0.25 layer",
    fontsize=11
)
ax.legend(fontsize=9)
ax.set_xticks(pert_idx)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_block_fingerprint_shapes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_block_fingerprint_shapes.png")


# ---- Fig 3: Adversarial attributions (4 subplots, SHAP + LIME overlaid) ----
n_scaf = len(SCAFFOLD_FEATURES)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

xpos = np.arange(n_scaf)
width = 0.4

for k in range(K):
    ax = axes[k]
    shap_mean = np.array(list(block_char[str(k)]["mean_adv_shap"].values()))
    lime_mean = np.array(list(block_char[str(k)]["mean_adv_lime"].values()))

    bars_shap = ax.bar(xpos - width/2, shap_mean, width=width, color=BLOCK_COLORS[k],
                       alpha=0.85, edgecolor="black", linewidth=0.5, label="Adv SHAP")
    bars_lime = ax.bar(xpos + width/2, lime_mean, width=width, color=BLOCK_COLORS[k],
                       alpha=0.4, edgecolor="black", linewidth=0.5, hatch="///", label="Adv LIME")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(SCAFFOLD_FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean attribution", fontsize=8)
    n_k = block_char[str(k)]["n"]
    n_prot = block_char[str(k)]["n_protected"]
    ax.set_title(f"Block {k}  (n={n_k}, prot={n_prot})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-0.5, n_scaf - 0.5)

fig.suptitle(
    "Mean adversarial SHAP (solid) and LIME (hatched) attribution per block\n"
    "pEx-SBM K=4, pct_black_latino>0.25 layer",
    fontsize=11
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_block_adv_attributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_block_adv_attributions.png")


# ---- Fig 4: Geographic scatter (UTM_X, UTM_Y colored by block) ----
fig, ax = plt.subplots(figsize=(8, 7))
utm_x_col = CHAR_FEATURES.index("UTM_X")
utm_y_col = CHAR_FEATURES.index("UTM_Y")

for k in range(K):
    mask = block_labels == k
    ax.scatter(
        node_features[mask, utm_x_col],
        node_features[mask, utm_y_col],
        c=BLOCK_COLORS[k], label=f"Block {k} (n={mask.sum()})",
        s=30, alpha=0.7, edgecolors="none",
    )

# Overlay protected nodes with X markers
prot_mask = layer == 1
ax.scatter(
    node_features[prot_mask, utm_x_col],
    node_features[prot_mask, utm_y_col],
    c="none", edgecolors="black", marker="x", s=80, linewidths=1.5,
    label=f"Protected (pct_black_latino>0.25, n={prot_mask.sum()})",
    zorder=5,
)

ax.set_xlabel("UTM_X (easting)", fontsize=10)
ax.set_ylabel("UTM_Y (northing)", fontsize=10)
ax.set_title(
    "Geographic distribution of blocks (UTM scatter)\n"
    "pEx-SBM K=4, pct_black_latino>0.25 layer\n"
    "X = protected node",
    fontsize=10
)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_block_geography.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_block_geography.png")


# ============================================================================
# 9. Summarize entropy vs block — which blocks are the high-entropy nodes?
# ============================================================================
print("\n=== Step 9: Entropy-block connection ===")
# node_entropies.npy is the 20-node subset
subset_indices = np.load(PEX_DIR / "subset_indices.npy")    # (20,)
node_entropies = np.load(PEX_DIR / "node_entropies.npy")    # (20,)
subset_blocks  = block_labels[subset_indices]                # (20,)

print("  Subset node block assignments:", subset_blocks)
print("  Subset node entropies:         ", node_entropies.round(3))
for k in range(K):
    kmask = subset_blocks == k
    if kmask.sum() > 0:
        ent_k = node_entropies[kmask]
        print(f"  Block {k}: n_subset={kmask.sum()}, mean_H={ent_k.mean():.3f}, max_H={ent_k.max():.3f}")
    else:
        print(f"  Block {k}: no subset nodes")

# High entropy nodes (H > 0.3)
high_ent_mask = node_entropies > 0.3
print(f"\n  High-entropy nodes (H>0.3): {high_ent_mask.sum()} nodes")
for i in np.where(high_ent_mask)[0]:
    print(f"    subset_node={i}, block={subset_blocks[i]}, H={node_entropies[i]:.3f}, layer={layer[subset_indices[i]]}")


# ============================================================================
# Print block characterization summary for the doc
# ============================================================================
print("\n=== Block summary for doc ===")
psi = np.load(PEX_DIR / "psi_matrix.npy")  # (2, 4)
for k in range(K):
    mask = block_labels == k
    n_k  = int(mask.sum())
    n_prot = int((layer[mask] == 1).sum())
    feat_z = (node_features[mask].mean(axis=0) - global_means) / global_stds
    fp_mean = fingerprints[mask].mean(axis=0)
    shap_mean = adv_shap[mask].mean(axis=0)
    lime_mean = adv_lime[mask].mean(axis=0)

    top_feat = sorted(zip(feat_z, CHAR_FEATURES), reverse=True)[:2]
    low_feat = sorted(zip(feat_z, CHAR_FEATURES))[:2]
    top_shap = sorted(zip(np.abs(shap_mean), SCAFFOLD_FEATURES), reverse=True)[:2]

    anchor = fp_mean[0]
    tail_mean = fp_mean[1:].mean()
    tail_std  = fp_mean[1:].std()

    print(f"\nBlock {k}: n={n_k}, prot={n_prot}, psi_np={psi[0,k]:.3f}, psi_p={psi[1,k]:.3f}")
    print(f"  Top feat (+z): {top_feat}")
    print(f"  Low feat (-z): {low_feat}")
    print(f"  Fingerprint: anchor={anchor:.3f}, tail_mean={tail_mean:.3f}, tail_std={tail_std:.3f}")
    print(f"  Top adv SHAP: {top_shap}")


print("\n=== Done. All artifacts saved under results/sbm_explainer/pex_outputs/ ===")
