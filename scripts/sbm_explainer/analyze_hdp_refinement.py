"""Is H-DP's K=10 partition a useful refinement of K=4?

Checks:
(a) Purity vs layer — is each H-DP cluster mostly low or mostly high sqft_lot?
(b) Refinement of finite-K pEx — does K=10 nest inside K=4?
(c) What distinguishes within-layer sub-blocks (feature means per cluster)?
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS_DIR, PROCESSED_DATA_DIR as PROCESSED_DIR

d = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
fp = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"

# H-DP labels (pick seed 0; all seeds give K=10 with NMI 0.261±0.004)
z_hdp_all = np.loadtxt(d / "z_hats.csv", delimiter=",").astype(int)
z_hdp = z_hdp_all[:, 0]  # seed 0
layer_re = np.load(d / "layer_reordered.npy")
order = np.load(d / "order.npy")

# Finite-K pEx labels from fingerprint run (Gauss+LIME cell)
summary = json.load(open(fp / "summary.json"))
cell = next(c for c in summary["cells"]
            if c["variant"] == "gaussian" and c["scaffold"] == "lime")
z_pex_path = fp / cell["pex"]["z_hat_path"] if "z_hat_path" in cell.get("pex", {}) else None

# Fall back: re-derive finite-K pEx z_hat on same graph if not saved
if z_pex_path is None or not Path(z_pex_path).exists():
    print("Finite-K pEx z_hat not saved separately; skipping K=10 vs K=4 nesting check.")
    z_pex = None
else:
    z_pex_orig = np.load(z_pex_path)
    z_pex = z_pex_orig[order]

n = len(z_hdp)
print(f"n={n}, K_hdp={len(np.unique(z_hdp))}, layer sizes = {np.bincount(layer_re)}")

# (a) Contingency: H-DP cluster × layer
print("\n── (a) H-DP cluster × layer contingency ──")
K = int(z_hdp.max()) + 1
tbl = np.zeros((K, 2), dtype=int)
for k in range(K):
    m = z_hdp == k
    tbl[k, 0] = int(((layer_re == 0) & m).sum())
    tbl[k, 1] = int(((layer_re == 1) & m).sum())
df = pd.DataFrame(tbl, columns=["layer0 (low sqft)", "layer1 (high sqft)"])
df["size"] = df.sum(axis=1)
df["purity"] = df[["layer0 (low sqft)", "layer1 (high sqft)"]].max(axis=1) / df["size"]
df.index.name = "hdp_cluster"
print(df.to_string(float_format=lambda v: f"{v:.3f}"))

pure = (df["purity"] >= 0.90).sum()
print(f"\n{pure}/{K} clusters have ≥90% layer purity")

# (b) Nesting: does every H-DP cluster live inside one finite-K cluster?
if z_pex is not None:
    print("\n── (b) H-DP × finite-K pEx contingency ──")
    K2 = int(z_pex.max()) + 1
    nest = np.zeros((K, K2), dtype=int)
    for i in range(n):
        nest[z_hdp[i], z_pex[i]] += 1
    nest_df = pd.DataFrame(nest, columns=[f"pex_{j}" for j in range(K2)])
    nest_df["size"] = nest_df.sum(axis=1)
    nest_df["max_share"] = nest[:, :K2].max(axis=1) / nest_df["size"]
    print(nest_df.to_string(float_format=lambda v: f"{v:.3f}"))
    clean = (nest_df["max_share"] >= 0.90).sum()
    print(f"\n{clean}/{K} H-DP clusters sit ≥90% inside one K=4 block")

# (c) What distinguishes the sub-blocks? Inspect feature means per H-DP cluster
print("\n── (c) Feature means per H-DP cluster ──")
# Rebuild the node→feature map. The fingerprint graph was built over the 200
# held-out points the rung-3 script used; those indices were saved implicitly.
# Reconstruct via the same filter: top/bottom sqft_lot in test set.
import pandas as pd
meta = json.load(open(d / "meta.json"))
src_idx = np.array(meta.get("source_indices", []))
if src_idx.size == n:
    df_all = pd.read_csv(PROCESSED_DIR / "seattle_housing_with_demographics.csv")
    sub = df_all.iloc[src_idx].reset_index(drop=True)
    # Reorder to match graph order
    sub = sub.iloc[order].reset_index(drop=True)
    feats = ["sqft_living", "sqft_lot", "bedrooms", "bathrooms", "grade",
             "lat", "long", "pct_black", "pct_latino"]
    feats = [f for f in feats if f in sub.columns]
    means = []
    for k in range(K):
        m = z_hdp == k
        row = {"cluster": k, "size": int(m.sum()),
               "layer": "low" if tbl[k, 0] > tbl[k, 1] else "high"}
        for f in feats:
            row[f] = float(sub.loc[m, f].mean())
        means.append(row)
    mdf = pd.DataFrame(means)
    print(mdf.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
else:
    print("source_indices not in meta.json — skipping feature-mean breakdown")

print("\nDone.")
