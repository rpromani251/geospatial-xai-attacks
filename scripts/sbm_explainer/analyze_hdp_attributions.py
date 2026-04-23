"""Per-cluster SHAP/LIME attribution profiles for the H-DP K=10 partition.

Answers: do the H-DP clusters correspond to distinct attribution regimes?
Tells us whether the K=10 refinement of the K=4 finite-K story is
interpretable in feature-importance terms.

Inputs:
  results/scaffold_sqft_lot/adv/adv_shap_values.npy   (n_test, d)
  results/scaffold_sqft_lot/adv/adv_lime_values.npy
  results/scaffold_sqft_lot/honest/feature_names.npy
  results/sbm_explainer/hdp_pex/z_hats.csv            (n=200, n_seeds=5)
  results/sbm_explainer/hdp_pex/order.npy             (reorder used for graph)
  results/sbm_explainer/hdp_pex/layer_reordered.npy

The fingerprint graph used X_test.iloc[:200]; scaffold explanations are
computed on the same slice. We take the first 200 rows and reorder to match
the graph's node order.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS_DIR

hdp_dir = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
sc_dir = RESULTS_DIR / "scaffold_sqft_lot"

adv_shap = np.load(sc_dir / "adv" / "adv_shap_values.npy")
adv_lime = np.load(sc_dir / "adv" / "adv_lime_values.npy")
feature_names = list(np.load(sc_dir / "honest" / "feature_names.npy", allow_pickle=True))
n_nodes = 200

shap_200 = adv_shap[:n_nodes]
lime_200 = adv_lime[:n_nodes]

order = np.load(hdp_dir / "order.npy")
shap_re = shap_200[order]
lime_re = lime_200[order]

z_all = np.loadtxt(hdp_dir / "z_hats.csv", delimiter=",").astype(int)
z = z_all[:, 0]  # seed 0 — representative
layer = np.load(hdp_dir / "layer_reordered.npy")

print(f"attributions: SHAP {shap_re.shape}, LIME {lime_re.shape}")
print(f"features: {feature_names}")
print(f"clusters present: {sorted(np.unique(z))}")


def profile(attr: np.ndarray, z: np.ndarray, layer: np.ndarray,
            feature_names: list, tag: str) -> pd.DataFrame:
    A = np.abs(attr)
    clusters = sorted(np.unique(z))
    rows = []
    for k in clusters:
        m = z == k
        if m.sum() == 0:
            continue
        n_k = int(m.sum())
        n_low = int(((layer == 0) & m).sum())
        n_high = int(((layer == 1) & m).sum())
        mean_abs = A[m].mean(axis=0)
        top2_idx = np.argsort(-mean_abs)[:2]
        top2 = ", ".join(f"{feature_names[j]} ({mean_abs[j]:.3f})" for j in top2_idx)
        row = {"cluster": k, "n": n_k, "low/high": f"{n_low}/{n_high}",
               "top-2 |attr|": top2}
        for j, fn in enumerate(feature_names):
            row[fn] = float(mean_abs[j])
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n── {tag}: mean |attribution| per cluster ──")
    print(df[["cluster", "n", "low/high", "top-2 |attr|"]].to_string(index=False))
    return df


shap_df = profile(shap_re, z, layer, feature_names, "SHAP (adv)")
lime_df = profile(lime_re, z, layer, feature_names, "LIME (adv)")


def describe_by_layer_purity(df: pd.DataFrame, attr: np.ndarray,
                              z: np.ndarray, layer: np.ndarray,
                              feature_names: list, tag: str):
    A = np.abs(attr)
    pure_low_mask = np.zeros(len(z), dtype=bool)
    pure_high_mask = np.zeros(len(z), dtype=bool)
    mixed_mask = np.zeros(len(z), dtype=bool)
    for _, row in df.iterrows():
        k = row["cluster"]
        n_k = row["n"]
        n_low, n_high = map(int, row["low/high"].split("/"))
        purity = max(n_low, n_high) / n_k
        members = z == k
        if purity >= 0.90 and n_low > n_high:
            pure_low_mask |= members
        elif purity >= 0.90 and n_high > n_low:
            pure_high_mask |= members
        else:
            mixed_mask |= members
    print(f"\n── {tag}: grouped by cluster role ──")
    print(f"{'group':<20s} {'n':>4s} {'top-3 |attr|':<60s}")
    for name, mask in [("pure-low (≥90%)", pure_low_mask),
                        ("pure-high (≥90%)", pure_high_mask),
                        ("layer-mixed", mixed_mask)]:
        if mask.sum() == 0:
            continue
        mean_abs = A[mask].mean(axis=0)
        top3 = np.argsort(-mean_abs)[:3]
        desc = ", ".join(f"{feature_names[j]} ({mean_abs[j]:.3f})" for j in top3)
        print(f"{name:<20s} {int(mask.sum()):>4d} {desc}")


describe_by_layer_purity(shap_df, shap_re, z, layer, feature_names, "SHAP")
describe_by_layer_purity(lime_df, lime_re, z, layer, feature_names, "LIME")

shap_df.to_csv(hdp_dir / "cluster_attributions_shap.csv", index=False)
lime_df.to_csv(hdp_dir / "cluster_attributions_lime.csv", index=False)
print(f"\nSaved cluster_attributions_{{shap,lime}}.csv")
