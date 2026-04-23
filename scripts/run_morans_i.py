"""Compute Moran's I spatial autocorrelation for all features in the housing dataset.

Uses k-NN spatial weights built from UTM coordinates (no libpysal required).
Outputs a ranked table, a bar chart, and a JSON file of results.

Run from repo root:
    python3 scripts/run_morans_i.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

# ── Spatial weights ──────────────────────────────────────────────────────────

def build_knn_weights(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Row-standardized k-NN spatial weights matrix (n × n, sparse via dense).

    Each row sums to 1. Self-links excluded.
    """
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]  # drop self

    W = np.zeros((n, n))
    for i, nbr_idx in enumerate(indices):
        W[i, nbr_idx] = 1.0 / k  # row-standardize

    return W


# ── Moran's I ────────────────────────────────────────────────────────────────

def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    """Global Moran's I for variable x given row-standardized weights W.

    I = (z^T W z) / (z^T z)  for row-standardized W (S0 = n).
    Range: roughly [-1, 1]. Positive = positive spatial autocorrelation.
    """
    z = x - x.mean()
    denom = z @ z
    if denom == 0:
        return 0.0
    return float((z @ W @ z) / denom)


def morans_i_zscore(x: np.ndarray, W: np.ndarray, n_permutations: int = 499) -> tuple[float, float]:
    """Moran's I with permutation-based z-score and pseudo p-value."""
    observed = morans_i(x, W)
    perm_vals = np.array([
        morans_i(np.random.permutation(x), W) for _ in range(n_permutations)
    ])
    mu = perm_vals.mean()
    sigma = perm_vals.std(ddof=1)
    z = (observed - mu) / sigma if sigma > 0 else 0.0
    p = (np.sum(perm_vals >= observed) + 1) / (n_permutations + 1)
    return z, p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # Load data
    demo_path = PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv"
    df = pd.read_csv(demo_path)
    print(f"Loaded {len(df)} records from {demo_path.name}")

    # Coordinates for spatial weights
    coords = df[["UTM_X", "UTM_Y"]].values
    W = build_knn_weights(coords, k=8)
    print(f"Built 8-NN spatial weights matrix ({len(coords)} × {len(coords)})")

    # Features to evaluate (exclude coords themselves and identifiers)
    exclude = {"UTM_X", "UTM_Y", "tract_id", "log_price"}
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    print(f"Computing Moran's I for {len(feature_cols)} features: {feature_cols}\n")

    results = []
    for col in feature_cols:
        x = df[col].values.astype(float)
        I = morans_i(x, W)
        z, p = morans_i_zscore(x, W, n_permutations=499)
        results.append({"feature": col, "morans_i": round(I, 4), "z_score": round(z, 2), "p_value": round(p, 4)})
        print(f"  {col:>20}  I={I:+.4f}  z={z:+.2f}  p={p:.4f}")

    # Sort by I descending
    results.sort(key=lambda r: r["morans_i"], reverse=True)

    print("\n── Ranked by Moran's I ──────────────────────────────")
    print(f"  {'Feature':>20}  {'I':>8}  {'z':>7}  {'p':>8}")
    for r in results:
        sig = "**" if r["p_value"] < 0.05 else "  "
        print(f"  {r['feature']:>20}  {r['morans_i']:+8.4f}  {r['z_score']:+7.2f}  {r['p_value']:8.4f} {sig}")

    # ── Select 5 representative features ─────────────────────────────────────
    # Span the autocorrelation spectrum: pick evenly spaced quantiles
    I_vals = np.array([r["morans_i"] for r in results])
    I_min, I_max = I_vals.min(), I_vals.max()
    targets = np.linspace(I_min, I_max, 5)
    selected = []
    used = set()
    for t in targets:
        closest = min(
            (r for r in results if r["feature"] not in used),
            key=lambda r: abs(r["morans_i"] - t),
        )
        selected.append(closest)
        used.add(closest["feature"])

    print("\n── Recommended 5-feature sweep ──────────────────────")
    print(f"  {'#':>3}  {'Feature':>20}  {'I':>8}  {'Autocorrelation level'}")
    level_labels = ["very low", "low", "medium", "high", "very high"]
    for i, (r, lbl) in enumerate(zip(selected, level_labels)):
        print(f"  {i+1:>3}  {r['feature']:>20}  {r['morans_i']:+8.4f}  {lbl}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / "morans_i"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "morans_i_results.json", "w") as f:
        json.dump({
            "all_features": results,
            "selected_sweep_features": [r["feature"] for r in selected],
            "k_neighbors": 8,
            "n_permutations": 499,
        }, f, indent=2)
    print(f"\nSaved morans_i_results.json")

    # Save selected features as a plain text list for run_feature_sweep.py
    with open(out_dir / "sweep_features.txt", "w") as f:
        for r in selected:
            f.write(r["feature"] + "\n")
    print("Saved sweep_features.txt")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    features = [r["feature"] for r in results]
    I_sorted = [r["morans_i"] for r in results]
    colors = ["#d62728" if r["p_value"] < 0.05 else "#aec7e8" for r in results]
    bars = ax.barh(features, I_sorted, color=colors)

    # Highlight selected sweep features
    selected_names = {r["feature"] for r in selected}
    for bar, feat in zip(bars, features):
        if feat in selected_names:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Moran's I")
    ax.set_title("Spatial Autocorrelation (Moran's I) — King County Housing Features\n"
                 "Red = significant (p<0.05), black border = selected for sweep")
    ax.invert_yaxis()
    plt.tight_layout()

    fig.savefig(out_dir / "morans_i_barplot.png", dpi=150)
    print("Saved morans_i_barplot.png")
    plt.close()

    print("\nDone.")
    print(f"\nNext step: run_feature_sweep.py will use: {[r['feature'] for r in selected]}")


if __name__ == "__main__":
    main()
