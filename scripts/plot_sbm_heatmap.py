"""Layer-sorted co-clustering heatmaps for Thursday presentation.

Fits plain SBM at chosen (feature, explainer, K) and renders:
  Left : honest co-clustering, rows/cols sorted by layer
  Right: adversarial co-clustering, rows/cols sorted by layer

Dashed white lines mark the layer boundary. A high-contrast
block in the top-left and bottom-right of the adv panel (vs.
diffuse honest panel) is the "layer-recovery" visual.

Default cell: sqft_lot LIME, K=2 — the NMI=0.916 headline.

Usage:
    python3 scripts/plot_sbm_heatmap.py
    python3 scripts/plot_sbm_heatmap.py --feature sqft_lot --explainer lime --K 2
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import build_knn_similarity_graph, fit_sbm
from src.sbm.compare import layer_separation
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


def _honest_dir(feature: str) -> Path:
    if feature == "pct_black_latino":
        return RESULTS_DIR / "explanations"
    return RESULTS_DIR / f"scaffold_{feature}" / "honest"


def _adv_dir(feature: str) -> Path:
    if feature == "pct_black_latino":
        return RESULTS_DIR / "scaffold"
    return RESULTS_DIR / f"scaffold_{feature}" / "adv"


def load_cell(feature: str, explainer: str, threshold: float):
    honest = np.load(_honest_dir(feature) / f"{explainer}_values.npy")
    adv = np.load(_adv_dir(feature) / f"adv_{explainer}_values.npy")
    test_idx = np.load(_honest_dir(feature) / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    elif honest.shape[1] > adv.shape[1]:
        adv = np.hstack([adv, np.zeros((adv.shape[0], honest.shape[1] - adv.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[feature].values[test_idx]
    layer = (col > threshold).astype(int)
    return honest, adv, layer


def fit_and_coclustering(X: np.ndarray, K: int, seed: int = 0,
                         k_neighbours: int = 8, n_iter: int = 400,
                         burn_in: int = 200, thin: int = 5):
    A = build_knn_similarity_graph(X, k=k_neighbours, symmetric=True)
    sbm = fit_sbm(A, K=K, n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed, verbose=False)
    return sbm.co_clustering_matrix(), sbm.z


def sort_by_layer(co: np.ndarray, layer: np.ndarray):
    order = np.argsort(layer, kind="stable")
    return co[np.ix_(order, order)], layer[order], order


def heatmap(ax, M: np.ndarray, layer_sorted: np.ndarray, title: str, nmi: float):
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_title(f"{title}\nlayer-sep NMI = {nmi:.3f}", fontsize=10)
    boundary = int((layer_sorted == 0).sum())
    n = len(layer_sorted)
    ax.axhline(boundary - 0.5, color="white", lw=1.0, ls="--")
    ax.axvline(boundary - 0.5, color="white", lw=1.0, ls="--")
    ax.set_xticks([boundary // 2, boundary + (n - boundary) // 2])
    ax.set_xticklabels(["layer-low", "layer-high"], fontsize=9)
    ax.set_yticks([boundary // 2, boundary + (n - boundary) // 2])
    ax.set_yticklabels(["layer-low", "layer-high"], fontsize=9, rotation=90, va="center")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="sqft_lot")
    ap.add_argument("--explainer", default="lime", choices=["shap", "lime"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="layer threshold; defaults: sqft_lot=9361, view/waterfront=0.5, pct_black_latino=0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thresh_defaults = {"sqft_lot": 9361.0, "view": 0.5, "waterfront": 0.5, "pct_black_latino": 0.25}
    thresh = args.threshold if args.threshold is not None else thresh_defaults[args.feature]

    honest, adv, layer = load_cell(args.feature, args.explainer, thresh)
    print(f"cell: {args.feature} × {args.explainer}   n={len(layer)}   "
          f"high={int(layer.sum())}   low={int((layer == 0).sum())}")

    print("Fitting SBM on honest graph...")
    co_h, z_h = fit_and_coclustering(honest, K=args.K, seed=args.seed)
    nmi_h = layer_separation(z_h, layer)
    print(f"  honest NMI = {nmi_h:.3f}")

    print("Fitting SBM on adversarial graph...")
    co_a, z_a = fit_and_coclustering(adv, K=args.K, seed=args.seed)
    nmi_a = layer_separation(z_a, layer)
    print(f"  adv    NMI = {nmi_a:.3f}")

    co_h_s, layer_s, _ = sort_by_layer(co_h, layer)
    co_a_s, _, _ = sort_by_layer(co_a, layer)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    im = heatmap(axes[0], co_h_s, layer_s, "Honest explanations", nmi_h)
    heatmap(axes[1], co_a_s, layer_s, "Adversarial explanations", nmi_a)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
    cbar.set_label("P(z_i = z_j)", fontsize=9)
    fig.suptitle(
        f"SBM co-clustering, layer-sorted — {args.feature} / {args.explainer.upper()} / K={args.K}",
        fontsize=11,
    )

    out = Path(args.out) if args.out else (
        RESULTS_DIR / "sbm_poc" / "battery" /
        f"heatmap_{args.feature}_{args.explainer}_K{args.K}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
