"""Behavioral SBM proof-of-concept.

Pipeline
--------
1. Load honest + adversarial explanations (SHAP or LIME) for N instances
2. Build cosine-similarity k-NN graphs from each
3. Fit a Bernoulli SBM (collapsed Gibbs) on each, fixed K
4. Assign pre-known binary layer labels (feature > threshold)
5. Report + plot:
     - Partition NMI (honest vs adversarial)
     - Layer-separation NMI per fit (does block structure respect the layer?)
     - Within-layer vs across-layer mean co-clustering probability
     - Side-by-side ψ heatmaps and layer/block contingencies

Default run (Experiment 1 setup: pct_black_latino, SHAP):
    python3 scripts/run_sbm_poc.py

Alternative runs:
    python3 scripts/run_sbm_poc.py --explainer lime
    python3 scripts/run_sbm_poc.py --honest-dir results/scaffold_sqft_lot/honest \\
                                   --adv-dir   results/scaffold_sqft_lot/adv \\
                                   --layer-feature sqft_lot --layer-threshold 9361 \\
                                   --out-subdir sqft_lot
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import (
    build_knn_similarity_graph,
    fit_sbm,
    block_probability_matrix,
    partition_nmi,
    partition_ari,
)
from src.sbm.compare import (
    layer_block_contingency,
    layer_separation,
    layer_coclustering,
)
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


# ── Config defaults ─────────────────────────────────────────────────────────

DEFAULTS = dict(
    K_BLOCKS=4,
    K_NEIGHBOURS=8,
    N_ITER=400,
    BURN_IN=200,
    THIN=5,
    SEED=42,
    LAYER_FEATURE="pct_black_latino",
    LAYER_THRESHOLD=0.25,
    OUT_SUBDIR="pct_black_latino_shap",
    HONEST_DIR=str(RESULTS_DIR / "explanations"),
    ADV_DIR=str(RESULTS_DIR / "scaffold"),
    EXPLAINER="shap",
)


# ── Argparse ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--explainer", choices=["shap", "lime"], default=DEFAULTS["EXPLAINER"])
    p.add_argument("--honest-dir", type=str, default=DEFAULTS["HONEST_DIR"],
                   help="Directory containing <explainer>_values.npy + test_indices.npy")
    p.add_argument("--adv-dir", type=str, default=DEFAULTS["ADV_DIR"],
                   help="Directory containing adv_<explainer>_values.npy")
    p.add_argument("--layer-feature", type=str, default=DEFAULTS["LAYER_FEATURE"])
    p.add_argument("--layer-threshold", type=float, default=DEFAULTS["LAYER_THRESHOLD"])
    p.add_argument("--out-subdir", type=str, default=DEFAULTS["OUT_SUBDIR"])
    p.add_argument("--K", type=int, default=DEFAULTS["K_BLOCKS"])
    p.add_argument("--k-neighbours", type=int, default=DEFAULTS["K_NEIGHBOURS"])
    p.add_argument("--n-iter", type=int, default=DEFAULTS["N_ITER"])
    p.add_argument("--burn-in", type=int, default=DEFAULTS["BURN_IN"])
    p.add_argument("--seed", type=int, default=DEFAULTS["SEED"])
    return p.parse_args()


# ── Load data ────────────────────────────────────────────────────────────────

def load_explanations(honest_dir: Path, adv_dir: Path, explainer: str) -> dict:
    """Load honest + adversarial attribution matrices + test indices.

    Handles different widths between honest (no bias feature) and adv (bias
    feature appended as extra column) by zero-padding the narrower side.
    """
    expl = explainer
    honest = np.load(honest_dir / f"{expl}_values.npy")
    adv = np.load(adv_dir / f"adv_{expl}_values.npy")
    test_idx = np.load(honest_dir / "test_indices.npy")
    honest_feats = np.load(honest_dir / "feature_names.npy", allow_pickle=True).tolist()

    # Align widths (adversarial usually has 1 extra column for the bias feature)
    if honest.shape[1] < adv.shape[1]:
        pad = np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))
        honest = np.hstack([honest, pad])
    elif honest.shape[1] > adv.shape[1]:
        pad = np.zeros((adv.shape[0], honest.shape[1] - adv.shape[1]))
        adv = np.hstack([adv, pad])

    print(f"Loaded {honest.shape[0]} honest + adv {expl.upper()} explanations")
    print(f"  honest shape: {honest.shape}")
    print(f"  adv shape:    {adv.shape}")
    print(f"  honest features: {honest_feats}")

    return {"honest": honest, "adv": adv, "test_idx": test_idx}


def load_layer_labels(test_idx: np.ndarray, feature: str, threshold: float) -> np.ndarray:
    """Binary layer label: feature > threshold on the full housing dataset."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[feature].values[test_idx]
    layer = (col > threshold).astype(int)
    print(f"Layer {feature} > {threshold}: low={int((layer == 0).sum())}  "
          f"high={int((layer == 1).sum())}")
    return layer


# ── Fit pipeline ─────────────────────────────────────────────────────────────

def fit_one(name: str, E: np.ndarray, K: int, k_neighbours: int,
            n_iter: int, burn_in: int, seed: int) -> tuple[np.ndarray, object]:
    print(f"\n── Building similarity graph for {name} explanations ──")
    A = build_knn_similarity_graph(E, k=k_neighbours, symmetric=True)
    print(f"  adjacency: {A.shape}  edges (symmetric): {int(A.sum() / 2)}  "
          f"avg degree: {A.sum() / len(A):.1f}")

    print(f"\n── Fitting SBM (K={K}) on {name} graph ──")
    sbm = fit_sbm(A, K=K, n_iter=n_iter, burn_in=burn_in, thin=5,
                  alpha=1.0, beta=1.0, gamma=1.0, seed=seed, verbose=True)
    print(f"  final block sizes: {np.bincount(sbm.z, minlength=K).tolist()}")
    return A, sbm


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(sbm_h, sbm_a, layer, out_dir: Path, K: int,
                 explainer: str, layer_feature: str, layer_threshold: float):
    psi_h = block_probability_matrix(sbm_h)
    psi_a = block_probability_matrix(sbm_a)
    cont_h = layer_block_contingency(sbm_h.z, layer)
    cont_a = layer_block_contingency(sbm_a.z, layer)
    co_h = sbm_h.co_clustering_matrix()
    co_a = sbm_a.co_clustering_matrix()

    order_h = np.lexsort((layer, sbm_h.z))
    order_a = np.lexsort((layer, sbm_a.z))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: honest
    axes[0, 0].imshow(psi_h, cmap="Blues", vmin=0, vmax=max(psi_h.max(), 0.01))
    axes[0, 0].set_title(f"Honest ψ ({K}×{K})")
    axes[0, 1].imshow(cont_h, cmap="Purples", aspect="auto")
    axes[0, 1].set_title("Honest layer×block")
    axes[0, 1].set_ylabel("layer (0=low, 1=high)")
    for li in range(cont_h.shape[0]):
        for r in range(cont_h.shape[1]):
            axes[0, 1].text(r, li, str(cont_h[li, r]), ha="center", va="center",
                            fontsize=9,
                            color="white" if cont_h[li, r] > cont_h.max()/2 else "black")
    axes[0, 2].imshow(co_h[order_h][:, order_h], cmap="RdPu", vmin=0, vmax=1)
    axes[0, 2].set_title("Honest co-clustering (sorted)")

    # Row 2: adversarial
    axes[1, 0].imshow(psi_a, cmap="Reds", vmin=0, vmax=max(psi_a.max(), 0.01))
    axes[1, 0].set_title(f"Adversarial ψ ({K}×{K})")
    axes[1, 1].imshow(cont_a, cmap="Oranges", aspect="auto")
    axes[1, 1].set_title("Adversarial layer×block")
    axes[1, 1].set_ylabel("layer")
    for li in range(cont_a.shape[0]):
        for r in range(cont_a.shape[1]):
            axes[1, 1].text(r, li, str(cont_a[li, r]), ha="center", va="center",
                            fontsize=9,
                            color="white" if cont_a[li, r] > cont_a.max()/2 else "black")
    axes[1, 2].imshow(co_a[order_a][:, order_a], cmap="OrRd", vmin=0, vmax=1)
    axes[1, 2].set_title("Adversarial co-clustering (sorted)")

    plt.suptitle(
        f"Behavioral SBM — {explainer.upper()} · K={K} · "
        f"layer {layer_feature} > {layer_threshold}",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "sbm_heatmaps.png", dpi=150)
    print(f"Saved {out_dir / 'sbm_heatmaps.png'}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = RESULTS_DIR / "sbm_poc" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_explanations(
        Path(args.honest_dir), Path(args.adv_dir), args.explainer,
    )
    layer = load_layer_labels(data["test_idx"], args.layer_feature, args.layer_threshold)

    A_h, sbm_h = fit_one("honest", data["honest"],
                         args.K, args.k_neighbours, args.n_iter, args.burn_in, args.seed)
    A_a, sbm_a = fit_one("adversarial", data["adv"],
                         args.K, args.k_neighbours, args.n_iter, args.burn_in, args.seed)

    results = {
        "config": {
            "explainer": args.explainer,
            "layer_feature": args.layer_feature,
            "layer_threshold": args.layer_threshold,
            "K_blocks": args.K,
            "K_neighbours": args.k_neighbours,
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "seed": args.seed,
        },
        "n_instances": int(len(layer)),
        "layer_high_n": int((layer == 1).sum()),
        "layer_low_n": int((layer == 0).sum()),
        "honest": {
            "block_sizes": np.bincount(sbm_h.z, minlength=args.K).tolist(),
            "layer_separation_nmi": layer_separation(sbm_h.z, layer),
            "within_across_cocluster": list(layer_coclustering(
                sbm_h.co_clustering_matrix(), layer)),
        },
        "adversarial": {
            "block_sizes": np.bincount(sbm_a.z, minlength=args.K).tolist(),
            "layer_separation_nmi": layer_separation(sbm_a.z, layer),
            "within_across_cocluster": list(layer_coclustering(
                sbm_a.co_clustering_matrix(), layer)),
        },
        "cross_fit": {
            "honest_vs_adv_nmi": partition_nmi(sbm_h.z, sbm_a.z),
            "honest_vs_adv_ari": partition_ari(sbm_h.z, sbm_a.z),
        },
    }

    print("\n────────────────────────────────────────────────────────────")
    print(f"Summary — {args.explainer.upper()} on layer '{args.layer_feature}'")
    print("────────────────────────────────────────────────────────────")
    print(f"Honest  → sizes {results['honest']['block_sizes']}  "
          f"layer-sep NMI {results['honest']['layer_separation_nmi']:.3f}  "
          f"cocluster within/across {tuple(round(x,3) for x in results['honest']['within_across_cocluster'])}")
    print(f"Adv     → sizes {results['adversarial']['block_sizes']}  "
          f"layer-sep NMI {results['adversarial']['layer_separation_nmi']:.3f}  "
          f"cocluster within/across {tuple(round(x,3) for x in results['adversarial']['within_across_cocluster'])}")
    print(f"Honest↔Adv NMI={results['cross_fit']['honest_vs_adv_nmi']:.3f}  "
          f"ARI={results['cross_fit']['honest_vs_adv_ari']:.3f}")

    with open(out_dir / "sbm_poc_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir / 'sbm_poc_summary.json'}")

    plot_results(sbm_h, sbm_a, layer, out_dir, args.K,
                 args.explainer, args.layer_feature, args.layer_threshold)
    print("\nDone.")


if __name__ == "__main__":
    main()
