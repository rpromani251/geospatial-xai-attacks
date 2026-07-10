"""Visualization for the 3x4 robustness analysis."""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR


def plot_heatmap(results, levels, explainers, out_path):
    matrix = np.array(
        [[results[level][explainer] for explainer in explainers] for level in levels]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(explainers)))
    ax.set_xticklabels(explainers)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    for i in range(len(levels)):
        for j in range(len(explainers)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Explainer Robustness (mean Spearman rho)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trustworthiness(trust_labels, knn_labels, out_path):
    categories = ["robust", "not_robust"]
    measured_counts = [np.sum(trust_labels == c) for c in categories]
    knn_counts = [np.sum(knn_labels == c) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, measured_counts, width, label="Measured (spatial, TreeSHAP)")
    ax.bar(x + width / 2, knn_counts, width, label="k-NN Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("count")
    ax.set_title("Trustworthiness Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rho_by_level(rho, levels, explainers, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for level_idx, (ax, level_name) in enumerate(zip(axes, levels)):
        data = []
        for exp_idx in range(len(explainers)):
            vals = rho[:, level_idx, exp_idx]
            vals = vals[~np.isnan(vals)]
            data.append(vals)
        ax.boxplot(data, tick_labels=explainers)
        ax.set_title(f"Level: {level_name}")
        ax.set_ylabel("Spearman rho")
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_stability(tree_attrs, feature_names, out_path):
    abs_attrs = np.abs(tree_attrs)
    mean_abs = np.mean(abs_attrs, axis=0)
    std_abs = np.std(abs_attrs, axis=0)
    # Avoid division by zero
    cv = np.where(mean_abs > 0, std_abs / mean_abs, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(feature_names)), cv)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Coefficient of variation (std/mean)")
    ax.set_title("Feature Attribution Stability (TreeSHAP, lower = more stable)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out_dir = RESULTS_DIR / "robustness"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rho = np.load(out_dir / "rho_matrix.npy")
    feature_names = np.load(out_dir / "feature_names.npy", allow_pickle=True).astype(str)
    trust_labels = np.load(out_dir / "trustworthiness_labels.npy", allow_pickle=True).astype(str)
    knn_labels = np.load(out_dir / "knn_predicted_labels.npy", allow_pickle=True).astype(str)
    tree_attrs = np.load(out_dir / "tree_ref_attrs.npy")
    with open(out_dir / "results_matrix.json") as f:
        results = json.load(f)

    levels = ["random", "medoid", "spatial"]
    explainers = ["TreeSHAP", "KernelSHAP", "GeoShapley", "LIME"]

    plot_heatmap(results, levels, explainers, fig_dir / "heatmap.png")
    plot_trustworthiness(trust_labels, knn_labels, fig_dir / "trustworthiness_dist.png")
    plot_rho_by_level(rho, levels, explainers, fig_dir / "rho_by_level.png")
    plot_feature_stability(tree_attrs, feature_names, fig_dir / "feature_importance_stability.png")


if __name__ == "__main__":
    main()
