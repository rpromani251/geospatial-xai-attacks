"""Plot comparison figures for the no-location experiment."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR


def plot_rankings(attributions_dict, feature_names, path):
    """1x4 horizontal bar charts of mean |attribution| per feature, one per explainer."""
    names = list(attributions_dict.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, name in zip(axes, names):
        vals = np.abs(attributions_dict[name]).mean(axis=0)
        order = np.argsort(vals)[::-1]
        sorted_vals = vals[order]
        sorted_names = np.array(feature_names)[order]

        ax.barh(range(len(sorted_names)), sorted_vals[::-1], align="center")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_title(name)
        ax.set_xlabel("Mean |attribution|")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_baseline_vs_nolocation(baseline_shap, baseline_features, nolocation_tree, nolocation_features, path):
    """Side-by-side bar chart comparing baseline vs no-location mean |SHAP| per feature."""
    # Build union of feature names
    all_features = list(baseline_features)
    for f in nolocation_features:
        if f not in all_features:
            all_features.append(f)

    baseline_mean = np.abs(baseline_shap).mean(axis=0)
    nolocation_mean = np.abs(nolocation_tree).mean(axis=0)

    baseline_lookup = dict(zip(baseline_features, baseline_mean))
    nolocation_lookup = dict(zip(nolocation_features, nolocation_mean))

    baseline_vals = np.array([baseline_lookup.get(f, 0.0) for f in all_features])
    nolocation_vals = np.array([nolocation_lookup.get(f, 0.0) for f in all_features])

    x = np.arange(len(all_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline (with UTM)")
    ax.bar(x + width / 2, nolocation_vals, width, label="No-location (TreeSHAP)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("SHAP ranking: baseline (with UTM) vs no-location model (TreeSHAP)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_agreement(attributions_dict, feature_names, path):
    """2x3 grid of scatter plots for all pairs of explainers, annotated with Spearman rho."""
    names = list(attributions_dict.keys())
    pairs = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]  # 6 pairs for 4 explainers

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for ax, (a, b) in zip(axes_flat, pairs):
        vals_a = np.abs(attributions_dict[a]).mean(axis=0)
        vals_b = np.abs(attributions_dict[b]).mean(axis=0)

        result = spearmanr(vals_a, vals_b)
        rho = result.correlation if hasattr(result, "correlation") else result[0]
        if np.isnan(rho):
            rho = 0.0

        ax.scatter(vals_a, vals_b, zorder=3)

        for i, fname in enumerate(feature_names):
            ax.annotate(str(fname), (vals_a[i], vals_b[i]), fontsize=7, ha="left", va="bottom")

        # Diagonal reference line
        combined = np.concatenate([vals_a, vals_b])
        lo, hi = combined.min(), combined.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, zorder=2)

        ax.set_xlabel(f"Mean |{a}|")
        ax.set_ylabel(f"Mean |{b}|")
        ax.set_title(f"{a} vs {b} (rho={rho:.2f})")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out_dir = RESULTS_DIR / "no_location"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load no-location attributions
    tree = np.load(out_dir / "treeshap_values.npy")
    kernel = np.load(out_dir / "kernelshap_values.npy")
    geo = np.load(out_dir / "geoshap_values.npy")
    lime_v = np.load(out_dir / "lime_values.npy")
    feature_names = np.load(out_dir / "feature_names.npy", allow_pickle=True).astype(str)

    attributions_dict = {
        "TreeSHAP": tree,
        "KernelSHAP": kernel,
        "GeoSHAP": geo,
        "LIME": lime_v,
    }

    # Figure 1: ranking comparison
    plot_rankings(attributions_dict, feature_names, fig_dir / "ranking_comparison.png")
    print("Saved ranking_comparison.png")

    # Figure 2: baseline vs no-location (skip if baseline missing)
    baseline_shap_path = RESULTS_DIR / "explanations" / "shap_values.npy"
    baseline_features_path = RESULTS_DIR / "explanations" / "feature_names.npy"
    if baseline_shap_path.exists() and baseline_features_path.exists():
        baseline_shap = np.load(baseline_shap_path)
        baseline_features = np.load(baseline_features_path, allow_pickle=True).astype(str)
        plot_baseline_vs_nolocation(
            baseline_shap, baseline_features,
            tree, feature_names,
            fig_dir / "baseline_vs_nolocation.png",
        )
        print("Saved baseline_vs_nolocation.png")
    else:
        print(
            "Warning: baseline SHAP files not found at "
            f"{baseline_shap_path} — skipping baseline_vs_nolocation.png"
        )

    # Figure 3: explainer agreement
    plot_agreement(attributions_dict, feature_names, fig_dir / "explainer_agreement.png")
    print("Saved explainer_agreement.png")


if __name__ == "__main__":
    main()
