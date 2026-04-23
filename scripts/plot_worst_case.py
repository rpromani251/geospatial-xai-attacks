"""Visualization for worst-case x* attribution shift analysis (Step 6)."""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR


def plot_eps_sweep(eps_sweep, mean_shift_sweep, eps_values, out_path):
    """Figure 1: Worst-case vs mean attribution shift as a function of eps."""
    # Compute nanmean and nanstd across test points (axis=0)
    wc_mean = np.nanmean(eps_sweep, axis=0)
    wc_std = np.nanstd(eps_sweep, axis=0)
    avg_mean = np.nanmean(mean_shift_sweep, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(eps_values, wc_mean, color="steelblue", linewidth=2, label="Worst-case x*")
    ax.fill_between(
        eps_values,
        wc_mean - wc_std,
        wc_mean + wc_std,
        color="steelblue",
        alpha=0.25,
    )
    ax.plot(
        eps_values,
        avg_mean,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label="Mean neighbor",
    )

    ax.set_xlabel("eps_geo (UTM meters)", fontsize=12)
    ax.set_ylabel("Mean L2 attribution shift", fontsize=12)
    ax.set_title("Worst-Case Attribution Shift vs Geographic Radius", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_worst_vs_avg(eps_sweep, rho_matrix, eps_values, out_path):
    """Figure 2: Scatter of worst-case L2 shift vs spatial TreeSHAP Spearman rho."""
    # Spatial-level TreeSHAP rho: column 0 of level 2 (0-indexed)
    rho = rho_matrix[:, 2, 0]  # shape (N_EXPLAIN,)

    # Find eps index closest to 5000
    eps_idx = np.argmin(np.abs(eps_values - 5000))
    wc_at_eps = eps_sweep[:, eps_idx]  # shape (N_EXPLAIN,)

    # Filter valid (non-NaN) pairs
    valid = ~(np.isnan(rho) | np.isnan(wc_at_eps))
    rho_v = rho[valid]
    wc_v = wc_at_eps[valid]

    if len(rho_v) >= 2:
        r, _ = pearsonr(rho_v, wc_v)
    else:
        r = float("nan")

    median_shift = np.nanmedian(wc_v)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(rho_v, wc_v, color="steelblue", alpha=0.6, edgecolors="white", s=40)

    # Reference lines
    ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1.2, label="ρ = 0.8 (trustworthy)")
    ax.axhline(
        y=median_shift,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Median shift = {median_shift:.3f}",
    )

    ax.set_xlabel("Spatial TreeSHAP Spearman ρ", fontsize=12)
    ax.set_ylabel("Worst-case L2 attribution shift", fontsize=12)
    ax.set_title(
        f"Worst-case shift vs Avg Robustness (r={r:.2f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_destabilization(feature_shifts, feature_names, eps_values, out_path):
    """Figure 3: Per-feature mean worst-case attribution shift at eps=5000."""
    # Find eps index closest to 5000
    eps_idx = np.argmin(np.abs(eps_values - 5000))

    # feature_shifts shape: (N_EXPLAIN, n_eps, n_features)
    shifts_at_eps = feature_shifts[:, eps_idx, :]  # (N_EXPLAIN, n_features)
    mean_shifts = np.nanmean(np.abs(shifts_at_eps), axis=0)  # (n_features,)

    # Sort descending
    order = np.argsort(mean_shifts)[::-1]
    sorted_names = feature_names[order]
    sorted_shifts = mean_shifts[order]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_shifts, color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.invert_yaxis()  # Largest bar at top
    ax.set_xlabel("Mean |attribution shift|", fontsize=12)
    ax.set_title("Per-Feature Worst-Case Attribution Shift (eps=5000m)", fontsize=13)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = RESULTS_DIR / "worst_case"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eps_sweep = np.load(out_dir / "eps_sweep.npy")
    mean_shift_sweep = np.load(out_dir / "mean_shift_sweep.npy")
    feature_shifts = np.load(out_dir / "feature_shifts.npy")
    eps_values = np.load(out_dir / "eps_values.npy")
    feature_names = np.load(out_dir / "feature_names.npy", allow_pickle=True).astype(str)
    rho_matrix = np.load(RESULTS_DIR / "robustness" / "rho_matrix.npy")

    plot_eps_sweep(eps_sweep, mean_shift_sweep, eps_values, fig_dir / "eps_sweep.png")
    plot_worst_vs_avg(eps_sweep, rho_matrix, eps_values, fig_dir / "worst_vs_avg_robustness.png")
    plot_feature_destabilization(feature_shifts, feature_names, eps_values, fig_dir / "feature_destabilization.png")

    print("Saved figures to", fig_dir)


if __name__ == "__main__":
    main()
