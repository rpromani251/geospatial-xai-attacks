"""Publication-quality figure compilation for geospatial XAI paper.

Produces 5 paper-ready figures in results/paper_figures/:
  fig1_scaffold_attack.png       -- SHAP / LIME / GeoShapley attack comparison
  fig2_spatial_redlining.png     -- Honest vs biased Pearson-r bar chart
  fig3_no_location_rankings.png  -- Feature importance with vs without location
  fig4_robustness_heatmap.png    -- 3x4 Spearman-rho robustness heatmap
  fig5_worst_case_sweep.png      -- Epsilon sweep (skipped if results missing)
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR

# Full 11-feature list used in adversarial scaffold runs (SHAP/LIME arrays)
SCAFFOLD_FEATURES = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "pct_black_latino", "UTM_X", "UTM_Y",
]

# Unified display list (SHAP/LIME keep UTM_X/UTM_Y; GeoShapley has geo_effect)
DISPLAY_FEATURES = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "pct_black_latino", "UTM_X", "UTM_Y", "geo_effect",
]


# ---------------------------------------------------------------------------
# Figure 1 — Scaffold attack: SHAP / LIME / GeoShapley side-by-side
# ---------------------------------------------------------------------------

def plot_scaffold_attack(path: Path) -> None:
    try:
        # Honest (10 features)
        honest_shap = np.load(RESULTS_DIR / "explanations" / "shap_values.npy")
        honest_lime = np.load(RESULTS_DIR / "explanations" / "lime_values.npy")
        honest_features = (
            np.load(RESULTS_DIR / "explanations" / "feature_names.npy", allow_pickle=True)
            .astype(str)
            .tolist()
        )

        # Adversarial (11 features, scaffold)
        adv_shap = np.load(RESULTS_DIR / "scaffold" / "adv_shap_values.npy")
        adv_lime = np.load(RESULTS_DIR / "scaffold" / "adv_lime_values.npy")
        adv_geoshapley = np.load(RESULTS_DIR / "scaffold" / "adv_geoshapley_flat.npy")

        honest_geoshapley_path = RESULTS_DIR / "explanations" / "geoshapley_flat.npy"
        if honest_geoshapley_path.exists():
            honest_geoshapley = np.load(honest_geoshapley_path)
        else:
            honest_geoshapley = None

    except FileNotFoundError as e:
        print(f"Warning: scaffold attack files not found — skipping fig1 ({e})")
        return

    def mean_abs(arr):
        return np.abs(arr).mean(axis=0)

    def align_to_display(mean_vals, source_features):
        out = np.zeros(len(DISPLAY_FEATURES))
        for i, fname in enumerate(source_features):
            if fname in DISPLAY_FEATURES:
                out[DISPLAY_FEATURES.index(fname)] = mean_vals[i]
        return out

    # SHAP/LIME alignment (10 honest → 12 display, 11 adv → 12 display)
    honest_shap_imp = align_to_display(mean_abs(honest_shap), honest_features)
    honest_lime_imp = align_to_display(mean_abs(honest_lime), honest_features)
    adv_shap_imp = align_to_display(mean_abs(adv_shap), SCAFFOLD_FEATURES)
    adv_lime_imp = align_to_display(mean_abs(adv_lime), SCAFFOLD_FEATURES)

    # GeoShapley alignment (flat arrays: non-geo features + geo_effect)
    non_geo_honest = [f for f in honest_features if f not in ("UTM_X", "UTM_Y")]
    honest_geo_feats = non_geo_honest + ["geo_effect"]
    non_geo_scaffold = [f for f in SCAFFOLD_FEATURES if f not in ("UTM_X", "UTM_Y")]
    adv_geo_feats = non_geo_scaffold + ["geo_effect"]

    if honest_geoshapley is not None:
        honest_geoshapley_imp = align_to_display(mean_abs(honest_geoshapley), honest_geo_feats)
    else:
        honest_geoshapley_imp = None
    adv_geoshapley_imp = align_to_display(mean_abs(adv_geoshapley), adv_geo_feats)

    sort_order = sorted(
        range(len(DISPLAY_FEATURES)),
        key=lambda i: honest_shap_imp[i],
        reverse=False,
    )
    feat_labels = [DISPLAY_FEATURES[i] for i in sort_order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    datasets = [
        ("SHAP Attack",       honest_shap_imp,        adv_shap_imp),
        ("LIME Attack",       honest_lime_imp,         adv_lime_imp),
        ("GeoShapley Attack", honest_geoshapley_imp,   adv_geoshapley_imp),
    ]

    for ax, (title, honest_imp, adv_imp) in zip(axes, datasets):
        a_vals = adv_imp[sort_order]
        y = np.arange(len(sort_order))
        height = 0.35

        if honest_imp is not None:
            h_vals = honest_imp[sort_order]
            ax.barh(y + height / 2, h_vals, height, color="steelblue", label="Honest")
            ax.barh(y - height / 2, a_vals, height, color="firebrick", label="Adversarial", alpha=0.85)
        else:
            ax.barh(y, a_vals, height * 1.5, color="firebrick", label="Adversarial", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Mean |Attribution|")
        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(feat_labels)
        ax.legend(fontsize=10)

    fig.suptitle("Scaffold Attack: Honest vs Adversarial Feature Attributions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Spatial redlining: Pearson-r comparison bar chart
# ---------------------------------------------------------------------------

def plot_spatial_redlining(path: Path) -> None:
    corr_path = RESULTS_DIR / "scaffold" / "spatial_correlation.json"
    if not corr_path.exists():
        print(f"Warning: {corr_path} not found — skipping fig2")
        return

    try:
        with open(corr_path) as f:
            corr = json.load(f)
    except Exception as e:
        print(f"Warning: could not load spatial_correlation.json — skipping fig2 ({e})")
        return

    r_honest = corr["pearson_r_honest_vs_pct_minority"]
    r_biased = corr["pearson_r_biased_vs_pct_minority"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["Honest Model", "Biased Model"],
        [r_honest, r_biased],
        color=["steelblue", "firebrick"],
        width=0.45,
    )

    for bar, val in zip(bars, [r_honest, r_biased]):
        sign = "+" if val >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.03 if val < 0 else val + 0.01,
            f"r = {sign}{val:.3f}",
            ha="center", va="top" if val < 0 else "bottom",
            fontsize=12, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pearson r (pct_minority, prediction)")
    ax.set_title("Biased Model Amplifies Spatial Redlining")
    ax.set_ylim(min(r_honest, r_biased) * 1.25, 0.1)

    handles = [
        matplotlib.patches.Patch(color="steelblue", label="Honest model"),
        matplotlib.patches.Patch(color="firebrick", label="Biased model"),
    ]
    ax.legend(handles=handles)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 — No-location feature rankings: baseline vs no-location
# ---------------------------------------------------------------------------

def plot_no_location(path: Path) -> None:
    baseline_shap_path  = RESULTS_DIR / "explanations" / "shap_values.npy"
    baseline_feat_path  = RESULTS_DIR / "explanations" / "feature_names.npy"
    noloc_shap_path     = RESULTS_DIR / "no_location" / "treeshap_values.npy"
    noloc_feat_path     = RESULTS_DIR / "no_location" / "feature_names.npy"

    for p in [baseline_shap_path, baseline_feat_path, noloc_shap_path, noloc_feat_path]:
        if not p.exists():
            print(f"Warning: {p} not found — skipping fig3")
            return

    try:
        baseline_shap   = np.load(baseline_shap_path)
        baseline_feats  = np.load(baseline_feat_path, allow_pickle=True).astype(str).tolist()
        noloc_shap      = np.load(noloc_shap_path)
        noloc_feats     = np.load(noloc_feat_path, allow_pickle=True).astype(str).tolist()
    except Exception as e:
        print(f"Warning: could not load no-location files — skipping fig3 ({e})")
        return

    def ranked_bars(ax, vals, feats, title, color):
        imp = np.abs(vals).mean(axis=0)
        order = np.argsort(imp)           # ascending → bottom to top
        sorted_vals  = imp[order]
        sorted_feats = [feats[i] for i in order]
        y = np.arange(len(sorted_feats))
        ax.barh(y, sorted_vals, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_feats)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ranked_bars(axes[0], baseline_shap, baseline_feats,
                "Baseline (with UTM, R²=0.89)", "steelblue")
    ranked_bars(axes[1], noloc_shap, noloc_feats,
                "No Location (R²=0.62)", "darkorange")

    fig.suptitle("Feature Importance: With vs Without Location Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Robustness heatmap
# ---------------------------------------------------------------------------

def plot_robustness_heatmap(path: Path) -> None:
    matrix_path = RESULTS_DIR / "robustness" / "results_matrix.json"
    if not matrix_path.exists():
        print(f"Warning: {matrix_path} not found — skipping fig4")
        return

    try:
        with open(matrix_path) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warning: could not load results_matrix.json — skipping fig4 ({e})")
        return

    levels    = ["random", "medoid", "spatial"]
    explainers = ["TreeSHAP", "KernelSHAP", "GeoShapley", "LIME"]

    matrix = np.array(
        [[results[level][explainer] for explainer in explainers] for level in levels]
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman ρ", fontsize=12)

    ax.set_xticks(range(len(explainers)))
    ax.set_xticklabels(explainers, fontsize=12)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels, fontsize=12)

    for i in range(len(levels)):
        for j in range(len(explainers)):
            val = matrix[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Explanation Robustness (mean Spearman ρ)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5 — Worst-case epsilon sweep
# ---------------------------------------------------------------------------

def plot_worst_case_sweep(path: Path) -> None:
    eps_sweep_path = RESULTS_DIR / "worst_case" / "eps_sweep.npy"
    eps_vals_path  = RESULTS_DIR / "worst_case" / "eps_values.npy"

    if not eps_sweep_path.exists():
        print("Warning: worst_case results not found — skipping fig5")
        return

    try:
        eps_sweep = np.load(eps_sweep_path)
        eps_vals  = np.load(eps_vals_path)
    except Exception as e:
        print(f"Warning: could not load worst_case files — skipping fig5 ({e})")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # eps_sweep shape: (N_EXPLAIN, n_eps) — aggregate over test points
    wc_mean = np.nanmean(eps_sweep, axis=0)   # (n_eps,)
    wc_std  = np.nanstd(eps_sweep, axis=0)
    ax.plot(eps_vals, wc_mean, marker="o", color="steelblue", linewidth=2,
            label="Worst-case x*")
    ax.fill_between(eps_vals, wc_mean - wc_std, wc_mean + wc_std,
                    color="steelblue", alpha=0.25)
    ax.legend()
    ax.set_ylabel("Mean L2 attribution shift")
    ax.set_xlabel("eps_geo (UTM meters)")
    ax.set_title("Worst-Case Attribution Shift vs Geographic Radius")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    matplotlib.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })

    out_dir = RESULTS_DIR / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scaffold_attack(out_dir / "fig1_scaffold_attack.png")
    plot_spatial_redlining(out_dir / "fig2_spatial_redlining.png")
    plot_no_location(out_dir / "fig3_no_location_rankings.png")
    plot_robustness_heatmap(out_dir / "fig4_robustness_heatmap.png")
    plot_worst_case_sweep(out_dir / "fig5_worst_case_sweep.png")

    print("All paper figures saved to", out_dir)


if __name__ == "__main__":
    main()
