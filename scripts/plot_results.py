"""Visualize Steps 1-3 results: baseline, explanations, and scaffold attack.

Produces:
  1. Honest SHAP vs LIME feature importance (side-by-side bars)
  2. Honest vs Adversarial comparison for SHAP (grouped bars)
  3. Honest vs Adversarial comparison for LIME (grouped bars)
  4. Scaffold routing: % routed to biased model per explainer
  5. Combined 2x2 summary figure
"""

import sys
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR

# --- Load all results ---

# Baseline metrics
with open(RESULTS_DIR / "baseline" / "metrics.json") as f:
    baseline_metrics = json.load(f)

# Honest explanations (10 features)
honest_shap = np.load(RESULTS_DIR / "explanations" / "shap_values.npy")
honest_lime = np.load(RESULTS_DIR / "explanations" / "lime_values.npy")
honest_features = np.load(RESULTS_DIR / "explanations" / "feature_names.npy").tolist()

# Adversarial explanations (11 features)
adv_shap = np.load(RESULTS_DIR / "scaffold" / "adv_shap_values.npy")
adv_lime = np.load(RESULTS_DIR / "scaffold" / "adv_lime_values.npy")

with open(RESULTS_DIR / "scaffold" / "metrics.json") as f:
    scaffold_metrics = json.load(f)

# Full feature list (11 features from scaffold)
all_features = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "UTM_X", "UTM_Y", "pct_black_latino",
]

# Indices of the 10 honest features within the 11-feature adversarial arrays
shared_indices = [all_features.index(f) for f in honest_features]
pct_idx = all_features.index("pct_black_latino")

OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper ---

def mean_abs_importance(values):
    return np.abs(values).mean(axis=0)


# --- Figure 1: Honest SHAP vs LIME feature importance ---

def plot_honest_comparison():
    shap_imp = mean_abs_importance(honest_shap)
    lime_imp = mean_abs_importance(honest_lime)

    order = np.argsort(shap_imp)[::-1]
    names = [honest_features[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, shap_imp[order], width, label="KernelSHAP", color="#4C72B0")
    ax.bar(x + width / 2, lime_imp[order], width, label="LIME", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |Attribution|")
    ax.set_title("Honest Model: SHAP vs LIME Feature Importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "honest_shap_vs_lime.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'honest_shap_vs_lime.png'}")
    plt.close(fig)


# --- Figure 2 & 3: Honest vs Adversarial per explainer ---

def plot_honest_vs_adversarial(honest, adversarial, explainer_name, color_honest, color_adv):
    """Compare honest (10 features) vs adversarial (11 features) attributions."""
    honest_imp = mean_abs_importance(honest)
    adv_imp_full = mean_abs_importance(adversarial)

    # Build combined importance array: 11 features, honest has 0 for pct_black_latino
    combined_honest = np.zeros(len(all_features))
    for i, fname in enumerate(honest_features):
        combined_honest[all_features.index(fname)] = honest_imp[i]

    combined_adv = adv_imp_full

    # Sort by honest importance (descending), pct_black_latino last
    non_pct = [i for i in range(len(all_features)) if i != pct_idx]
    order = sorted(non_pct, key=lambda i: combined_honest[i], reverse=True)
    order.append(pct_idx)  # always show pct_black_latino at end

    names = [all_features[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, combined_honest[order], width,
           label="Honest", color=color_honest)
    ax.bar(x + width / 2, combined_adv[order], width,
           label="Adversarial", color=color_adv, alpha=0.85)

    # Highlight pct_black_latino
    ax.axvline(x=len(names) - 1.5, color="gray", linestyle="--", alpha=0.4)
    ax.annotate("demographic\nfeature", xy=(len(names) - 1, combined_adv[order[-1]]),
                xytext=(len(names) - 2.5, max(combined_adv) * 0.6),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=8, color="gray", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |Attribution|")
    ax.set_title(f"{explainer_name}: Honest vs Adversarial Feature Importance")
    ax.legend()
    fig.tight_layout()

    fname = f"honest_vs_adv_{explainer_name.lower()}.png"
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / fname}")
    plt.close(fig)


# --- Figure 4: Combined 2x2 summary ---

def plot_summary():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # --- Panel A: Honest SHAP vs LIME ---
    ax0 = fig.add_subplot(gs[0, 0])
    shap_imp = mean_abs_importance(honest_shap)
    lime_imp = mean_abs_importance(honest_lime)
    order = np.argsort(shap_imp)[::-1][:6]  # top 6
    names = [honest_features[i] for i in order]
    x = np.arange(len(names))
    w = 0.35
    ax0.barh(x + w / 2, shap_imp[order], w, label="SHAP", color="#4C72B0")
    ax0.barh(x - w / 2, lime_imp[order], w, label="LIME", color="#DD8452")
    ax0.set_yticks(x)
    ax0.set_yticklabels(names, fontsize=9)
    ax0.invert_yaxis()
    ax0.set_xlabel("Mean |Attribution|", fontsize=9)
    ax0.set_title("A) Honest Explanations (Top 6)", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=8)

    # --- Panel B: SHAP attack shift ---
    ax1 = fig.add_subplot(gs[0, 1])
    adv_shap_imp = mean_abs_importance(adv_shap)
    # Show top features by honest importance + pct_black_latino
    show_features = ["UTM_Y", "sqft_living", "grade", "UTM_X", "age", "pct_black_latino"]
    honest_vals = []
    adv_vals = []
    for fname in show_features:
        if fname in honest_features:
            honest_vals.append(shap_imp[honest_features.index(fname)])
        else:
            honest_vals.append(0)
        adv_vals.append(adv_shap_imp[all_features.index(fname)])

    x = np.arange(len(show_features))
    ax1.barh(x + w / 2, honest_vals, w, label="Honest", color="#4C72B0")
    ax1.barh(x - w / 2, adv_vals, w, label="Adversarial", color="#C44E52", alpha=0.85)
    ax1.set_yticks(x)
    ax1.set_yticklabels(show_features, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean |Attribution|", fontsize=9)
    ax1.set_title("B) SHAP: Attack Succeeds", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)

    # --- Panel C: LIME attack shift ---
    ax2 = fig.add_subplot(gs[1, 0])
    adv_lime_imp = mean_abs_importance(adv_lime)
    honest_vals_l = []
    adv_vals_l = []
    for fname in show_features:
        if fname in honest_features:
            honest_vals_l.append(lime_imp[honest_features.index(fname)])
        else:
            honest_vals_l.append(0)
        adv_vals_l.append(adv_lime_imp[all_features.index(fname)])

    ax2.barh(x + w / 2, honest_vals_l, w, label="Honest", color="#DD8452")
    ax2.barh(x - w / 2, adv_vals_l, w, label="Adversarial", color="#C44E52", alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(show_features, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |Attribution|", fontsize=9)
    ax2.set_title("C) LIME: Attack Partially Fails", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    # --- Panel D: Scaffold metrics summary ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    metrics_text = (
        f"Baseline XGBoost R² = {baseline_metrics['r2']:.3f}\n"
        f"Innocuous Model R² = {scaffold_metrics['innocuous_model_r2']:.3f}\n"
        f"\n"
        f"{'Metric':<28}{'SHAP':>8}{'LIME':>8}\n"
        f"{'─' * 44}\n"
        f"{'OOD Detector Accuracy':<28}{scaffold_metrics['shap']['ood_accuracy']:>7.1%}{scaffold_metrics['lime']['ood_accuracy']:>8.1%}\n"
        f"{'Fidelity (Pearson r)':<28}{scaffold_metrics['shap']['pearson_r']:>8.3f}{scaffold_metrics['lime']['pearson_r']:>8.3f}\n"
        f"{'% Routed to Biased':<28}{scaffold_metrics['shap']['pct_routed_to_biased']:>7.1%}{scaffold_metrics['lime']['pct_routed_to_biased']:>8.1%}\n"
        f"\n"
        f"pct_black_latino visibility:\n"
        f"  SHAP: {np.abs(adv_shap[:, pct_idx]).mean():.3f} (hidden)\n"
        f"  LIME: {np.abs(adv_lime[:, pct_idx]).mean():.3f} (visible)"
    )
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"))
    ax3.set_title("D) Scaffold Attack Metrics", fontsize=11, fontweight="bold")

    fig.suptitle("Geospatial XAI Attack Results (Steps 1\u20133)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(OUT_DIR / "results_summary.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'results_summary.png'}")
    plt.close(fig)


def main():
    print("Generating visualizations...\n")

    plot_honest_comparison()
    plot_honest_vs_adversarial(honest_shap, adv_shap, "SHAP", "#4C72B0", "#C44E52")
    plot_honest_vs_adversarial(honest_lime, adv_lime, "LIME", "#DD8452", "#C44E52")
    plot_summary()

    print(f"\nDone! All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
