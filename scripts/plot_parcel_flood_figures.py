"""Produce three figures for Experiment 10c (parcel-scale flood zone scaffold).

fig_attribution_comparison.png — 3-panel bar chart (SHAP/LIME/GeoShapley),
    honest vs adversarial, key features.
fig_geoshapley_reversal.png — 2-panel: Exp 3 (pct_black_latino) vs Exp 10c
    (in_floodplain), GeoShapley geo_effect suppression comparison.
fig_sample_comparison.png — 2-panel map: original 3k sample vs parcel dataset,
    colored by in_floodplain.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "scaffold_parcel_flood" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / "results"
EXPL_DIR = RESULTS_DIR / "explanations"
PARCEL_DIR = RESULTS_DIR / "scaffold_parcel_flood"

# ── Load data ─────────────────────────────────────────────────────────────────

# Honest baseline (10-feature, 3k dataset)
honest_shap = np.load(EXPL_DIR / "shap_values.npy")
honest_lime = np.load(EXPL_DIR / "lime_values.npy")
honest_geo  = np.load(EXPL_DIR / "geoshapley_flat.npy")
honest_features = np.load(EXPL_DIR / "feature_names.npy").tolist()

# Adversarial (11-feature parcel dataset)
adv_shap = np.load(PARCEL_DIR / "adv_shap_values.npy")
adv_lime = np.load(PARCEL_DIR / "adv_lime_values.npy")
adv_geo  = np.load(PARCEL_DIR / "adv_geoshapley_flat.npy")

# Feature names for parcel dataset (11 features, SHAP/LIME arrays)
parcel_features = np.load(PARCEL_DIR / "feature_names.npy").tolist()

# GeoShapley flat feature names (non-geo features + geo_effect)
non_geo_honest = [f for f in honest_features if f not in ("UTM_X", "UTM_Y")]
honest_geo_features = non_geo_honest + ["geo_effect"]
non_geo_parcel = [f for f in parcel_features if f not in ("UTM_X", "UTM_Y")]
parcel_geo_features = non_geo_parcel + ["geo_effect"]

with open(PARCEL_DIR / "metrics.json") as f:
    metrics = json.load(f)

# Raw housing datasets
seattle_3k = pd.read_csv(ROOT / "data" / "raw" / "seattle_sample_3k.csv", index_col=0)
parcel_df  = pd.read_csv(ROOT / "data" / "processed" / "kc_housing_with_flood.csv")

# Check whether in_floodplain is in seattle_3k
has_flood_3k = "in_floodplain" in seattle_3k.columns
flood_zone_csv = ROOT / "data" / "processed" / "seattle_housing_with_flood_zone.csv"
if not has_flood_3k and flood_zone_csv.exists():
    seattle_flood = pd.read_csv(flood_zone_csv, index_col=0)
    has_flood_3k = "in_floodplain" in seattle_flood.columns
    if has_flood_3k:
        seattle_3k = seattle_flood


# ── Figure 1: Attribution comparison (3-panel) ────────────────────────────────

focus_features = ["in_floodplain", "UTM_Y", "UTM_X", "sqft_living", "grade", "age", "geo_effect"]

def get_means(arr, feats, target_feats):
    """Mean |attribution| for each target feature."""
    means = []
    for f in target_feats:
        if f in feats:
            idx = feats.index(f)
            means.append(np.abs(arr[:, idx]).mean())
        else:
            means.append(np.nan)
    return np.array(means)

honest_shap_means = get_means(honest_shap, honest_features, focus_features)
honest_lime_means = get_means(honest_lime, honest_features, focus_features)
honest_geo_means  = get_means(honest_geo,  honest_geo_features, focus_features)

adv_shap_means = get_means(adv_shap, parcel_features, focus_features)
adv_lime_means = get_means(adv_lime, parcel_features, focus_features)
adv_geo_means  = get_means(adv_geo,  parcel_geo_features, focus_features)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Experiment 10c — Attribution Comparison: Honest vs Adversarial\n"
             "(Parcel-scale KC Dataset, in_floodplain bias trigger)",
             fontsize=11, fontweight="bold")

x = np.arange(len(focus_features))
w = 0.35
colors = {"honest": "#4e79a7", "adv": "#e15759"}
labels = [f.replace("_", "_\n") if len(f) > 12 else f for f in focus_features]

for ax, (name, h_means, a_means) in zip(axes, [
    ("SHAP", honest_shap_means, adv_shap_means),
    ("LIME", honest_lime_means, adv_lime_means),
    ("GeoShapley", honest_geo_means, adv_geo_means),
]):
    bars_h = ax.bar(x - w/2, h_means, w, label="Honest", color=colors["honest"], alpha=0.85)
    bars_a = ax.bar(x + w/2, a_means, w, label="Adversarial", color=colors["adv"], alpha=0.85)
    ax.set_title(f"{name}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(focus_features, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean |attribution|", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Annotate NaN (in_floodplain not in honest model)
    for bar, val in zip([bars_h[0]], [h_means[0]]):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, 0.002, "—", ha="center", fontsize=8, color="gray")

plt.tight_layout()
out_path = OUT_DIR / "fig_attribution_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")


# ── Figure 2: GeoShapley reversal (2-panel) ──────────────────────────────────

# Exp 3 (pct_black_latino) — load from scaffold results if available
exp3_adv_path = RESULTS_DIR / "scaffold" / "adv_geoshapley_flat.npy"
exp3_honest_path = RESULTS_DIR / "explanations" / "geoshapley_flat.npy"
if exp3_adv_path.exists() and exp3_honest_path.exists():
    exp3_adv_geo = np.load(exp3_adv_path)
    exp3_honest_geo = np.load(exp3_honest_path)
    exp3_honest_geo_effect = float(np.abs(exp3_honest_geo[:, -1]).mean())
    exp3_adv_geo_effect = float(np.abs(exp3_adv_geo[:, -1]).mean())
    exp3_scaffold_feats = np.load(RESULTS_DIR / "scaffold" / "feature_names.npy").tolist()
    exp3_non_geo = [f for f in exp3_scaffold_feats if f not in ("UTM_X", "UTM_Y")]
    exp3_adv_trigger = float(np.abs(exp3_adv_geo[:, exp3_non_geo.index("pct_black_latino")]).mean())
    has_exp3 = True
else:
    has_exp3 = False

# Exp 10c values — GeoShapley geo_effect (last column of flat array)
exp10c_honest_geo_effect = float(np.abs(honest_geo[:, -1]).mean())
exp10c_adv_geo_effect = float(np.abs(adv_geo[:, -1]).mean())
exp10c_adv_trigger = float(np.abs(adv_geo[:, parcel_geo_features.index("in_floodplain")]).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("GeoShapley Reversal Test: High-I (pct_black_latino) vs Low-I (in_floodplain)\n"
             "geo_effect suppression under adversarial scaffold",
             fontsize=10, fontweight="bold")

# Left panel: Exp 1 / Exp 3 pct_black_latino
ax = axes[0]
if has_exp3:
    feats_left = ["geo_effect\n(honest)", "geo_effect\n(adv)", "bias trigger\n(adv)"]
    vals_left = [exp3_honest_geo_effect, exp3_adv_geo_effect, exp3_adv_trigger]
    bar_colors_left = ["#4e79a7", "#e15759", "#f28e2b"]
    bars = ax.bar(feats_left, vals_left, color=bar_colors_left, alpha=0.85, edgecolor="white")
    ax.set_title("Exp 1 — pct_black_latino bias\n(Moran's I ≈ 0.83, HIGH)", fontsize=10)
    for bar, val in zip(bars, vals_left):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0003, f"{val:.4f}",
                ha="center", fontsize=9, fontweight="bold")
    suppression_pct_left = (exp3_adv_geo_effect - exp3_honest_geo_effect) / (exp3_honest_geo_effect + 1e-9) * 100
    ax.set_ylim(0, max(vals_left) * 1.3)
    ax.text(0.5, 0.93, f"geo_effect suppression: {suppression_pct_left:+.0f}%",
            transform=ax.transAxes, ha="center", fontsize=9, color="#e15759",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", edgecolor="#e15759"))
else:
    ax.text(0.5, 0.5, "Exp 3 GeoShapley\nnot available",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Exp 1 — pct_black_latino bias\n(Moran's I ≈ 0.83, HIGH)", fontsize=10)
ax.set_ylabel("Mean |GeoShapley attribution|", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Right panel: Exp 10c in_floodplain
ax = axes[1]
feats_right = ["geo_effect\n(honest)", "geo_effect\n(adv)", "in_floodplain\n(adv)"]
vals_right = [exp10c_honest_geo_effect, exp10c_adv_geo_effect, exp10c_adv_trigger]
bar_colors_right = ["#4e79a7", "#e15759", "#f28e2b"]
bars = ax.bar(feats_right, vals_right, color=bar_colors_right, alpha=0.85, edgecolor="white")
ax.set_title("Exp 10c — in_floodplain bias\n(Moran's I = +0.051, LOW)", fontsize=10)
ax.set_ylabel("Mean |GeoShapley attribution|", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, vals_right):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0003, f"{val:.4f}",
            ha="center", fontsize=9, fontweight="bold")
suppression_pct_right = (exp10c_adv_geo_effect - exp10c_honest_geo_effect) / (exp10c_honest_geo_effect + 1e-9) * 100
ax.set_ylim(0, max(vals_right) * 1.3)
verdict = metrics.get("geoshapley_reversal_verdict", metrics.get("geoshap_reversal_verdict", ""))
verdict_short = "PARTIAL" if "PARTIAL" in verdict else ("CONFIRMED" if "CONFIRMED" in verdict else "NO REVERSAL")
ax.text(0.5, 0.93, f"geo_effect suppression: {suppression_pct_right:+.0f}% | {verdict_short}",
        transform=ax.transAxes, ha="center", fontsize=9, color="#e15759",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", edgecolor="#e15759"))

plt.tight_layout()
out_path = OUT_DIR / "fig_geoshapley_reversal.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")


# ── Figure 3: Sample comparison map (2-panel) ────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Dataset Coverage Comparison: Original 3k Sample vs Parcel-Scale Dataset\n"
             "Colored by in_floodplain (1 = FEMA SFHA, 100-yr flood zone)",
             fontsize=10, fontweight="bold")

# Panel 1: original 3k sample
ax = axes[0]
if has_flood_3k and "in_floodplain" in seattle_3k.columns:
    mask_flood_3k = seattle_3k["in_floodplain"] == 1
    mask_no_3k    = seattle_3k["in_floodplain"] == 0
    ax.scatter(seattle_3k.loc[mask_no_3k, "UTM_X"],
               seattle_3k.loc[mask_no_3k, "UTM_Y"],
               s=4, c="#94c4df", alpha=0.5, label="Not in flood zone")
    ax.scatter(seattle_3k.loc[mask_flood_3k, "UTM_X"],
               seattle_3k.loc[mask_flood_3k, "UTM_Y"],
               s=25, c="#e63946", alpha=0.9, label="In SFHA (100-yr)")
    n_pos = int(mask_flood_3k.sum())
    pct = n_pos / len(seattle_3k) * 100
    ax.set_title(f"Original seattle_sample_3k.csv\n(n={len(seattle_3k):,}, flood positives: {n_pos} = {pct:.1f}%)",
                 fontsize=9)
else:
    # No flood column in 3k — show sample without flood coloring
    ax.scatter(seattle_3k["UTM_X"], seattle_3k["UTM_Y"],
               s=4, c="#94c4df", alpha=0.5, label="No flood data available")
    ax.set_title(f"Original seattle_sample_3k.csv\n(n={len(seattle_3k):,}, flood data: N/A — 0.2% est. from Exp 10)",
                 fontsize=9)
ax.set_xlabel("UTM_X (m)", fontsize=8)
ax.set_ylabel("UTM_Y (m)", fontsize=8)
ax.legend(fontsize=7, markerscale=2)
ax.grid(alpha=0.2)
ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

# Panel 2: parcel dataset
ax = axes[1]
mask_flood_p = parcel_df["in_floodplain"] == 1
mask_no_p    = parcel_df["in_floodplain"] == 0
ax.scatter(parcel_df.loc[mask_no_p, "UTM_X"],
           parcel_df.loc[mask_no_p, "UTM_Y"],
           s=1, c="#94c4df", alpha=0.25, label="Not in flood zone",
           rasterized=True)
ax.scatter(parcel_df.loc[mask_flood_p, "UTM_X"],
           parcel_df.loc[mask_flood_p, "UTM_Y"],
           s=15, c="#e63946", alpha=0.9, label="In SFHA (100-yr)",
           rasterized=True)
n_pos_p = int(mask_flood_p.sum())
pct_p = n_pos_p / len(parcel_df) * 100
ax.set_title(f"KC Parcel Dataset (kc_housing_with_flood.csv)\n"
             f"(n={len(parcel_df):,}, flood positives: {n_pos_p} = {pct_p:.2f}%)",
             fontsize=9)
ax.set_xlabel("UTM_X (m)", fontsize=8)
ax.set_ylabel("UTM_Y (m)", fontsize=8)
ax.legend(fontsize=7, markerscale=2)
ax.grid(alpha=0.2)
ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

plt.tight_layout()
out_path = OUT_DIR / "fig_sample_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

print("\nAll three figures saved to:", OUT_DIR)
