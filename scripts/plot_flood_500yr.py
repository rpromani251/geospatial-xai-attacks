"""
Generate three figures for Experiment 10b — Extended Floodplain (500yr).

Figures saved to results/scaffold_flood_500yr/figures/:
  fig_attribution_comparison.png  — honest vs adv bar chart (3-panel: SHAP/LIME/GeoShapley)
  fig_geoshapley_reversal.png     — GeoShapley geo_effect suppression: pct_black_latino vs in_flood_500yr
  fig_coverage_map.png            — scatter plot colored by in_flood_500yr

Run from repo root:
    python3 scripts/plot_flood_500yr.py
"""

import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR, PROCESSED_DATA_DIR

# ── Load data ────────────────────────────────────────────────────────────────

out_dir_500 = RESULTS_DIR / "scaffold_flood_500yr"
out_dir_100 = RESULTS_DIR / "scaffold_flood_zone"
expl_dir = RESULTS_DIR / "explanations"

# 500yr results
adv_shap_500 = np.load(out_dir_500 / "adv_shap_values.npy")
adv_lime_500 = np.load(out_dir_500 / "adv_lime_values.npy")
adv_geo_500  = np.load(out_dir_500 / "adv_geoshapley_flat.npy")
with open(out_dir_500 / "metrics.json") as f:
    metrics_500 = json.load(f)

# Honest baseline (10-feature)
honest_shap = np.load(expl_dir / "shap_values.npy")
honest_lime = np.load(expl_dir / "lime_values.npy")
honest_features = np.load(expl_dir / "feature_names.npy").tolist()

# Honest GeoShapley
honest_geo_path = expl_dir / "geoshapley_flat.npy"
honest_geo = np.load(honest_geo_path) if honest_geo_path.exists() else None

# 100yr GeoShapley for comparison panel
adv_geo_100 = np.load(out_dir_100 / "adv_geoshapley_flat.npy")

# Exp 3 (pct_black_latino) GeoShapley
geo_pct_path = RESULTS_DIR / "scaffold" / "adv_geoshapley_flat.npy"
if geo_pct_path.exists():
    adv_geo_pct = np.load(geo_pct_path)
    pct_scaffold_feats = np.load(RESULTS_DIR / "scaffold" / "feature_names.npy").tolist()
    pct_non_geo = [f for f in pct_scaffold_feats if f not in ("UTM_X", "UTM_Y")]
    pct_geo_features = pct_non_geo + ["geo_effect"]
else:
    adv_geo_pct = None
    pct_geo_features = None

# 500yr feature list (11 features, SHAP/LIME arrays)
feat_500 = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "in_flood_500yr", "UTM_X", "UTM_Y",
]

# GeoShapley flat feature lists
non_geo_honest = [f for f in honest_features if f not in ("UTM_X", "UTM_Y")]
honest_geo_features = non_geo_honest + ["geo_effect"]
non_geo_500 = [f for f in feat_500 if f not in ("UTM_X", "UTM_Y")]
feat_500_geo = non_geo_500 + ["geo_effect"]

# Focus features for attribution comparison
focus_features = ["in_flood_500yr", "UTM_Y", "UTM_X", "sqft_living", "grade", "age", "geo_effect"]
focus_display  = ["in_flood_500yr", "UTM_Y", "UTM_X", "sqft_living", "grade", "age", "geo_effect"]

# Indices in the 11-feature 500yr arrays
def idx(feat, feat_list):
    return feat_list.index(feat) if feat in feat_list else None

# ── Style ────────────────────────────────────────────────────────────────────

COLOR_HONEST = "#2c7bb6"
COLOR_ADV    = "#d7191c"
ALPHA        = 0.85
FIGSIZE_W    = 14
PANEL_H      = 4.0
FONT_TITLE   = 12
FONT_TICK    = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig_dir = out_dir_500 / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Attribution Comparison (3-panel) ───────────────────────────────

fig1, axes = plt.subplots(1, 3, figsize=(FIGSIZE_W, PANEL_H), sharey=False)
fig1.suptitle(
    "Experiment 10b — Adversarial Scaffold Attribution\n"
    "(in_flood_500yr trigger, 100yr SFHA + 500yr Zone X Shaded)",
    fontsize=FONT_TITLE, fontweight="bold", y=1.01,
)

panel_data = [
    ("SHAP",       adv_shap_500, honest_shap, feat_500,     honest_features),
    ("LIME",       adv_lime_500, honest_lime,  feat_500,     honest_features),
    ("GeoShapley", adv_geo_500,  honest_geo,   feat_500_geo, honest_geo_features),
]

for ax, (label, adv_arr, hon_arr, adv_feats, hon_feats) in zip(axes, panel_data):
    x = np.arange(len(focus_features))
    width = 0.35

    honest_vals = []
    adv_vals = []
    for feat in focus_features:
        if hon_arr is not None and feat in hon_feats:
            hi = hon_feats.index(feat)
            honest_vals.append(np.abs(hon_arr[:, hi]).mean())
        else:
            honest_vals.append(0.0)
        ai = idx(feat, adv_feats)
        if ai is not None:
            adv_vals.append(np.abs(adv_arr[:, ai]).mean())
        else:
            adv_vals.append(0.0)

    bars_h = ax.bar(x - width/2, honest_vals, width, color=COLOR_HONEST, alpha=ALPHA, label="Honest")
    bars_a = ax.bar(x + width/2, adv_vals,    width, color=COLOR_ADV,    alpha=ALPHA, label="Adversarial")

    ax.set_title(label, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(focus_display, rotation=35, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Mean |attribution|", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig1.tight_layout()
path1 = fig_dir / "fig_attribution_comparison.png"
fig1.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[SAVE] {path1}")

# ── Figure 2: GeoShapley Reversal — geo_effect suppression comparison ────────
# Left panel: pct_black_latino bias (Exp 3, high-I)
# Right panel: in_flood_500yr bias (Exp 10b, low-I)

has_pct = (adv_geo_pct is not None) and (pct_geo_features is not None)

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.5))
fig2.suptitle(
    "GeoShapley geo_effect Suppression: High-I vs Low-I Bias Feature (Acid Test)",
    fontsize=FONT_TITLE, fontweight="bold",
)

# Honest geo_effect from GeoShapley
if honest_geo is not None:
    geo_effect_honest = float(np.abs(honest_geo[:, -1]).mean())
else:
    geo_effect_honest = 0.0

# Panel left: pct_black_latino (Exp 3, I=0.834)
ax_l = axes2[0]
if has_pct:
    geo_effect_pct_adv = float(np.abs(adv_geo_pct[:, -1]).mean())
    bars_l = ax_l.bar(
        ["Honest", "Adversarial"],
        [geo_effect_honest, geo_effect_pct_adv],
        color=[COLOR_HONEST, COLOR_ADV], alpha=ALPHA, width=0.5
    )
    ax_l.set_title("Exp 3 — pct_black_latino\n(Moran's I ≈ 0.834, high)", fontsize=10)
    suppression_pct = (geo_effect_honest - geo_effect_pct_adv) / (geo_effect_honest + 1e-9) * 100
    ax_l.text(0.5, 0.92, f"geo_effect suppression: {suppression_pct:+.0f}%",
              ha="center", va="top", transform=ax_l.transAxes,
              fontsize=9, color="darkred")
else:
    ax_l.text(0.5, 0.5, "Exp 3 GeoShapley\nnot available",
              ha="center", va="center", transform=ax_l.transAxes, fontsize=10)
    ax_l.set_title("Exp 3 — pct_black_latino\n(Moran's I ≈ 0.834, high)", fontsize=10)
ax_l.set_ylabel("Mean |GeoShapley| geo_effect")
ax_l.set_ylim(0, max(geo_effect_honest * 1.3, 0.01))
ax_l.grid(axis="y", alpha=0.3)

# Panel right: in_flood_500yr (Exp 10b, low-I)
ax_r = axes2[1]
geo_effect_500_adv = float(np.abs(adv_geo_500[:, -1]).mean())
bars_r = ax_r.bar(
    ["Honest", "Adversarial"],
    [geo_effect_honest, geo_effect_500_adv],
    color=[COLOR_HONEST, COLOR_ADV], alpha=ALPHA, width=0.5
)
ax_r.set_title("Exp 10b — in_flood_500yr\n(Moran's I = +0.239, low)", fontsize=10)
suppression_500 = (geo_effect_honest - geo_effect_500_adv) / (geo_effect_honest + 1e-9) * 100
ax_r.text(0.5, 0.92, f"geo_effect suppression: {suppression_500:+.0f}%",
          ha="center", va="top", transform=ax_r.transAxes,
          fontsize=9, color="darkred")
ax_r.set_ylabel("Mean |GeoShapley| geo_effect")
ax_r.set_ylim(0, max(geo_effect_honest * 1.3, 0.01))
ax_r.grid(axis="y", alpha=0.3)

fig2.tight_layout()
path2 = fig_dir / "fig_geoshapley_reversal.png"
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[SAVE] {path2}")

# ── Figure 3: Coverage Map ────────────────────────────────────────────────────

df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_flood_500yr.csv", index_col=0)

fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))

# Background: all housing points (grey)
ax3.scatter(
    df[df["in_flood_500yr"] == 0]["UTM_X"],
    df[df["in_flood_500yr"] == 0]["UTM_Y"],
    s=4, c="lightgrey", alpha=0.5, label="Outside flood zone",
)
# Positives: in_flood_500yr = 1 (red)
ax3.scatter(
    df[df["in_flood_500yr"] == 1]["UTM_X"],
    df[df["in_flood_500yr"] == 1]["UTM_Y"],
    s=40, c=COLOR_ADV, alpha=0.9, zorder=5, label="In 500yr flood zone",
    edgecolors="darkred", linewidths=0.5,
)

n_pos = df["in_flood_500yr"].sum()
n_total = len(df)
ax3.set_title(
    f"Housing Points by 500-yr Flood Zone Status\n"
    f"({n_pos}/{n_total} = {100*n_pos/n_total:.1f}% positive — 100yr SFHA + 500yr Zone X Shaded)",
    fontsize=11,
)
ax3.set_xlabel("UTM_X (m, Zone 10N)", fontsize=9)
ax3.set_ylabel("UTM_Y (m, Zone 10N)", fontsize=9)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.2)

# Format tick labels in km
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

fig3.tight_layout()
path3 = fig_dir / "fig_coverage_map.png"
fig3.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[SAVE] {path3}")

print("\nAll figures saved.")
print(f"Directory: {fig_dir}")
