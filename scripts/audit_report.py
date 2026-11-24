"""What a typical audit report would show, with and without the scaffold (feature design, King County).

A typical explanation-based fairness audit produces two things:
  1. a global importance summary (mean |attribution| per feature over the audited homes, as in a SHAP summary bar
     plot), read as "is the protected attribute among the most important features?";
  2. sometimes, a disaggregated check: the protected attribute's attribution on homes in the group of concern,
     read as "does the model systematically lower prices there?".
For every seed and explainer this script applies both checks to the biased model served directly and to the
matched scaffold, from the saved per-home attributions:
  summary check        flag if the trigger ranks in the top three features by mean |attribution| over all 200
                       explained homes
  disaggregated check  flag if the mean trigger attribution on homes in tracts above 0.25 is negative with a
                       one-sided t-test p < 0.05 over those homes; the material version also requires the mean to
                       be at least 0.05 in log10 price (an 11% price effect)
and draws, for one seed, the importance bars an auditor would see.

Writes results/audit_report/audit_checks.csv (one row per seed, explainer, deployment) and
docs/paper/figures/F10_audit_report.{pdf,png}.

  python scripts/audit_report.py [--figure-seed 0]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
COL = {"kernelshap": "#2a78d6", "lime": "#eb6834", "geoshapley": "#1baf7a"}
NAME = {"kernelshap": "KernelSHAP", "lime": "LIME", "geoshapley": "GeoShapley"}
LABEL = {"pct_black_latino": "Tract share Black+Latino", "sqft_living": "Living area", "grade": "Grade",
         "UTM_Y": "Northing", "UTM_X": "Easting", "age": "Age", "view": "View", "waterfront": "Waterfront",
         "bathrooms": "Bathrooms", "sqft_lot": "Lot area", "condition": "Condition", "GEO": "Location (GEO)"}
MATERIAL = 0.05     # log10 price: an 11% price effect, a size an auditor would act on
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-seed", type=int, default=0)
    a = ap.parse_args()
    res = ROOT / "results/matched_attack"
    rows, imp = [], {}
    for p in sorted(glob.glob(str(res / "feature_seed*.json"))):
        d = json.load(open(p))
        if (d["seed"] == 42 or not d.get("complete") or d.get("model", "xgb") != "xgb" or d.get("audit", "match") != "match"
                or abs(d.get("penalty", 0.3) - 0.3) > 1e-9 or d.get("shuffle", 1.0) != 1.0
                or d.get("attacker", "rf") != "rf" or d.get("harvest_scale", 1) != 1
                or d.get("penalty_shape", "step") != "step" or d.get("attacker_bg_kind", "sample") != "sample"):
            continue
        for e in NAME:
            for m in ["biased", "mat"]:
                z = res / f"{d['tag']}__{e}_{m}.npz"
                if not z.exists():
                    continue
                z = np.load(z)
                W, pen, names = z["flat"], z["penalised"].astype(bool), [str(n) for n in z["names"]]
                j = names.index("pct_black_latino")
                mean_abs = np.abs(W).mean(0)
                rank = int(1 + (np.delete(mean_abs, j) > mean_abs[j]).sum())
                t = stats.ttest_1samp(W[pen, j], 0.0, alternative="less")
                rows.append(dict(seed=d["seed"], explainer=e, deployment=m, rank_mean_abs=rank,
                                 summary_flag=rank <= 3, mean_pen=float(W[pen, j].mean()),
                                 p_disagg=float(t.pvalue), disagg_flag=bool(t.pvalue < 0.05),
                                 disagg_material_flag=bool(t.pvalue < 0.05 and W[pen, j].mean() <= -MATERIAL),
                                 share_of_total_abs=float(mean_abs[j] / mean_abs.sum())))
                if d["seed"] == a.figure_seed:
                    imp[(e, m)] = pd.Series(mean_abs, index=names)
    R = pd.DataFrame(rows)
    out = ROOT / "results/audit_report"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "audit_checks.csv", index=False)
    print(R.groupby(["explainer", "deployment"])[["summary_flag", "disagg_flag", "disagg_material_flag", "rank_mean_abs",
                                               "mean_pen"]].mean().round(3))

    # figure: the importance bars an auditor would see (one seed), biased model vs matched scaffold
    exps = [e for e in ["kernelshap", "geoshapley", "lime"] if (e, "biased") in imp and (e, "mat") in imp]
    fig, axes = plt.subplots(1, len(exps), figsize=(3.0 * len(exps) + 0.6, 3.6), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, exps):
        b, s = imp[(e, "biased")], imp[(e, "mat")]
        order = b.sort_values(ascending=False).index[:8][::-1]
        y = np.arange(len(order))
        ax.barh(y + 0.2, b[order].values, height=0.38, color="#c9c8c3", label="Biased model, no scaffold")
        cols = [COL[e] if n == "pct_black_latino" else "#8a8984" for n in order]
        ax.barh(y - 0.2, s[order].values, height=0.38, color=cols, label="Matched scaffold")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(n, n) for n in order], fontsize=8)
        for lab in ax.get_yticklabels():
            if lab.get_text() == LABEL["pct_black_latino"]:
                lab.set_fontweight("bold")
        rb = int(1 + (b.drop("pct_black_latino") > b["pct_black_latino"]).sum())
        rs = int(1 + (s.drop("pct_black_latino") > s["pct_black_latino"]).sum())
        ax.set_title(f"{NAME[e]}: trigger rank {rb} → {rs}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Mean |attribution|")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fo = ROOT / "docs/paper/figures"
    fig.savefig(fo / "F10_audit_report.pdf", bbox_inches="tight")
    fig.savefig(fo / "F10_audit_report.png", bbox_inches="tight")
    print("figure seed", a.figure_seed)


if __name__ == "__main__":
    main()
