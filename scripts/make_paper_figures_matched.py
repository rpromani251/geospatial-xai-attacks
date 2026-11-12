"""Figures F1-F8 for the draft, from the CSVs and JSONs (run make_paper_tables.py first: it writes the
summary CSVs this script reads). Saves PDF and PNG to docs/paper/figures/.

Colour: one fixed categorical slot per explainer in every figure (validated palette, three slots pass the
all-pairs colour-vision checks): KernelSHAP blue, LIME orange, GeoShapley aqua; every series also has its own
marker and a direct label or legend, so identity never rests on colour alone. Maps use a single-hue blue ramp.

  python scripts/make_paper_figures_matched.py [--results results]
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
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
COL = {"kernelshap": "#2a78d6", "lime": "#eb6834", "geoshapley": "#1baf7a"}
MRK = {"kernelshap": "o", "lime": "s", "geoshapley": "^"}
NAME = {"kernelshap": "KernelSHAP", "lime": "LIME", "geoshapley": "GeoShapley"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 2.0})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def summ(D, sfx):
    p = D / f"summary_{sfx}.csv"
    return pd.read_csv(p) if p.exists() else None


def get(S, e, m, k):
    r = S[(S.explainer == e) & (S.deployment == m)]
    if r.empty or k not in r:
        return np.nan, np.nan
    return float(r[k].iloc[0]), float(r.get(k + "_ci", pd.Series([np.nan])).iloc[0])


def f1_map(out):
    g = json.load(open(ROOT / "data/processed/king_county_tracts_demographics.geojson"))
    polys, vals = [], []
    for ft in g["features"]:
        v = ft["properties"].get("pct_black_latino")
        geom = ft["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
        for r in rings:
            polys.append(np.asarray(r)[:, :2])
            vals.append(np.nan if v is None else v)
    vals = np.asarray(vals, float)
    import pyproj
    df = pd.read_csv(ROOT / "data/processed/seattle_housing_with_demographics.csv")
    tr = pyproj.Transformer.from_crs("EPSG:32610", "EPSG:4269", always_xy=True)
    lon, lat = tr.transform(df.UTM_X.values, df.UTM_Y.values)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    pc = PolyCollection(polys, array=np.nan_to_num(vals), cmap=BLUES, edgecolors="#ffffff", linewidths=0.2)
    pc.set_clim(0, max(0.6, np.nanmax(vals)))
    ax.add_collection(pc)
    hot = [p for p, v in zip(polys, vals) if np.isfinite(v) and v > 0.25]
    ax.add_collection(PolyCollection(hot, facecolors="none", edgecolors=INK, linewidths=0.7))
    pen = df.pct_black_latino.values > 0.25
    ax.scatter(lon[~pen], lat[~pen], s=2, c="#8a8984", lw=0, label="Sale, tract share ≤ 0.25")
    ax.scatter(lon[pen], lat[pen], s=3, c="#e34948", lw=0, label="Sale, tract share > 0.25 (penalised)")
    pad = 0.03
    ax.set_xlim(lon.min() - pad, lon.max() + pad)
    ax.set_ylim(lat.min() - pad, lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.grid(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(pc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Tract share of Black and Latino residents (ACS 2011–2015)")
    cb.outline.set_visible(False)
    ax.legend(loc="lower left", markerscale=4, fontsize=7.5)
    ax.set_title("King County sales and tract demographics; black outline: tract share above 0.25", fontsize=9,
                 color=INK, loc="left")
    save(fig, out, "F1_study_area")


def f2_main(D, out):
    S = summ(D, "feature")
    if S is None:
        return
    deps = [("biased", "Biased model,\nno scaffold"), ("gen", "Scaffold,\ngeneric detector"),
            ("mat", "Scaffold,\nmatched detector")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, e in enumerate(["lime", "kernelshap", "geoshapley"]):
        xs, ys, es = [], [], []
        for j, (m, _) in enumerate(deps):
            mu, hw = get(S, e, m, "penalty_share_kept")
            xs.append(j + (i - 1) * 0.18)
            ys.append(mu)
            es.append(0 if not np.isfinite(hw) else hw)
        ax.errorbar(xs, ys, yerr=es, fmt=MRK[e], color=COL[e], ms=7, mec="#fcfcfb", mew=1.5, capsize=0,
                    elinewidth=1.5, label=NAME[e], ls="none")
    ax.set_xticks(range(3))
    ax.set_xticklabels([d[1] for d in deps])
    ax.set_ylabel("Penalty share kept\n(1 = fully visible)")
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, out, "F2_main")


def f3_frontier(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, design in zip(axes, ["feature", "location"]):
        S = summ(D, design)
        if S is None:
            ax.set_visible(False)
            continue
        kk = "penalty_share_kept_excess" if design == "location" else "penalty_share_kept"
        for e in ["kernelshap", "geoshapley"]:
            pts = [(get(S, e, m, "fidelity"), get(S, e, m, kk)) for m in ["mat", "mat75", "mat50"]]
            pts = [p for p in pts if np.isfinite(p[0][0]) and np.isfinite(p[1][0])]
            if pts:
                x = [p[0][0] for p in pts]
                y = [p[1][0] for p in pts]
                ax.plot(x, y, color=COL[e], lw=1.5, zorder=2)
                ax.errorbar(x, y, xerr=[np.nan_to_num(p[0][1]) for p in pts], yerr=[np.nan_to_num(p[1][1]) for p in pts],
                            fmt=MRK[e], color=COL[e], ms=7, mec="#fcfcfb", mew=1.5, elinewidth=1.2, ls="none",
                            label=f"{NAME[e]}, matched (90/75/50%)", zorder=3)
            (gx, gxh), (gy, gyh) = get(S, e, "gen", "fidelity"), get(S, e, "gen", kk)
            if np.isfinite(gx):
                ax.errorbar([gx], [gy], xerr=[np.nan_to_num(gxh)], yerr=[np.nan_to_num(gyh)], fmt="none",
                            ecolor=COL[e], elinewidth=1.2, zorder=2)
                ax.plot([gx], [gy], MRK[e], mfc="#fcfcfb", mec=COL[e], ms=7, mew=1.6, ls="none",
                        label=f"{NAME[e]}, generic", zorder=4)
        (lx, lxh), (ly, lyh) = get(S, "lime", "mat", "fidelity"), get(S, "lime", "mat", kk)
        if np.isfinite(lx):
            ax.errorbar([lx], [ly], xerr=[np.nan_to_num(lxh)], yerr=[np.nan_to_num(lyh)], fmt=MRK["lime"],
                        color=COL["lime"], ms=7, mec="#fcfcfb", mew=1.5, label="LIME, matched", zorder=3)
        ax.set_xlabel("Fidelity: real homes served by $f$")
        ax.set_title({"feature": "Feature design", "location": "Location design (net of honest)"}[design],
                     loc="left", fontsize=9, color=INK)
        ax.set_xlim(0.3, 1.02)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("Penalty share kept")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=7.5)
    save(fig, out, "F3_frontier")


def f4_geo(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    deps = ["honest", "biased", "gen", "mat"]
    lab = {"honest": "Honest", "biased": "Biased", "gen": "Generic\nscaffold", "mat": "Matched\nscaffold"}
    mk = {"feature": "o", "location": "s"}
    shade = {"feature": "#1baf7a", "location": "#0b6b4a"}
    for ax, (k, ttl) in zip(axes, [("geo_signed_pen", "GEO effect on penalised homes"),
                                   ("intera_abs_sum", "Sum of |interaction effects| per home")]):
        for i, design in enumerate(["feature", "location"]):
            S = summ(D, design)
            if S is None or k not in S:
                continue
            xs, ys, es = [], [], []
            for j, m in enumerate(deps):
                mu, hw = get(S, "geoshapley", m, k)
                if np.isfinite(mu):
                    xs.append(j + (i - 0.5) * 0.2)
                    ys.append(mu)
                    es.append(np.nan_to_num(hw))
            ax.errorbar(xs, ys, yerr=es, fmt=mk[design], color=shade[design], ms=6.5, mec="#fcfcfb", mew=1.4,
                        ls="none", label=f"{design.capitalize()} design")
        ax.set_xticks(range(len(deps)))
        ax.set_xticklabels([lab[d] for d in deps])
        ax.set_title(ttl, loc="left", fontsize=9, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
    axes[0].legend(fontsize=7.5)
    save(fig, out, "F4_geoshapley_decomposition")


def f5_e7(R, out):
    p = R / "residual_diagnostic/summary.csv"
    if not p.exists():
        return
    E = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for mdl, lab, c, m in [("biased", "Biased model", "#52514e", "o"),
                           ("scaffold_lime_matched", "Scaffold (LIME-matched)", "#6250d6", "D")]:
        q = E[E.model == mdl].sort_values("penalty")
        ax.errorbar(q.penalty, q.auc_lag, yerr=q.auc_lag_ci, fmt=m + "-", color=c, ms=6, mec="#fcfcfb", mew=1.3,
                    label=lab)
    h = E[E.model == "honest"]
    if not h.empty:
        ax.axhline(float(h.auc_lag.iloc[0]), color="#8a8984", lw=1, ls=(0, (1, 0)))
        ax.annotate("honest model", (0.3, float(h.auc_lag.iloc[0])), xytext=(-2, 4), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK2)
    ax.axhline(0.8, color=GRID, lw=1)
    ax.set_xlabel("Penalty (log$_{10}$ price)")
    ax.set_ylabel("AUC of local mean residual\nfor the penalised homes")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, out, "F5_outcome_check")


def f6_autocorr(D, R, out):
    rows = []
    for lam in ["", "_lam0.5", "_lam0"]:
        S = summ(D, f"feature{lam}")
        if S is None:
            continue
        js = [json.load(open(p)) for p in glob.glob(str(R / f"matched_attack/feature_seed[0-9]{lam}.json"))]
        js = [j for j in js if j.get("model", "xgb") == "xgb" and j.get("audit", "match") == "match"
              and j.get("penalty", 0.3) == 0.3 and j.get("attacker", "rf") == "rf"]
        if not js:
            continue
        I = float(np.mean([j["trigger_morans_I"] for j in js]))
        for e in ["lime", "kernelshap", "geoshapley"]:
            for m in ["gen", "mat"]:
                mu, hw = get(S, e, m, "leak")
                rows.append(dict(I=I, e=e, m=m, mu=mu, hw=hw))
    if len({r["I"] for r in rows}) < 2:
        return
    T = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, m, ttl in zip(axes, ["gen", "mat"], ["Generic detector", "Matched detector"]):
        for e in ["lime", "kernelshap", "geoshapley"]:
            q = T[(T.e == e) & (T.m == m)].sort_values("I")
            ax.errorbar(q.I, q.mu, yerr=q.hw.fillna(0), fmt=MRK[e] + "-", color=COL[e], ms=6, mec="#fcfcfb",
                        mew=1.3, label=NAME[e])
        ax.set_title(ttl, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Moran's $I$ of the trigger")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Explainer queries reaching $f$")
    axes[0].legend(fontsize=7.5)
    save(fig, out, "F6_autocorrelation")


def f7_mechanism(R, out):
    files = sorted(glob.glob(str(R / "mechanism/routing_by_size_*.csv")))
    files = [p for p in files if "_m-" not in p]
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    ls = {"taustar": "-", "fid75": "--", "fid50": ":"}
    for p in files:
        key = Path(p).stem.replace("routing_by_size_", "")
        design, e = key.split("_")[:2]
        if design != "feature":
            continue
        Rb = pd.read_csv(p)
        g = Rb.groupby(["tau_name", "k_with_trigger"]).rho_with.mean().reset_index()
        for tn in ["taustar", "fid75", "fid50"]:
            q = g[g.tau_name == tn]
            ax.plot(q.k_with_trigger, q.rho_with, ls=ls[tn], color=COL[e], marker=MRK[e], ms=4, lw=1.5,
                    label=f"{NAME[e]}, {dict(taustar='90%', fid75='75%', fid50='50%')[tn]}")
    ax.set_xlabel("Coalition size $k$ including the trigger\n($k = M$: the explained home itself)")
    ax.set_ylabel("Share of hybrids routed to $f$")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, ncol=2, loc="lower center")
    ax.set_title("Routing by coalition size (feature design)", loc="left", fontsize=9, color=INK)
    ax = axes[1]
    for p in sorted(glob.glob(str(R / "mechanism/predicted_vs_measured_*.csv"))):
        if "_m-" in p:
            continue
        key = Path(p).stem.replace("predicted_vs_measured_", "")
        design, e = key.split("_")[:2]
        C = pd.read_csv(p)
        if "measured_share_kept" not in C:
            continue
        ax.scatter(C.measured_share_kept, C.predicted_share_kept, marker=MRK[e] if design == "feature" else "D",
                   facecolors=COL[e] if design == "feature" else "none", edgecolors=COL[e], s=28, lw=1.2,
                   label=f"{NAME[e]}, {design}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Measured penalty share kept")
    ax.set_ylabel("Predicted from routing rates")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Prediction against measurement", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, out, "F7_mechanism")


def f8_robust(D, out):
    cfg = [("Main", "feature"), ("Random forest", "feature_m-rf"), ("Neural net", "feature_m-mlp"),
           ("Background 25", "feature_aud-bg25"), ("Background 200", "feature_aud-bg200"),
           ("k-means background", "feature_aud-bg100kmeans"), ("LIME 5,000 samples", "feature_aud-lime5000"),
           ("LIME Gaussian sampler", "feature_aud-lime500gauss"), ("Attacker: forest leaf 5", "feature_att-rf_leaf5"),
           ("Attacker: XGBoost", "feature_att-xgb"), ("Attacker: 4x queries", "feature_hs4")]
    rows = []
    for nm, sfx in cfg:
        S = summ(D, sfx)
        if S is None:
            continue
        for e in ["lime", "kernelshap", "geoshapley"]:
            mu, hw = get(S, e, "mat", "penalty_share_kept")
            if np.isfinite(mu):
                rows.append((nm, e, mu, hw))
    if not rows:
        return
    names = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(names) + 1.0))
    off = {"lime": -0.22, "kernelshap": 0.0, "geoshapley": 0.22}
    for nm, e, mu, hw in rows:
        y = len(names) - 1 - names.index(nm) + off[e]
        ax.errorbar([mu], [y], xerr=[np.nan_to_num(hw)], fmt=MRK[e], color=COL[e], ms=6, mec="#fcfcfb", mew=1.3,
                    elinewidth=1.3, label=NAME[e])
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.9 / len(names)),
              fontsize=7.5)
    ax.axvline(0.10, color=INK2, lw=0.8)
    ax.annotate("fully fooled\nbelow 0.10", (0.10, len(names) - 0.6), xytext=(3, 0), textcoords="offset points",
                fontsize=7, color=INK2, va="top")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Penalty share kept under the matched scaffold")
    ax.grid(axis="y", visible=False)
    save(fig, out, "F8_robustness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    R = Path(a.results)
    D = R / "matched_attack"
    out = ROOT / "docs/paper/figures"
    out.mkdir(parents=True, exist_ok=True)
    f1_map(out)
    f2_main(D, out)
    f3_frontier(D, out)
    f4_geo(D, out)
    f5_e7(R, out)
    f6_autocorr(D, R, out)
    f7_mechanism(R, out)
    f8_robust(D, out)
    print("figures in", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
