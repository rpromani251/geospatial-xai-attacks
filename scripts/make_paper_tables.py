"""Build every table, every number cited in prose, and the decision-rule outcomes for the draft.

Reads the attack JSONs and the CSVs written by summarize_matched_attack.py, run_residual_diagnostic.py and
mechanism_routing.py. Writes
  docs/paper/tables/*.tex        one LaTeX table fragment per table
  docs/paper/tables/numbers.tex  \\newcommand macros for numbers quoted in the text
  docs/paper/data/*.csv          copies of every CSV a table draws on, plus decisions.csv
Nothing in the paper is typed by hand: prose uses the macros, tables are \\input.

  python scripts/make_paper_tables.py [--results results]
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXPL = ["lime", "kernelshap", "geoshapley"]
ENAME = {"lime": "LIME", "kernelshap": "KernelSHAP", "geoshapley": "GeoShapley"}
DNAME = {"honest": "Honest model", "biased": "Biased model, no scaffold", "gen": "Scaffold, generic detector",
         "mat": "Scaffold, matched detector (90\\%)", "mat75": "Matched detector, 75\\% fidelity",
         "mat50": "Matched detector, 50\\% fidelity"}
MACROS = {}
MNAME = {"honest": "Honest", "biased": "Biased", "gen": "Gen", "mat": "Mat", "mat75": "MatSeventyFive",
         "mat50": "MatFifty"}
PNAME = {0.02: "Two", 0.05: "Five", 0.10: "Ten", 0.20: "Twenty", 0.30: "Thirty"}


def tq(n):
    return stats.t.ppf(0.975, n - 1) if n > 1 else np.nan


def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, 1
    return float(x.mean()), float(tq(len(x)) * x.std(ddof=1) / np.sqrt(len(x))), len(x)


def f(mu, hw=np.nan, d=2, pct=False, sign=False):
    if mu is None or not np.isfinite(mu):
        return "--"
    k = 100.0 if pct else 1.0
    d = 1 if pct else d
    s = f"{mu * k:+.{d}f}" if sign else f"{mu * k:.{d}f}"
    if float(s) == 0:                       # no "-0.00"
        s = f"{0:.{d}f}"
    s = s.replace("-", "$-$")
    if hw is not None and np.isfinite(hw):
        s += f" $\\pm$ {hw * k:.{d}f}"
    return s


def macro(name, value, d=2, pct=False, sign=False):
    """Register \\name -> formatted value (no interval)."""
    MACROS[name] = f(value, d=d, pct=pct, sign=sign) if not isinstance(value, str) else value


def macro_ci(name, mu, hw, d=2, pct=False, sign=False):
    """\\name = mean, \\nameLo / \\nameHi = 95% interval ends."""
    macro(name, mu, d, pct, sign)
    macro(name + "Lo", mu - hw if np.isfinite(hw) else np.nan, d, pct, sign)
    macro(name + "Hi", mu + hw if np.isfinite(hw) else np.nan, d, pct, sign)


def wrap(h, width=13):
    """Header cell on several lines (makecell), breaking at spaces."""
    words, lines, cur = h.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\\makecell[b]{" + " \\\\ ".join(lines) + "}" if len(lines) > 1 else h


def table(path, cols, rows, caption, label, colspec=None, notes=None):
    colspec = colspec or ("l" * 2 + "r" * (len(cols) - 2))
    out = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
           "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(wrap(c) for c in cols) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\caption{{{caption}}}",
            f"\\label{{{label}}}"]
    if notes:
        out.insert(-2, f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {notes}}}")
    out.append("\\end{table}")
    Path(path).write_text("\n".join(out) + "\n")


def summarize(res, *extra):
    """Run summarize_matched_attack.py and return (per_seed, summary) or (None, None)."""
    cmd = [sys.executable, str(ROOT / "scripts/summarize_matched_attack.py"), "--dir", str(res), *extra]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    a = dict(zip(extra[::2], extra[1::2]))
    sfx = a.get("--design", "feature")
    if a.get("--penalty") not in (None, "0.3", "0.30"):
        sfx += f"_pen{float(a['--penalty']):g}"
    if a.get("--attacker") not in (None, "rf"):
        sfx += f"_att-{a['--attacker']}"
    if a.get("--harvest-scale") not in (None, "1"):
        sfx += f"_hs{a['--harvest-scale']}"
    if a.get("--shuffle") not in (None, "1.0", "1"):
        sfx += f"_lam{float(a['--shuffle']):g}"
    if a.get("--model") not in (None, "xgb"):
        sfx += f"_m-{a['--model']}"
    if a.get("--audit") not in (None, "match"):
        sfx += f"_aud-{a['--audit']}"
    p, s = Path(res) / f"per_seed_{sfx}.csv", Path(res) / f"summary_{sfx}.csv"
    if not p.exists():
        return None, None
    return pd.read_csv(p), pd.read_csv(s), sfx


def runs(res, design, **flt):
    out = []
    for p in sorted(glob.glob(f"{res}/{design}_seed*.json")):
        d = json.load(open(p))
        if d["seed"] == 42 or not d.get("complete"):     # partial runs never enter a table or a rule
            continue
        ok = all(d.get(k, dflt) == v for k, (v, dflt) in
                 {"penalty": (flt.get("penalty", 0.30), 0.30), "model": (flt.get("model", "xgb"), "xgb"),
                  "audit": (flt.get("audit", "match"), "match"), "attacker": (flt.get("attacker", "rf"), "rf"),
                  "shuffle": (flt.get("shuffle", 1.0), 1.0), "harvest_scale": (1, 1)}.items())
        if ok:
            out.append(d)
    return out


def cell(S, e, m, k):
    r = S[(S.explainer == e) & (S.deployment == m)]
    if r.empty or k not in r:
        return np.nan, np.nan
    return float(r[k].iloc[0]), float(r[k + "_ci"].iloc[0]) if k + "_ci" in r else np.nan


def paired(P, e1, m1, e2, m2, k):
    a = P[(P.explainer == e1) & (P.deployment == m1)].set_index("seed")[k]
    b = P[(P.explainer == e2) & (P.deployment == m2)].set_index("seed")[k]
    d = (a - b).dropna()
    mu, hw, n = ci(d.values)
    p = float(stats.ttest_1samp(d.values, 0).pvalue) if n > 1 else np.nan
    return mu, hw, n, p


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    R = Path(a.results)
    res = R / "matched_attack"
    T = ROOT / "docs/paper/tables"
    D = ROOT / "docs/paper/data"
    T.mkdir(parents=True, exist_ok=True)
    D.mkdir(parents=True, exist_ok=True)
    decisions = []

    # ------------------------------------------------------------------ data and models
    rows = []
    for design in ["feature", "location"]:
        rr = runs(res, design)
        if not rr:
            continue
        hr, hw_, n = ci([d["honest_r2"] for d in rr])
        pr, pw, _ = ci([d["psi_r2"] for d in rr])
        sp, sw, _ = ci([d["share_test_penalised"] for d in rr])
        ne, nw, _ = ci([d["n_explained_penalised"] for d in rr])
        mi = rr[0].get("trigger_morans_I")
        row = [design.capitalize(), str(n), f(hr, hw_), f(pr, pw), f(sp, sw, pct=True), f(ne, nw, d=1)]
        if design == "location":
            ra, rw, _ = ci([d.get("raster_agrees_with_csv_tract", np.nan) for d in rr])
            row.append(f(ra, rw, pct=True))
            macro("RasterAgree", f"{ra * 100:.2f}")      # 1 decimal would print 100.0
            macro("RasterAgreeLo", max(0.0, ra - rw) if np.isfinite(rw) else np.nan, pct=True)
            macro("RasterAgreeHi", min(1.0, ra + rw) if np.isfinite(rw) else np.nan, pct=True)
        else:
            row.append("--")
            macro("TriggerMoran", mi, d=2)
        rows.append(row)
        tag = design.capitalize()
        macro_ci(f"HonestRtwo{tag}", hr, hw_)
        macro_ci(f"PsiRtwo{tag}", pr, pw)
        macro_ci(f"SharePen{tag}", sp, sw, pct=True)
        macro(f"NPen{tag}", ne, d=0)
        macro(f"Seeds{tag}", str(n))
    if rows:
        table(T / "tab_data.tex", ["Design", "Seeds", "Honest $R^2$", "Innocuous $R^2$", "Test homes penalised (\\%)",
                                   "Penalised among 200 explained", "Raster = tract file (\\%)"], rows,
              "Data and models per design, mean $\\pm$ 95\\% interval over seeds. $R^2$ on the 600 test homes. "
              "The raster column checks that the model's coordinate lookup puts each test home in the same tract "
              "as the spatial join in the data file.", "tab:data", colspec="lrrrrrr")

    # ------------------------------------------------------------------ main tables
    P = {}
    S = {}
    for design in ["feature", "location", "nocoord"]:
        out = summarize(res, "--design", design)
        if out[0] is None:
            continue
        P[design], S[design], sfx = out
        shutil.copy(res / f"per_seed_{sfx}.csv", D)
        shutil.copy(res / f"summary_{sfx}.csv", D)

    def main_table(design, label, deps):
        if design not in S:
            return
        Sd = S[design]
        rows = []
        for e in EXPL:
            if e not in set(Sd.explainer):
                continue
            for m in deps:
                if Sd[(Sd.explainer == e) & (Sd.deployment == m)].empty:
                    continue
                top3, top3h = cell(Sd, e, m, "top3")
                tp, tph = cell(Sd, e, m, "top3_pen")
                kp, kph = cell(Sd, e, m, "penalty_share_kept")
                fid, fidh = cell(Sd, e, m, "fidelity")
                fp, fph = cell(Sd, e, m, "fid_pen")
                lk, lkh = cell(Sd, e, m, "leak")
                au, auh = cell(Sd, e, m, "auc")
                row = [ENAME[e] if m == deps[0] else "", DNAME[m], f(tp, tph, pct=True), f(top3, top3h, pct=True),
                       f(kp, kph)]
                if design == "location":
                    kx, kxh = cell(Sd, e, m, "penalty_share_kept_excess")
                    row.append(f(kx, kxh))
                rows.append(row + [f(fid, fidh, pct=True), f(fp, fph, pct=True), f(lk, lkh, pct=True), f(au, auh)])
            rows.append("MID")
        rows = rows[:-1]
        cols = ["Explainer", "Deployed model", "Top 3, penalised homes (\\%)", "Top 3, all homes (\\%)",
                "Penalty share kept"] + (["Share kept, net of honest"] if design == "location" else []) + \
               ["Fidelity, all homes (\\%)", "Fidelity, penalised homes (\\%)", "Queries reaching $f$ (\\%)",
                "Detector AUC"]
        table(T / f"tab_main_{design}.tex", cols, rows,
              {"feature": "Feature design: the penalty is triggered by the tract share column. ",
               "location": "Location design: the penalty is decided inside the model from the coordinates; the trigger "
                           "is location (both UTM columns for LIME and KernelSHAP; for GeoShapley the location player, which carries "
                           "the intrinsic GEO effect plus half of each feature-by-location interaction). ",
               "nocoord": "Design without coordinates (LIME and KernelSHAP only). "}[design]
              + "Mean $\\pm$ 95\\% interval over seeds; 200 explained homes per seed. Penalty share kept = signed "
                "attribution to the trigger on penalised homes divided by the same under the biased model served "
                "directly" + ("; the net share first subtracts what the honest model attributes to location on "
                              "those homes, so it is negative when an explanation shows less location effect than "
                              "even the honest model does" if design == "location" else "")
              + ". Top 3 = share of explained homes whose trigger attribution (in absolute value) is among the three "
                "largest, on the penalised homes and on all 200. Fidelity = share of the explained (real) homes the "
                "scaffold serves with $f$; on penalised homes it is the share that still receive the penalty. AUC = "
                "probability that a real home scores higher than an explainer query on the detector.", label,
              colspec="llrrrrrrrr" if design == "location" else "llrrrrrrr")

    main_table("feature", "tab:main_feature", ["biased", "gen", "mat"])
    main_table("location", "tab:main_location", ["honest", "biased", "gen", "mat"])
    main_table("nocoord", "tab:main_nocoord", ["biased", "gen", "mat"])

    # macros for the prose (feature design, main numbers)
    for design, tagd in [("feature", "F"), ("location", "L"), ("nocoord", "N")]:
        if design not in S:
            continue
        Sd = S[design]
        kk = "penalty_share_kept_excess" if design == "location" else "penalty_share_kept"
        for e in EXPL:
            for m in ["honest", "biased", "gen", "mat", "mat75", "mat50"]:
                if Sd[(Sd.explainer == e) & (Sd.deployment == m)].empty:
                    continue
                tg = f"{tagd}{ENAME[e]}{MNAME[m]}"
                for k, nm, pct in [(kk, "Kept", False), ("top3", "Top", True), ("fidelity", "Fid", True),
                                   ("top3_pen", "TopPen", True), ("top3_unpen", "TopUnpen", True),
                                   ("fid_pen", "FidPen", True),
                                   ("leak", "Leak", True), ("auc", "Auc", False), ("signed_pen", "Signed", False),
                                   ("gap_total", "GapTotal", False), ("gap_relocated", "Reloc", False)]:
                    mu, hw = cell(Sd, e, m, k)
                    if np.isfinite(mu):
                        macro_ci(tg + nm, mu, hw, pct=pct)
                mu, hw = cell(Sd, e, m, "rank_of_mean")     # rank of the trigger in mean |attribution|, all homes
                if np.isfinite(mu):
                    macro_ci(tg + "Rank", mu, hw, d=1)
                r = Sd[(Sd.explainer == e) & (Sd.deployment == m)]
                if "top_target_mode" in r and isinstance(r.top_target_mode.iloc[0], str):
                    macro(tg + "RelocTo", r.top_target_mode.iloc[0].replace("_", "\\_"))
                    pt = P[design][(P[design].explainer == e) & (P[design].deployment == m)]
                    if "top_target" in pt:
                        macro(tg + "RelocToN", str(int((pt.top_target == r.top_target_mode.iloc[0]).sum())))
                        macro(tg + "RelocN", str(int(pt.top_target.notna().sum())))
                if design == "location":
                    mu, hw = cell(Sd, e, m, "penalty_share_kept")
                    if np.isfinite(mu):
                        macro_ci(tg + "KeptRaw", mu, hw)

    # ------------------------------------------------------------------ decision rules C1, C1', C2, C3
    for design in ["feature", "location"]:
        if design not in P:
            continue
        Pd, Sd = P[design], S[design]
        kk = "penalty_share_kept_excess" if design == "location" else "penalty_share_kept"
        c1 = []
        for e in EXPL:
            x = Pd[(Pd.explainer == e) & (Pd.deployment == "mat")]
            if x.empty:
                continue
            dd = (1.0 - x[kk]).dropna().values          # biased minus matched (biased = 1 by definition)
            mu, hw, n = ci(dd)
            p = float(stats.ttest_1samp(dd, 0, alternative="greater").pvalue) if n > 1 else np.nan
            c1.append([e, mu, hw, n, p])
        if c1:
            adj = holm([r[4] for r in c1])
            for r, pa in zip(c1, adj):
                decisions.append(dict(rule="C1", design=design, explainer=r[0], estimate=r[1], ci_half=r[2],
                                      seeds=r[3], p_holm=pa,
                                      outcome="fooled" if (np.isfinite(r[2]) and r[1] - r[2] > 0 and pa < 0.05)
                                      else "not shown"))
        for e in EXPL:
            if Sd[(Sd.explainer == e) & (Sd.deployment == "mat")].empty:
                continue
            top3 = cell(Sd, e, "mat", "top3")[0]
            kp = cell(Sd, e, "mat", kk)[0]
            fid = cell(Sd, e, "mat", "fidelity")[0]
            tpen = cell(Sd, e, "mat", "top3_pen")[0]
            full = top3 <= 0.05 and kp <= 0.10 and fid >= 0.90
            full_pen = tpen <= 0.05 and kp <= 0.10 and fid >= 0.90
            decisions.append(dict(rule="C1'", design=design, explainer=e, estimate=kp, top3=top3, fidelity=fid,
                                  top3_pen=tpen, outcome="fully fooled" if full else "partly fooled",
                                  outcome_top3_penalised="fully fooled" if full_pen else "partly fooled"))
        if {"geoshapley", "kernelshap"} <= set(Pd.explainer):
            for k in [kk, "auc", "leak_at_fid90", "top3"]:
                if k not in Pd:
                    continue
                mu, hw, n, p = paired(Pd, "geoshapley", "mat", "kernelshap", "mat", k)
                rule = "C2" if k == kk else "C2-aux"
                out = ("no detectable difference at this sample size"
                       if (not np.isfinite(hw) or (mu - hw) <= 0 <= (mu + hw))
                       else ("GeoShapley slightly harder to fool" if mu > 0 else "GeoShapley easier to fool")) \
                    if rule == "C2" else ""
                decisions.append(dict(rule=rule, design=design, explainer="geoshapley-kernelshap", metric=k,
                                      estimate=mu, ci_half=hw, seeds=n, p=p, outcome=out))
                if rule == "C2":
                    macro(f"{'F' if design == 'feature' else 'L'}CtwoClause",
                          {"no detectable difference at this sample size":
                               "no detectable difference between GeoShapley and KernelSHAP",
                           "GeoShapley slightly harder to fool":
                               "GeoShapley keeps slightly more of the penalty than KernelSHAP",
                           "GeoShapley easier to fool": "GeoShapley keeps less of the penalty than KernelSHAP"}[out])
                tg = f"{'F' if design == 'feature' else 'L'}GeoMinusKs{ {kk: 'Kept', 'auc': 'Auc', 'leak_at_fid90': 'Leak', 'top3': 'Top'}[k] }"
                macro_ci(tg.replace(" ", ""), mu, hw, sign=True, pct=(k == "top3" or k == "leak_at_fid90"))
        if "geo_signed_pen" in Pd:
            for k in ["geo_signed_pen", "geo_abs", "intera_abs_sum"]:
                mu, hw, n, p = paired(Pd, "geoshapley", "mat", "geoshapley", "biased", k)
                decisions.append(dict(rule="C3", design=design, explainer="geoshapley", metric=k, estimate=mu,
                                      ci_half=hw, seeds=n, p=p,
                                      outcome="changes" if (np.isfinite(hw) and not (mu - hw <= 0 <= mu + hw))
                                      else "no detectable change"))
                macro_ci(f"{'F' if design == 'feature' else 'L'}GeoDelta{ {'geo_signed_pen': 'Pen', 'geo_abs': 'Abs', 'intera_abs_sum': 'Int'}[k] }".replace(" ", ""),
                         mu, hw, sign=True)

    if "location" in P and "geo_signed_pen" in P["location"]:
        Pl = P["location"][P["location"].explainer == "geoshapley"].pivot(index="seed", columns="deployment",
                                                                          values="geo_signed_pen")
        if {"honest", "biased", "mat"} <= set(Pl.columns):
            g = ((Pl["mat"] - Pl["honest"]) / (Pl["biased"] - Pl["honest"])).dropna()
            mu, hw, n = ci(g.values)
            macro_ci("LGeoGeoNetKept", mu, hw)

    # ------------------------------------------------------------------ frontier table
    rows = []
    for design in ["feature", "location"]:
        if design not in S:
            continue
        Sd = S[design]
        kk = "penalty_share_kept_excess" if design == "location" else "penalty_share_kept"
        for e in ["kernelshap", "geoshapley"]:
            first = True
            for m in ["mat", "mat75", "mat50", "gen"]:
                if Sd[(Sd.explainer == e) & (Sd.deployment == m)].empty:
                    continue
                fid, fh = cell(Sd, e, m, "fidelity")
                kp, kh = cell(Sd, e, m, kk)
                lk, lh = cell(Sd, e, m, "leak")
                t3, th = cell(Sd, e, m, "top3_pen")
                fp, fph = cell(Sd, e, m, "fid_pen")
                rows.append([design.capitalize() if first and e == "kernelshap" else "",
                             ENAME[e] if first else "", {"mat": "Matched, $\\tau^*$", "mat75": "Matched, 75\\%",
                                                         "mat50": "Matched, 50\\%", "gen": "Generic, 0.5"}[m],
                             f(fid, fh, pct=True), f(fp, fph, pct=True), f(kp, kh), f(t3, th, pct=True),
                             f(lk, lh, pct=True)])
                first = False
            # is the generic detector on the matched curve? per seed: generic share kept minus the matched curve
            # (linear in fidelity through the 50/75/90% points) at the generic detector's fidelity
            Pd = P[design]
            dd = []
            for s_ in sorted(Pd.seed.unique()):
                q = Pd[(Pd.seed == s_) & (Pd.explainer == e)].set_index("deployment")
                if not {"gen", "mat", "mat75", "mat50"} <= set(q.index):
                    continue
                xs = q.loc[["mat50", "mat75", "mat"], "fidelity"].values
                ys = q.loc[["mat50", "mat75", "mat"], kk].values
                o = np.argsort(xs)
                dd.append(float(q.loc["gen", kk] - np.interp(q.loc["gen", "fidelity"], xs[o], ys[o])))
            if dd:
                mu, hw, n = ci(dd)
                pv = float(stats.ttest_1samp(dd, 0).pvalue) if n > 1 else np.nan
                macro_ci(f"{design[0].upper()}{ENAME[e]}GenMinusCurve", mu, hw, sign=True)
                macro(f"{design[0].upper()}{ENAME[e]}GenMinusCurveP", pv, d=3)
        rows.append("MID")
    if rows:
        table(T / "tab_frontier.tex", ["Design", "Explainer", "Detector, threshold", "Fidelity, all homes (\\%)",
                                       "Fidelity, penalised homes (\\%)", "Penalty share kept",
                                       "Top 3, penalised homes (\\%)", "Queries reaching $f$ (\\%)"],
              rows[:-1], "The stealth--fidelity frontier for the Shapley explainers. Each matched row uses the same "
                         "detector with a stricter threshold. Penalty share kept is net of the honest model in the "
                         "location design. Mean $\\pm$ 95\\% interval over seeds.",
              "tab:frontier", colspec="lllrrrrr")

    # ------------------------------------------------------------------ GeoShapley decomposition
    rows = []
    for design in ["feature", "location"]:
        if design not in S or "geo_signed_pen" not in S[design]:
            continue
        Sd = S[design]
        for m in ["honest", "biased", "gen", "mat"]:
            if Sd[(Sd.explainer == "geoshapley") & (Sd.deployment == m)].empty:
                continue
            g1 = cell(Sd, "geoshapley", m, "geo_signed_pen")
            g2 = cell(Sd, "geoshapley", m, "geo_abs")
            g3 = cell(Sd, "geoshapley", m, "intera_abs_sum")
            rows.append([design.capitalize() if m in ("honest", "biased") and (rows == [] or rows[-1] == "MID")
                         else "", DNAME[m], f(*g1), f(*g2), f(*g3)])
            gt = f"{design[0].upper()}Geo{MNAME[m]}"
            macro_ci(gt + "GeoPen", *g1)
            macro_ci(gt + "GeoAbs", *g2)
            macro_ci(gt + "Int", *g3)
        rows.append("MID")
    if rows:
        table(T / "tab_geo.tex", ["Design", "Deployed model", "GEO on penalised homes", "Mean $|$GEO$|$",
                                  "Sum of $|$interaction effects$|$"], rows[:-1],
              "GeoShapley's decomposition by deployment. GEO is the intrinsic location effect; interaction effects are "
              "the feature-by-location terms, summed in absolute value per home and averaged. Mean $\\pm$ 95\\% "
              "interval over seeds.", "tab:geo", colspec="llrrr")

    # ------------------------------------------------------------------ mechanism
    rows, mech = [], []
    for p in sorted(glob.glob(str(R / "mechanism/predicted_vs_measured_*.csv"))):
        C = pd.read_csv(p)
        key = Path(p).stem.replace("predicted_vs_measured_", "")
        if "_m-" in key:
            continue
        shutil.copy(p, D)
        rb = Path(p).parent / f"routing_by_size_{key}.csv"
        if rb.exists():
            shutil.copy(rb, D)
        design, e = key.split("_")[:2]
        # descriptive simplification (not pre-specified): the plain average over coalition sizes of the share of
        # with-trigger hybrids routed to f
        if rb.exists():
            RB = pd.read_csv(rb)
            ra = RB.groupby(["seed", "tau_name"]).rho_with.mean().rename("rho_avg").reset_index()
            C = C.merge(ra, on=["seed", "tau_name"], how="left")
            rs = RB[RB.tau_name == "taustar"].groupby("k_with_trigger").rho_with.mean()
            mtag = f"Mech{design.capitalize()}{ENAME[e]}"
            macro(mtag + "RhoFirst", float(rs.iloc[0]), d=2)
            macro(mtag + "RhoLast", float(rs.iloc[-1]), d=2)
            macro(mtag + "RhoMin", float(rs.min()), d=2)
            macro(mtag + "RhoMinK", str(int(rs.idxmin())))
            macro(mtag + "M", str(int(RB.M.iloc[0])))
        first = True
        for tn in ["taustar", "fid75", "fid50"]:
            c = C[C.tau_name == tn]
            if c.empty:
                continue
            pm, ph, n = ci(c.predicted_share_kept)
            mm, mh, _ = ci(c.measured_share_kept) if "measured_share_kept" in c else (np.nan, np.nan, 0)
            fm, fh, _ = ci(c.fidelity_explained)
            rm, rh, _ = ci(c.rho_avg) if "rho_avg" in c else (np.nan, np.nan, 0)
            rows.append([f"{design.capitalize()}, {ENAME[e]}" + (" (raw share)" if design == "location" else "")
                         if first else "",
                         {"taustar": "90\\%", "fid75": "75\\%", "fid50": "50\\%"}[tn], f(fm, fh, pct=True),
                         f(pm, ph), f(rm, rh), f(mm, mh), str(n)])
            ttag = {"taustar": "Ninety", "fid75": "SeventyFive", "fid50": "Fifty"}[tn]
            mtag = f"Mech{design.capitalize()}{ENAME[e]}"
            macro_ci(mtag + "Pred" + ttag, pm, ph)
            macro_ci(mtag + "Meas" + ttag, mm, mh)
            macro_ci(mtag + "Rho" + ttag, rm, rh)
            macro_ci(mtag + "Fid" + ttag, fm, fh, pct=True)
            first = False
        rows.append("MID")
        if "measured_share_kept" in C:
            mae = float((C.predicted_share_kept - C.measured_share_kept).abs().mean())
            sd = float(C.groupby("tau_name").measured_share_kept.std(ddof=1).mean())
            ok = mae <= sd
            mech.append(dict(rule="M", design=design, explainer=e, estimate=mae, sd_measured=sd,
                             outcome="prediction holds" if ok else "identity only"))
            macro(f"Mech{design.capitalize()}{ENAME[e]}MAE", mae, d=3)
            macro(f"Mech{design.capitalize()}{ENAME[e]}SD", sd, d=3)
            if "rho_avg" in C:
                macro(f"Mech{design.capitalize()}{ENAME[e]}RhoMAE",
                      float((C.rho_avg - C.measured_share_kept).abs().mean()), d=3)
    if rows:
        table(T / "tab_mechanism.tex", ["Design, explainer", "Attacker fidelity target", "Fidelity (\\%)",
                                        "Predicted share kept", "Routing average", "Measured share kept", "Seeds"],
              rows[:-1],
              "Mechanism check on the penalised explained homes. \\emph{Predicted}: an independent Monte Carlo "
              "evaluation of Equation~(\\ref{eq:sizes}) for the scaffold, divided by the same for the biased model, "
              "with the attack's detector and thresholds and the auditor's background rows (200 hybrids per home and "
              "coalition size; pre-specified, rule M). \\emph{Routing average}: the plain mean over coalition sizes of "
              "the share of with-trigger hybrids the detector routes to $f$ (descriptive, not pre-specified). "
              "\\emph{Measured}: the penalty share kept in the attack run of the same seed; in the location design the raw "
              "share (not net of the honest model), which is what the routing decomposition predicts. Because "
              "\\emph{Predicted} evaluates the same Shapley average with the same detector, its agreement with "
              "\\emph{Measured} checks the decomposition and the implementation; the routing average is the "
              "substantive approximation. Mean $\\pm$ 95\\% interval over seeds.", "tab:mechanism", colspec="llrrrrr")
    decisions += mech

    # ------------------------------------------------------------------ robustness
    configs = [("Main (XGBoost, default settings)", {}),
               ("Random forest", {"--model": "rf"}), ("Neural net", {"--model": "mlp"}),
               ("Background 25 rows", {"--audit": "bg25"}), ("Background 200 rows", {"--audit": "bg200"}),
               ("$k$-means background", {"--audit": "bg100kmeans"}),
               ("LIME 5,000 samples", {"--audit": "lime5000"}), ("LIME Gaussian sampler", {"--audit": "lime500gauss"}),
               ("Attacker: forest, leaf 5", {"--attacker": "rf_leaf5"}), ("Attacker: XGBoost", {"--attacker": "xgb"}),
               ("Attacker: 4$\\times$ harvested queries", {"--harvest-scale": "4"})]
    main_class = {d["explainer"]: d["outcome"] for d in decisions if d["rule"] == "C1'" and d["design"] == "feature"}
    rows, rob, pending = [], [], []
    for nm, opt in configs:
        ex = [x for kv in opt.items() for x in kv]
        out = summarize(res, "--design", "feature", *ex)
        if out[0] is None:
            continue
        Pc, Sc, sfx = out
        shutil.copy(res / f"per_seed_{sfx}.csv", D)
        shutil.copy(res / f"summary_{sfx}.csv", D)
        first = True
        for e in EXPL:
            if Sc[(Sc.explainer == e) & (Sc.deployment == "mat")].empty:
                continue
            if int(Sc[(Sc.explainer == e) & (Sc.deployment == "mat")].seeds.iloc[0]) < 3:
                pending.append(f"{nm} ({ENAME[e]})")
                continue
            kp, kh = cell(Sc, e, "mat", "penalty_share_kept")
            fid, fh = cell(Sc, e, "mat", "fidelity")
            t3, th = cell(Sc, e, "mat", "top3")
            tp, tph = cell(Sc, e, "mat", "top3_pen")
            n = int(Sc[(Sc.explainer == e) & (Sc.deployment == "mat")].seeds.iloc[0])
            full = t3 <= 0.05 and kp <= 0.10 and fid >= 0.90
            cls = "fully fooled" if full else "partly fooled"
            same = cls == main_class.get(e, cls)
            rob.append(dict(rule="R", design="feature", config=nm, explainer=e, estimate=kp, ci_half=kh,
                            fidelity=fid, top3=t3, seeds=n, outcome=("holds" if same else "BREAKS") + f" ({cls})"))
            rows.append([nm if first else "", ENAME[e], f(kp, kh), f(fid, fh, pct=True), f(t3, th, pct=True),
                         f(tp, tph, pct=True), str(n), "yes" if same else "\\textbf{no}"])
            first = False
            key = "".join(ch for ch in nm if ch.isalnum())[:18]
            macro_ci(f"Rob{key}{ENAME[e]}Kept", kp, kh)
            lk, lkh = cell(Sc, e, "mat", "leak")
            macro_ci(f"Rob{key}{ENAME[e]}Leak", lk, lkh, pct=True)
            macro_ci(f"Rob{key}{ENAME[e]}TopPen", tp, tph, pct=True)
            macro(f"Rob{key}{ENAME[e]}N", str(n))
        if rows and rows[-1] != "MID":
            rows.append("MID")
    if rows:
        table(T / "tab_robust.tex", ["Configuration", "Explainer", "Penalty share kept", "Fidelity (\\%)",
                                     "Top 3, all homes (\\%)", "Top 3, penalised homes (\\%)", "Seeds",
                                     "Same class as main"], rows[:-1],
              "Robustness of the matched scaffold, feature design. Audit-setting rows keep the attacker's detector "
              "trained for the default settings and change only what the auditor does; the biased reference uses the "
              "same changed settings. Class = fully fooled (trigger in top three $\\le$ 5\\%, share kept $\\le$ 0.10, "
              "fidelity $\\ge$ 90\\%, as fixed before the runs) or partly fooled. Configurations with fewer than three "
              "complete seeds are not shown.", "tab:robust", colspec="llrrrrrl")
    decisions += rob
    for mdl, nm in [("rf", "Forest"), ("mlp", "Net")]:
        rm = runs(res, "feature", model=mdl)
        if rm:
            hr, hw_, n = ci([d["honest_r2"] for d in rm])
            macro_ci(f"HonestRtwo{nm}", hr, hw_)
            pr, pw, _ = ci([d["psi_r2"] for d in rm])
            macro_ci(f"PsiRtwo{nm}", pr, pw)
    rr_ = [r for r in rob if not r["config"].startswith("Main")]
    if rr_:
        macro("RobCells", str(len(rr_)))
        macro("RobHold", str(sum(r["outcome"].startswith("holds") for r in rr_)))
        br = [f"{r['config']} ({ENAME[r['explainer']]}: {r['outcome'].split('(')[1].rstrip(')')})"
              for r in rr_ if not r["outcome"].startswith("holds")]
        macro("RobBreakList", "; ".join(br) if br else "none")
        macro("RobConfigs", str(len({r["config"] for r in rr_})))
    macro("RobPending", "; ".join(pending) if pending else "none")

    # ------------------------------------------------------------------ extra: autocorrelation, penalty, no coordinates
    rows, slopes = [], []
    lev = []
    for lam in ["1.0", "0.5", "0.0"]:
        out = summarize(res, "--design", "feature", "--shuffle", lam)
        if out[0] is None:
            continue
        Pc, Sc, sfx = out
        if lam != "1.0":
            shutil.copy(res / f"per_seed_{sfx}.csv", D)
            shutil.copy(res / f"summary_{sfx}.csv", D)
        rr = runs(res, "feature", shuffle=float(lam))
        mi = float(np.mean([d["trigger_morans_I"] for d in rr])) if rr else np.nan
        seeds_here = sorted({d["seed"] for d in rr})
        for e in ["lime", "kernelshap", "geoshapley"]:
            x = Pc[(Pc.explainer == e) & (Pc.deployment == "mat") & (Pc.seed.isin(range(5)))]
            g = Pc[(Pc.explainer == e) & (Pc.deployment == "gen") & (Pc.seed.isin(range(5)))]
            if x.empty:
                continue
            lev.append(dict(lam=float(lam), I=mi, explainer=e, seeds=x.seed.tolist(), leak=x.leak.tolist(),
                            kept=x.penalty_share_kept.tolist(), gen_leak=g.leak.tolist()))
            km, kh, n = ci(x.penalty_share_kept)
            lm, lh, _ = ci(x.leak)
            gm, gh, _ = ci(g.leak)
            lv = {"1.0": "High", "0.5": "Mid", "0.0": "Low"}[lam]
            macro_ci(f"Auto{lv}{ENAME[e]}Leak", lm, lh, pct=True)
            macro_ci(f"Auto{lv}{ENAME[e]}Kept", km, kh)
            macro(f"Auto{lv}Moran", mi, d=2)
            rows.append([f"{mi:.2f}" if e == "lime" else "", ENAME[e], f(gm, gh, pct=True), f(lm, lh, pct=True),
                         f(km, kh), str(n)])
        rows.append("MID")
    if len({l["lam"] for l in lev}) >= 3:
        for e in ["lime", "kernelshap", "geoshapley"]:
            L = [l for l in lev if l["explainer"] == e]
            per = {}
            for l in L:
                for s, v in zip(l["seeds"], l["leak"]):
                    per.setdefault(s, []).append((l["I"], v))
            sl = [np.polyfit(*zip(*pts), 1)[0] for pts in per.values() if len(pts) >= 3]
            mu, hw, n = ci(sl)
            slopes.append(dict(rule="C5", design="feature", explainer=e, estimate=mu, ci_half=hw, seeds=n,
                               outcome="leak depends on Moran's I" if np.isfinite(hw) and not (mu - hw <= 0 <= mu + hw)
                               else "no detectable dependence"))
            macro_ci(f"SlopeMoran{ENAME[e]}", mu, hw, sign=True)
    else:
        slopes.append(dict(rule="C5", design="feature", outcome="fewer than three levels ran: no claim"))
    decisions += slopes
    if rows and len({l["lam"] for l in lev}) >= 2:
        table(T / "tab_autocorr.tex", ["Trigger Moran's $I$", "Explainer", "Generic: queries reaching $f$ (\\%)",
                                       "Matched: queries reaching $f$ (\\%)", "Matched: penalty share kept", "Seeds"],
              rows[:-1], "Spatial autocorrelation of the trigger, lowered by giving a share of homes the trigger "
                         "value of a random other home (seeds 0--4 at every level). Mean $\\pm$ 95\\% interval.",
              "tab:autocorr", colspec="llrrrr")
    # penalty 0.10
    out = summarize(res, "--design", "feature", "--penalty", "0.1")
    if out[0] is not None:
        Pc, Sc, sfx = out
        shutil.copy(res / f"per_seed_{sfx}.csv", D)
        shutil.copy(res / f"summary_{sfx}.csv", D)
        rows = []
        for e in EXPL:
            for m in ["biased", "mat", "mat75", "mat50"]:
                if Sc[(Sc.explainer == e) & (Sc.deployment == m)].empty:
                    continue
                kp, kh = cell(Sc, e, m, "penalty_share_kept")
                fid, fh = cell(Sc, e, m, "fidelity")
                t3, th = cell(Sc, e, m, "top3_pen")
                rows.append([ENAME[e] if m == "biased" else "", DNAME[m], f(t3, th, pct=True), f(kp, kh),
                             f(fid, fh, pct=True)])
            rows.append("MID")
        table(T / "tab_penalty010.tex", ["Explainer", "Deployed model", "Top 3, penalised homes (\\%)",
                                         "Penalty share kept",
                                         "Fidelity (\\%)"], rows[:-1],
              "Penalty 0.10 in log$_{10}$ price (a 21\\% price cut) instead of 0.30, feature design, seeds 0--4.",
              "tab:penalty010", colspec="llrrr")
        for e in EXPL:
            kp, kh = cell(Sc, e, "mat", "penalty_share_kept")
            if np.isfinite(kp):
                macro_ci(f"PenTen{ENAME[e]}Kept", kp, kh)
                macro_ci(f"PenTen{ENAME[e]}TopPen", *cell(Sc, e, "mat", "top3_pen"), pct=True)
                macro_ci(f"PenTen{ENAME[e]}BiasedTopPen", *cell(Sc, e, "biased", "top3_pen"), pct=True)
                macro_ci(f"PenTen{ENAME[e]}FiftyKept", *cell(Sc, e, "mat50", "penalty_share_kept"))

    # ------------------------------------------------------------------ E7 outcome diagnostic
    e7 = R / "residual_diagnostic/summary.csv"
    if e7.exists():
        E = pd.read_csv(e7)
        shutil.copy(e7, D)
        shutil.copy(R / "residual_diagnostic/per_seed.csv", D)
        rows = []
        for (mdl, pen), g in E.groupby(["model", "penalty"], sort=False):
            r = g.iloc[0]
            rows.append([{"honest": "Honest", "biased": "Biased", "scaffold_lime_matched": "Scaffold (LIME-matched)"}[mdl],
                         f"{pen:.2f}", f(r.morans_I, r.morans_I_ci), f(r.auc_resid, r.auc_resid_ci),
                         f(r.auc_lag, r.auc_lag_ci), f(r.lisa_recall_pen, r.lisa_recall_pen_ci, pct=True),
                         f(r.lisa_flag_unpen, r.lisa_flag_unpen_ci, pct=True)])
        table(T / "tab_e7.tex", ["Model", "Penalty", "Moran's $I$ of residuals", "AUC, own residual",
                                 "AUC, local mean residual", "LISA low-low recall (\\%)",
                                 "LISA flags, unpenalised (\\%)"], rows,
              "Outcome-based check on the 600 held-out homes, no explainer involved: residual = served prediction "
              "minus observed log$_{10}$ price; 8 nearest neighbours; AUC against the true penalised set; LISA at "
              "conditional-permutation $p<0.05$. Mean $\\pm$ 95\\% interval over seeds.", "tab:e7",
              colspec="llrrrrr")
        sc = E[(E.model == "scaffold_lime_matched") & (np.isclose(E.penalty, 0.10))]
        hn = E[E.model == "honest"]
        if not sc.empty and not hn.empty:
            aucl, flag = float(sc.auc_lag.iloc[0]), float(hn.lisa_flag_unpen.iloc[0])
            go = aucl >= 0.80 and flag <= 0.06
            decisions.append(dict(rule="E7", design="feature", estimate=aucl, honest_flag=flag,
                                  outcome="go: include section" if go else "no-go: Discussion paragraph"))
            macro_ci("EsevenAucLagScaffoldTen", aucl, float(sc.auc_lag_ci.iloc[0]))
            macro_ci("EsevenHonestFlag", flag, float(hn.lisa_flag_unpen_ci.iloc[0]), pct=True)
            macro_ci("EsevenRecallTen", float(sc.lisa_recall_pen.iloc[0]), float(sc.lisa_recall_pen_ci.iloc[0]),
                     pct=True)
            macro_ci("EsevenPrecTen", float(sc.lisa_precision.iloc[0]), float(sc.lisa_precision_ci.iloc[0]),
                     pct=True)
            meta = R / "residual_diagnostic/meta.json"
            if meta.exists():
                macro("EsevenSeeds", str(json.load(open(meta))["seeds"]))
            for pen in [0.02, 0.05, 0.10, 0.20, 0.30]:
                for mdl, nm in [("biased", "Biased"), ("scaffold_lime_matched", "Scaffold")]:
                    q = E[(E.model == mdl) & (np.isclose(E.penalty, pen))]
                    if not q.empty:
                        macro_ci(f"Eseven{nm}AucLag{PNAME[pen]}",
                                 float(q.auc_lag.iloc[0]), float(q.auc_lag_ci.iloc[0]))

    # ------------------------------------------------------------------ appendix: detectors, per seed, versions
    rows = []
    for design in ["feature", "location"]:
        rr = runs(res, design)
        if not rr:
            continue
        keys = [k for k in ["lime_gen", "lime_mat", "shap_gen", "kernelshap_mat", "geoshapley_mat"]
                if k in rr[0]["detectors"]]
        for k in keys:
            op = "tau05" if k.endswith("_gen") else "taustar"
            tau = ci([d["detectors"][k][op]["tau"] for d in rr])
            fe = ci([d["detectors"][k][op]["fidelity_explained_homes"] for d in rr])
            fc = ci([d["detectors"][k][op]["fidelity_calibration_homes"] for d in rr])
            pr = ci([d["detectors"][k][op]["pearson_r_scaffold_vs_f"] for d in rr])
            nh = ci([d["detectors"][k].get("n_harvested", d["detectors"][k].get("n_neg", np.nan)) for d in rr])
            rows.append([design.capitalize() if k == keys[0] else "", k.replace("_", "\\_"),
                         f(tau[0], tau[1], d=2), f(fe[0], fe[1], pct=True), f(fc[0], fc[1], pct=True),
                         f(pr[0], pr[1]), f(nh[0], nh[1], d=0)])
        rows.append("MID")
    if rows:
        table(T / "tab_detectors.tex", ["Design", "Detector", "Threshold", "Fidelity, explained (\\%)",
                                        "Fidelity, calibration (\\%)", "$r$(scaffold, $f$), test homes",
                                        "Synthetic training rows"], rows[:-1],
              "Detector diagnostics. Generic detectors at threshold 0.5; matched detectors at $\\tau^*$. Fidelity on "
              "the 200 explained homes (not used to set $\\tau^*$) and on the 400 calibration homes; $r$ is the "
              "correlation of scaffold and biased predictions on all 600 test homes.", "tab:detectors",
              colspec="llrrrrr")
    rows = []
    for design in ["feature", "location"]:
        if design not in P:
            continue
        Pd = P[design]
        kk = "penalty_share_kept_excess" if design == "location" else "penalty_share_kept"
        for s_ in sorted(Pd.seed.unique()):
            r = [design.capitalize() if s_ == sorted(Pd.seed.unique())[0] else "", str(int(s_))]
            for e in EXPL:
                x = Pd[(Pd.seed == s_) & (Pd.explainer == e) & (Pd.deployment == "mat")]
                r += [f(float(x[kk].iloc[0])) if not x.empty else "--",
                      f(float(x.fidelity.iloc[0]), pct=True) if not x.empty else "--"]
            rows.append(r)
        rows.append("MID")
    if rows:
        table(T / "tab_perseed.tex", ["Design", "Seed", "LIME $K$", "LIME fid.", "KernelSHAP $K$", "KernelSHAP fid.",
                                      "GeoShapley $K$", "GeoShapley fid."], rows[:-1],
              "Per-seed penalty share kept ($K$, net of honest in the location design) and fidelity (\\%) under the "
              "matched scaffold.", "tab:perseed", colspec="llrrrrrr")
    rr = runs(res, "feature") + runs(res, "location")
    if rr:
        v = rr[0]["versions"]
        macro("VerPython", v.get("python", "?"))
        macro("VerMachine", v.get("machine", "?").replace("_", "\\_"))
        macro("VerShap", v.get("shap", "?"))
        macro("VerXgb", v.get("xgboost", "?"))
        macro("VerSklearn", v.get("sklearn", "?"))
        macro("VerNumpy", v.get("numpy", "?"))
        secs = [sum(c.get("seconds", 0) for c in d["cells"].values()) for d in rr]
        macro("MinutesPerSeed", float(np.mean(secs)) / 60, d=0)

    # ------------------------------------------------------------------ rule outcomes as macros (prose checks)
    RN = {"C1": "Cone", "C1'": "ConePrime", "C2": "Ctwo", "M": "Mech", "C5": "Cfive", "E7": "Eseven"}
    for d in decisions:
        if d["rule"] not in RN or (d["rule"] == "C2" and d.get("metric") not in (None, "penalty_share_kept",
                                                                                  "penalty_share_kept_excess")):
            continue
        ex = d.get("explainer")
        ex = "" if not isinstance(ex, str) else ("GeoMinusKs" if "-" in ex else ENAME.get(ex, ex))
        macro(f"Rule{RN[d['rule']]}{d['design'].capitalize()}{ex}", d["outcome"])

    # ------------------------------------------------------------------ write
    Dd = pd.DataFrame(decisions)
    Dd.to_csv(D / "decisions.csv", index=False)
    lines = ["% generated by scripts/make_paper_tables.py -- do not edit by hand"]
    for k, v in sorted(MACROS.items()):
        if not k.isalpha():
            k2 = "".join(ch if ch.isalpha() else chr(ord("A") + int(ch)) if ch.isdigit() else "" for ch in k)
        else:
            k2 = k
        lines.append(f"\\newcommand{{\\{k2}}}{{{v}}}")
    (T / "numbers.tex").write_text("\n".join(lines) + "\n")
    # macros the prose uses that this run could not produce (results not in yet): visible placeholders, so the
    # draft always compiles and a missing number is obvious in the PDF
    import re
    defined = {l.split("{\\", 1)[1].split("}", 1)[0] for l in lines[1:]}
    used = set()
    for tf in ["abstract", "results", "discussion", "conclusion", "appendix", "fooling_geoxai_draft_v1"]:
        tp = ROOT / "docs/paper" / f"{tf}.tex"
        if tp.exists():
            used |= set(re.findall(r"\\([A-Z][a-z]*[A-Z][A-Za-z]*)", tp.read_text()))
    known_latex = {"IfFileExists", "LaTeX", "TeX"}
    missing = sorted(used - defined - known_latex)
    (T / "numbers_missing.tex").write_text(
        "% generated: macros used in the text but not produced by this run\n"
        + "".join(f"\\providecommand{{\\{m}}}{{\\textbf{{\\textcolor{{red}}{{??{m}}}}}}}\n" for m in missing))
    if missing:
        print(f"{len(missing)} macros used in the text are not produced yet: {', '.join(missing)}")
    pd.set_option("display.width", 220)
    print(Dd.to_string())
    print(f"{len(MACROS)} macros, tables in {T}")


if __name__ == "__main__":
    main()
