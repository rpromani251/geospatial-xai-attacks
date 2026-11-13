"""Independent check of the numbers and rule outcomes in the draft.

Recomputes from the raw files only -- the per-home attribution files (results/matched_attack/*.npz), the cached
detectors (results/matched_attack/_cache/<tag>/det_*.pkl), the run JSONs (for settings, thresholds and complete
flags), results/mechanism/predicted_vs_measured_*.csv (predicted column only) and
results/residual_diagnostic/per_seed.csv -- without importing any pipeline script, and compares with the macros in
docs/paper/tables/numbers.tex as printed (rounded). Then checks that the compiled PDF is newer than the macros and
prints no missing-number placeholder.

Covered: main tables (share kept, raw and net; top three on all and on penalised homes; fidelity on all and on
penalised homes, from the detectors; queries reaching f; signed attribution), data macros, GeoShapley minus
KernelSHAP (C2), the GeoShapley decomposition and its changes (C3), the generic-detector-against-curve contrast, rule
outcomes C1 (Holm), C1', C2, M and E7, the mechanism errors, robustness rule R, and the outcome check (E7).

  python scripts/verify_paper_numbers.py [--results results] [--allow-missing]

Prints one line per mismatch and a count; exit code 1 if anything disagrees (or a macro the text uses is missing,
unless --allow-missing).
"""
import argparse
import glob
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EN = {"lime": "LIME", "kernelshap": "KernelSHAP", "geoshapley": "GeoShapley"}
MN = {"honest": "Honest", "biased": "Biased", "gen": "Gen", "mat": "Mat", "mat75": "MatSeventyFive",
      "mat50": "MatFifty"}
DIG = {str(i): chr(ord("A") + i) for i in range(10)}


def macros():
    out = {}
    for line in (ROOT / "docs/paper/tables/numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def fmt(x, pct=False, d=2, sign=False):
    """The table script's printed form: 2 decimals (or d), 1 decimal in percent; no '-0.00'."""
    if x is None or not np.isfinite(x):
        return "--"
    d = 1 if pct else d
    s = f"{x * (100 if pct else 1):{'+' if sign else ''}.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$")


def mci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v[0]) if len(v) else np.nan), np.nan, len(v)
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))), len(v)


def load_runs(res, design, model="xgb", audit="match", penalty=0.3, shuffle=1.0):
    out = []
    for p in sorted(glob.glob(f"{res}/{design}_seed*.json")):
        d = json.load(open(p))
        if (d["seed"] == 42 or d.get("model", "xgb") != model or d.get("audit", "match") != audit
                or abs(d.get("penalty", 0.3) - penalty) > 1e-9 or d.get("attacker", "rf") != "rf"
                or d.get("shuffle", 1.0) != shuffle or d.get("harvest_scale", 1) != 1 or not d.get("complete")):
            continue
        out.append(d)
    return out


def trig_of(design, e):
    if design == "location":
        return ["GEO"] if e == "geoshapley" else ["UTM_X", "UTM_Y"]
    return ["pct_black_latino"]


def per_home(res, tag, e, m, trig):
    p = Path(res) / f"{tag}__{e}_{m}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    W, pen, names = z["flat"], z["penalised"].astype(bool), [str(n) for n in z["names"]]
    ti = [names.index(t) for t in trig]
    A = np.abs(W)
    ta = A[:, ti].sum(1)
    top = (1 + (np.delete(A, ti, axis=1) > ta[:, None]).sum(1)) <= 3
    out = dict(signed=float(W[pen][:, ti].sum(1).mean()), top3=float(top.mean()), top3_pen=float(top[pen].mean()),
               pen=pen)
    if "geo" in z.files:
        out.update(geo_pen=float(z["geo"][pen].mean()), geo_abs=float(np.abs(z["geo"]).mean()),
                   intera=float(np.abs(z["intera"]).sum(1).mean()))
    return out


class Fidelity:
    """Detector scores on the explained homes, recomputed from the cached detectors and the data split."""
    def __init__(self):
        self.df = pd.read_csv(ROOT / "data/processed/seattle_housing_with_demographics.csv")
        self.cache = {}

    def __call__(self, res, d, key, tau, pen):
        from sklearn.model_selection import train_test_split
        ck = (d["tag"], key)
        if ck not in self.cache:
            pk = Path(res) / "_cache" / d["tag"].split("_aud-")[0] / f"det_{key}.pkl"
            if not pk.exists():
                self.cache[ck] = None
            else:
                df = self.df.copy()
                if d["design"] == "feature" and d.get("shuffle", 1.0) < 1.0:
                    rng = np.random.default_rng(d["seed"] + 5000)
                    v = df["pct_black_latino"].values.copy()
                    sw = rng.random(len(v)) >= d["shuffle"]
                    v[sw] = df["pct_black_latino"].values[rng.integers(0, len(v), int(sw.sum()))]
                    df["pct_black_latino"] = v
                _, Xte = train_test_split(df[d["feats"]].astype(float), test_size=0.2, random_state=d["seed"])
                det, _ = pickle.load(open(pk, "rb"))
                self.cache[ck] = det.predict_proba(Xte.values[:d["n_explain"]])[:, list(det.classes_).index(1)]
        s = self.cache[ck]
        if s is None:
            return np.nan, np.nan
        r = s >= tau
        return float(r.mean()), float(r[pen].mean())


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj, run = np.empty_like(ps), 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    R = Path(a.results)
    res = R / "matched_attack"
    M = macros()
    bad, notes = [], []
    n_checked = [0]
    FID = Fidelity()

    def check(name, value, pct=False, d=2, sign=False):
        if name not in M:
            notes.append(f"not in numbers.tex: {name}")
            return
        n_checked[0] += 1
        got, want = M[name], fmt(value, pct, d, sign)
        if got == want:
            return
        try:   # rounding boundary: accept half a unit in the last printed digit
            g = float(got.replace("$-$", "-"))
            unit = 0.1 if pct else 10.0 ** -d
            if abs(g - value * (100 if pct else 1)) <= unit / 2 + 1e-6:
                return
        except ValueError:
            pass
        bad.append(f"{name}: numbers.tex has {got}, recomputed {want}")

    def check_ci(name, v, pct=False, d=2, sign=False):
        mu, hw, n = mci(v)
        check(name, mu, pct, d, sign)
        if np.isfinite(hw):
            check(name + "Lo", mu - hw, pct, d, sign)
            check(name + "Hi", mu + hw, pct, d, sign)
        return mu, hw, n

    def check_str(name, want):
        if name not in M:
            notes.append(f"not in numbers.tex: {name}")
            return
        n_checked[0] += 1
        if M[name] != want:
            bad.append(f"{name}: numbers.tex has '{M[name]}', recomputed '{want}'")

    def cells_for(rr, design, e):
        """per seed: dict deployment -> metrics (from npz and detectors)"""
        out = {}
        for d in rr:
            seed_cells = {}
            for m in MN:
                c = d["cells"].get(f"{e}/{m}")
                if c is None:
                    continue
                h = per_home(res, d["tag"], e, m, trig_of(design, e))
                if h is None:
                    continue
                if m in ("gen", "mat", "mat75", "mat50"):
                    h["fid"], h["fid_pen"] = FID(res, d, c["detector"], c["tau"], h["pen"])
                    h["leak"] = c["queries_routed_to_f"]
                seed_cells[m] = h
            out[d["seed"]] = seed_cells
        return out

    def kept(sc, m, net):
        if m not in sc or "biased" not in sc:
            return np.nan
        if net:
            if "honest" not in sc:
                return np.nan
            return (sc[m]["signed"] - sc["honest"]["signed"]) / (sc["biased"]["signed"] - sc["honest"]["signed"])
        return sc[m]["signed"] / sc["biased"]["signed"]

    rule_main = {}
    for design, D in [("feature", "F"), ("location", "L")]:
        rr = load_runs(res, design)
        if not rr:
            continue
        net = design == "location"
        C = {e: cells_for(rr, design, e) for e in EN}
        for e in EN:
            for m in MN:
                tg = f"{D}{EN[e]}{MN[m]}"
                seeds = [s for s in C[e] if m in C[e][s]]
                if not seeds:
                    continue
                for k, nm, pct in [("signed", "Signed", False), ("top3", "Top", True), ("top3_pen", "TopPen", True),
                                   ("fid", "Fid", True), ("fid_pen", "FidPen", True), ("leak", "Leak", True)]:
                    if k in C[e][seeds[0]][m]:
                        check_ci(tg + nm, [C[e][s][m][k] for s in seeds], pct)
                if m != "biased":
                    check_ci(tg + "Kept", [kept(C[e][s], m, net) for s in seeds])
                    if net:
                        check(tg + "KeptRaw", mci([kept(C[e][s], m, False) for s in seeds])[0])
        # data macros
        Dn = design.capitalize()
        check(f"HonestRtwo{Dn}", mci([d["honest_r2"] for d in rr])[0])
        check(f"PsiRtwo{Dn}", mci([d["psi_r2"] for d in rr])[0])
        check_str(f"Seeds{Dn}", str(len(rr)))
        # C1 with Holm, C1'
        c1 = []
        for e in EN:
            ks = [kept(C[e][s], "mat", net) for s in C[e]]
            dd = np.array([1 - k for k in ks if np.isfinite(k)])
            mu, hw, n = mci(dd)
            c1.append((e, mu, hw, stats.ttest_1samp(dd, 0, alternative="greater").pvalue))
        for (e, mu, hw, _), pa in zip(c1, holm([c[3] for c in c1])):
            check_str(f"RuleCone{Dn}{EN[e]}", "fooled" if (mu - hw > 0 and pa < 0.05) else "not shown")
        for e in EN:
            S = C[e]
            K = mci([kept(S[s], "mat", net) for s in S])[0]
            t3 = mci([S[s]["mat"]["top3"] for s in S if "mat" in S[s]])[0]
            fd = mci([S[s]["mat"]["fid"] for s in S if "mat" in S[s]])[0]
            cls = "fully fooled" if (t3 <= 0.05 and K <= 0.10 and fd >= 0.90) else "partly fooled"
            check_str(f"RuleConePrime{Dn}{EN[e]}", cls)
            if design == "feature":
                rule_main[e] = cls
        # C2
        common = sorted(set(C["geoshapley"]) & set(C["kernelshap"]))
        diff = [kept(C["geoshapley"][s], "mat", net) - kept(C["kernelshap"][s], "mat", net) for s in common]
        mu, hw, n = check_ci(f"{D}GeoMinusKsKept", diff, sign=True)
        out = ("no detectable difference at this sample size" if (mu - hw) <= 0 <= (mu + hw)
               else ("GeoShapley slightly harder to fool" if mu > 0 else "GeoShapley easier to fool"))
        check_str(f"RuleCtwo{Dn}GeoMinusKs", out)
        # C3: GeoShapley decomposition, levels and changes
        G = C["geoshapley"]
        for m in ["honest", "biased", "gen", "mat"]:
            seeds = [s for s in G if m in G[s]]
            if seeds:
                check_ci(f"{D}Geo{MN[m]}GeoPen", [G[s][m]["geo_pen"] for s in seeds])
                check_ci(f"{D}Geo{MN[m]}Int", [G[s][m]["intera"] for s in seeds])
        for k, nm in [("geo_pen", "Pen"), ("geo_abs", "Abs"), ("intera", "Int")]:
            check_ci(f"{D}GeoDelta{nm}", [G[s]["mat"][k] - G[s]["biased"][k] for s in G], sign=True)
        # generic detector against the matched curve
        for e in ["kernelshap", "geoshapley"]:
            dd = []
            for s, sc in C[e].items():
                if not all(m in sc for m in ["gen", "mat", "mat75", "mat50"]):
                    continue
                xs = np.array([sc[m]["fid"] for m in ["mat50", "mat75", "mat"]])
                ys = np.array([kept(sc, m, net) for m in ["mat50", "mat75", "mat"]])
                o = np.argsort(xs)
                dd.append(kept(sc, "gen", net) - np.interp(sc["gen"]["fid"], xs[o], ys[o]))
            if dd:
                check_ci(f"{D}{EN[e]}GenMinusCurve", dd, sign=True)
        # mechanism: measured share recomputed from the attribution files; predicted taken from the Monte Carlo CSV
        for e in ["kernelshap", "geoshapley"]:
            pth = R / f"mechanism/predicted_vs_measured_{design}_{e}.csv"
            if not pth.exists():
                continue
            P = pd.read_csv(pth)
            P["meas"] = [kept(C[e].get(int(s), {}), {"taustar": "mat", "fid75": "mat75", "fid50": "mat50"}[t], False)
                         for s, t in zip(P.seed, P.tau_name)]
            mae = float((P.predicted_share_kept - P.meas).abs().mean())
            sd = float(P.groupby("tau_name").meas.std(ddof=1).mean())
            tg = f"Mech{Dn}{EN[e]}"
            check(tg + "MAE", mae, d=3)
            check(tg + "SD", sd, d=3)
            check_str(f"RuleMech{Dn}{EN[e]}", "prediction holds" if mae <= sd else "identity only")

    # robustness, rule R (feature design, matched scaffold, configurations with >= 3 complete seeds)
    cfgs = [("Random forest", dict(model="rf")), ("Neural net", dict(model="mlp")),
            ("Background 25 rows", dict(audit="bg25")), ("Background 200 rows", dict(audit="bg200")),
            ("$k$-means background", dict(audit="bg100kmeans")), ("LIME 5,000 samples", dict(audit="lime5000")),
            ("LIME Gaussian sampler", dict(audit="lime500gauss"))]
    hold = cells = 0
    for nm, kw in cfgs:
        rr = load_runs(res, "feature", **kw)
        key = "".join(ch for ch in nm if ch.isalnum())[:18]
        key = "".join(DIG.get(ch, ch) for ch in key)
        for e in EN:
            S = cells_for(rr, "feature", e)
            S = {s: v for s, v in S.items() if "mat" in v and "biased" in v}
            if len(S) < 3:
                continue
            ks = [kept(v, "mat", False) for v in S.values()]
            check_ci(f"Rob{key}{EN[e]}Kept", ks)
            K = mci(ks)[0]
            t3 = mci([v["mat"]["top3"] for v in S.values()])[0]
            fd = mci([v["mat"]["fid"] for v in S.values()])[0]
            if not np.isfinite(fd):     # audit-setting runs: detectors and thresholds of the default run
                fd = mci([v["mat"].get("fid", np.nan) for v in S.values()])[0]
            cls = "fully fooled" if (t3 <= 0.05 and K <= 0.10 and fd >= 0.90) else "partly fooled"
            cells += 1
            hold += cls == rule_main.get(e)
    if cells:
        check_str("RobCells", str(cells))
        check_str("RobHold", str(hold))

    # smaller penalty, design without coordinates, autocorrelation (C5)
    rr = load_runs(res, "feature", penalty=0.1)
    if rr:
        for e in EN:
            S = {s: v for s, v in cells_for(rr, "feature", e).items() if "mat" in v and "biased" in v}
            if S:
                check_ci(f"PenTen{EN[e]}Kept", [kept(v, "mat", False) for v in S.values()])
                check_ci(f"PenTen{EN[e]}TopPen", [v["mat"]["top3_pen"] for v in S.values()], pct=True)
    rr = load_runs(res, "nocoord")
    if rr:
        for e in ["lime", "kernelshap"]:
            S = {s: v for s, v in cells_for(rr, "nocoord", e).items() if "mat" in v and "biased" in v}
            if S:
                check_ci(f"N{EN[e]}MatKept", [kept(v, "mat", False) for v in S.values()])
                check_ci(f"N{EN[e]}MatLeak", [v["mat"]["leak"] for v in S.values()], pct=True)
    lev = {}
    for lam in [1.0, 0.5, 0.0]:
        for d in load_runs(res, "feature", shuffle=lam):
            if d["seed"] in range(5):
                for e in EN:
                    c = d["cells"].get(f"{e}/mat")
                    if c:
                        lev.setdefault(e, {}).setdefault(d["seed"], []).append((d["trigger_morans_I"],
                                                                                c["queries_routed_to_f"]))
    for e, per in lev.items():
        sl = [np.polyfit(*zip(*pts), 1)[0] for pts in per.values() if len(pts) >= 3]
        if sl:
            mu, hw, n = check_ci(f"SlopeMoran{EN[e]}", sl, sign=True)
            check_str(f"RuleCfiveFeature{EN[e]}", "leak depends on Moran's I" if not (mu - hw <= 0 <= mu + hw)
                      else "no detectable dependence")

    # outcome check (E7)
    e7 = R / "residual_diagnostic/per_seed.csv"
    if e7.exists():
        E = pd.read_csv(e7)
        sc = E[(E.model == "scaffold_lime_matched") & np.isclose(E.penalty, 0.10)]
        hn = E[E.model == "honest"]
        mu, _, _ = check_ci("EsevenAucLagScaffoldTen", sc.auc_lag.values)
        fl, _, _ = check_ci("EsevenHonestFlag", hn.lisa_flag_unpen.values, pct=True)
        check_ci("EsevenRecallTen", sc.lisa_recall_pen.values, pct=True)
        check_str("RuleEsevenFeature", "go: include section" if (mu >= 0.80 and fl <= 0.06)
                  else "no-go: Discussion paragraph")

    # the PDF
    paper = ROOT / "docs/paper"
    pdf, nums = paper / "fooling_geoxai_draft_v1.pdf", paper / "tables/numbers.tex"
    if not pdf.exists() or pdf.stat().st_mtime < nums.stat().st_mtime:
        bad.append("PDF is older than numbers.tex: recompile before checking")
    else:
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        q = sorted(set(re.findall(r"\?\?[A-Za-z]+", txt)))
        if q:
            (notes if a.allow_missing else bad).append(f"PDF prints missing-number placeholders: {', '.join(q)}")
        notes.append(f"PDF has {txt.count('TODO:')} TODO markers")
    miss = (paper / "tables/numbers_missing.tex")
    if miss.exists():
        mm = re.findall(r"providecommand\{\\(\w+)\}", miss.read_text())
        if mm:
            (notes if a.allow_missing else bad).append(f"macros used in the text but not produced: {', '.join(mm)}")

    print(f"checked {n_checked[0]} printed numbers and rule outcomes against an independent recomputation")
    for n in notes:
        print("note:", n)
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
