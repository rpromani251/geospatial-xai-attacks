"""Independent check of the main numbers in the draft.

Recomputes, from the raw per-home attribution files (results/matched_attack/*.npz) and the run JSONs only, without
importing summarize_matched_attack.py or make_paper_tables.py, the main-table quantities for the feature and location
designs, and compares them with the macros in docs/paper/tables/numbers.tex as printed (rounded) in the PDF.

  python scripts/verify_paper_numbers.py [--results results]

Prints one line per mismatch and a count; exit code 1 if anything disagrees.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EN = {"lime": "LIME", "kernelshap": "KernelSHAP", "geoshapley": "GeoShapley"}
MN = {"honest": "Honest", "biased": "Biased", "gen": "Gen", "mat": "Mat", "mat75": "MatSeventyFive",
      "mat50": "MatFifty"}
OP = {"gen": "tau05", "mat": "taustar", "mat75": "tau_fid75", "mat50": "tau_fid50"}


def macros():
    out = {}
    for line in (ROOT / "docs/paper/tables/numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def fmt(x, pct=False):
    """The table script's printed form: 2 decimals, or 1 decimal in percent; no '-0.00'."""
    if x is None or not np.isfinite(x):
        return "--"
    d = 1 if pct else 2
    s = f"{x * (100 if pct else 1):.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$")


def mci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v[0]) if len(v) else np.nan), np.nan
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main_runs(res, design):
    out = []
    for p in sorted(glob.glob(f"{res}/{design}_seed*.json")):
        d = json.load(open(p))
        if (d["seed"] == 42 or d.get("model", "xgb") != "xgb" or d.get("audit", "match") != "match"
                or abs(d.get("penalty", 0.3) - 0.3) > 1e-9 or d.get("attacker", "rf") != "rf"
                or d.get("shuffle", 1.0) != 1.0 or d.get("harvest_scale", 1) != 1):
            continue
        out.append(d)
    return out


def per_home(res, tag, e, m, trig):
    p = Path(res) / f"{tag}__{e}_{m}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    W, pen, names = z["flat"], z["penalised"].astype(bool), [str(n) for n in z["names"]]
    ti = [names.index(t) for t in trig]
    A = np.abs(W)
    ta = A[:, ti].sum(1)
    rank = 1 + (np.delete(A, ti, axis=1) > ta[:, None]).sum(1)
    return dict(signed=float(W[pen][:, ti].sum(1).mean()), top3=float((rank <= 3).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    res = Path(a.results) / "matched_attack"
    M = macros()
    bad, checked = [], 0

    def check(name, value, pct=False):
        nonlocal checked
        if name not in M:
            return
        checked += 1
        got, want = M[name], fmt(value, pct)
        if got == want:
            return
        # a value on a rounding boundary can print either way depending on summation order: accept a difference
        # of at most one unit in the last printed digit when the unrounded value is within 1e-6 of the boundary
        try:
            g = float(got.replace("$-$", "-"))
            v = value * (100 if pct else 1)
            unit = 0.1 if pct else 0.01
            if abs(g - v) <= unit / 2 + 1e-6:
                return
        except ValueError:
            pass
        bad.append(f"{name}: numbers.tex has {got}, recomputed {want}")

    for design, D in [("feature", "F"), ("location", "L")]:
        rr = main_runs(res, design)
        if not rr:
            continue
        for e in EN:
            trig = (["pct_black_latino"] if design == "feature" else
                    (["GEO"] if e == "geoshapley" else ["UTM_X", "UTM_Y"]))
            vals = {}
            for d in rr:
                for m in MN:
                    if f"{e}/{m}" not in d["cells"]:
                        continue
                    h = per_home(res, d["tag"], e, m, trig)
                    if h is None:
                        continue
                    c = d["cells"][f"{e}/{m}"]
                    if m in OP:
                        det = d["detectors"][c["detector"]]
                        h["fid"] = det[OP[m]]["fidelity_explained_homes"]
                        h["leak"] = c["queries_routed_to_f"]
                    vals.setdefault(m, {})[d["seed"]] = h
            if "biased" not in vals:
                continue
            for m, by_seed in vals.items():
                tg = f"{D}{EN[e]}{MN[m]}"
                seeds = sorted(by_seed)
                for k, nm, pct in [("signed", "Signed", False), ("top3", "Top", True), ("fid", "Fid", True),
                                   ("leak", "Leak", True)]:
                    mu, hw = mci([by_seed[s].get(k, np.nan) for s in seeds])
                    check(tg + nm, mu, pct)
                    if np.isfinite(hw):
                        check(tg + nm + "Lo", mu - hw, pct)
                        check(tg + nm + "Hi", mu + hw, pct)
                raw = [by_seed[s]["signed"] / vals["biased"][s]["signed"] for s in seeds if s in vals["biased"]]
                if design == "feature":
                    mu, hw = mci(raw)
                    check(tg + "Kept", mu)
                    if np.isfinite(hw):
                        check(tg + "KeptLo", mu - hw)
                        check(tg + "KeptHi", mu + hw)
                else:
                    mu, _ = mci(raw)
                    check(tg + "KeptRaw", mu)
                    if "honest" in vals:
                        net = [(by_seed[s]["signed"] - vals["honest"][s]["signed"])
                               / (vals["biased"][s]["signed"] - vals["honest"][s]["signed"])
                               for s in seeds if s in vals["biased"] and s in vals["honest"]]
                        mu, hw = mci(net)
                        check(tg + "Kept", mu)
                        if np.isfinite(hw):
                            check(tg + "KeptLo", mu - hw)
                            check(tg + "KeptHi", mu + hw)
        # seeds and data macros
        check(f"HonestRtwo{design.capitalize()}", mci([d["honest_r2"] for d in rr])[0])
        check(f"PsiRtwo{design.capitalize()}", mci([d["psi_r2"] for d in rr])[0])
        if M.get(f"Seeds{design.capitalize()}") != str(len(rr)):
            bad.append(f"Seeds{design.capitalize()}: numbers.tex has {M.get(f'Seeds{design.capitalize()}')}, "
                       f"{len(rr)} complete main runs found")
        checked += 1
        inc = [d["tag"] for d in rr if not d.get("complete", True)]
        if inc:
            bad.append(f"{design}: incomplete runs included: {inc}")

    print(f"checked {checked} printed numbers against an independent recomputation from the npz/json files")
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
