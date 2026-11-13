"""Aggregate results/matched_attack/*.json into seed-level tables with 95% intervals.

Interval: mean +/- t_{S-1}^{-1}(0.975) * sd / sqrt(S) over S seeds (seed = unit of replication).

  python scripts/summarize_matched_attack.py [--design feature] [--min-cells 12]
Writes results/matched_attack/summary_<design>.csv and prints the tables.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPL = ["lime", "kernelshap", "geoshapley"]
DEP = ["honest", "biased", "gen", "mat", "mat75", "mat50"]


def auc_from_hists(real_hist, query_hist):
    """P(score_real > score_query) + 0.5 P(tie): separability of held-out real homes from queries."""
    r, q = np.asarray(real_hist, float), np.asarray(query_hist, float)
    r, q = r / r.sum(), q / q.sum()
    cq = np.cumsum(q) - q            # P(query score < bin)
    return float((r * (cq + 0.5 * q)).sum())


def frontier(real_hist, query_hist):
    """For every threshold t/100: (fidelity = share of real homes >= t, leak = share of queries >= t)."""
    r, q = np.asarray(real_hist, float), np.asarray(query_hist, float)
    fid = np.cumsum(r[::-1])[::-1] / r.sum()
    leak = np.cumsum(q[::-1])[::-1] / q.sum()
    return fid, leak


def leak_at_fidelity(real_hist, query_hist, target):
    fid, leak = frontier(real_hist, query_hist)
    ok = np.where(fid >= target)[0]
    return float(leak[ok[-1]]) if len(ok) else float("nan")   # strictest threshold still meeting target


def by_feature(d, e, m, cell, dirpath):
    """(signed mean on penalised homes, on unpenalised homes) per feature; from the JSON if the run
    recorded it, otherwise from the saved per-cell attributions."""
    if cell.get("signed_pen_by_feature"):
        return cell["signed_pen_by_feature"], cell["signed_unpen_by_feature"]
    p = Path(dirpath) / f"{d['tag']}__{e}_{m}.npz"
    if not p.exists():
        return None, None
    z = np.load(p)
    W, pen, names = z["flat"], z["penalised"].astype(bool), [str(n) for n in z["names"]]
    if not pen.any():
        return None, None
    return ({n: float(W[pen, i].mean()) for i, n in enumerate(names)},
            {n: float(W[~pen, i].mean()) for i, n in enumerate(names)})


def top3_split(d, e, m, trig, dirpath):
    """Share of penalised / unpenalised explained homes with the trigger among the three largest |attributions|
    (same rank rule as run_matched_attack.summarise: ties go to the trigger)."""
    p = Path(dirpath) / f"{d['tag']}__{e}_{m}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    W, pen, names = z["flat"], z["penalised"].astype(bool), [str(n) for n in z["names"]]
    if not all(t in names for t in trig):
        return {}
    A = np.abs(W)
    ti = [names.index(t) for t in trig]
    ta = A[:, ti].sum(1)
    rank = 1 + (np.delete(A, ti, axis=1) > ta[:, None]).sum(1)
    top = rank <= 3
    return dict(top3_pen=float(top[pen].mean()) if pen.any() else np.nan,
                top3_unpen=float(top[~pen].mean()) if (~pen).any() else np.nan)


def relocation(bp, bu, cp, cu, trig):
    """Penalty bookkeeping on the gap 'penalised minus unpenalised homes', relative to the biased
    reference (the gap removes the baseline shift a scaffold adds to every home).
      gap_kept_trigger   share of the reference trigger gap still on the trigger
      gap_relocated      share that moved onto the other features (efficiency says it cannot vanish
                         while the explained home itself is served by f)
      top_target         feature that received most of it"""
    gb = sum(bp[t] - bu[t] for t in trig)
    if abs(gb) < 1e-9:
        return {}
    others = [n for n in bp if n not in trig]
    moved = {n: ((cp[n] - cu[n]) - (bp[n] - bu[n])) / gb for n in others}
    kept = sum(cp[t] - cu[t] for t in trig) / gb
    tgt = max(moved, key=moved.get) if moved else None
    return dict(gap_kept_trigger=kept, gap_relocated=sum(moved.values()),
                gap_total=kept + sum(moved.values()), top_target=tgt,
                top_target_share=moved.get(tgt, np.nan))


def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    if len(x) < 2:
        return x.mean(), np.nan, len(x)
    from scipy.stats import t as student_t      # t quantile: with ~10 seeds the normal one is ~13% too narrow
    return x.mean(), student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "results" / "matched_attack"))
    ap.add_argument("--design", default="feature")
    ap.add_argument("--exclude-seeds", default="42")
    ap.add_argument("--penalty", type=float, default=0.30)
    ap.add_argument("--attacker", default="rf")
    ap.add_argument("--harvest-scale", type=int, default=1)
    ap.add_argument("--shuffle", type=float, default=1.0)
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--audit", default="match", help="audit-settings label, e.g. bg25, bg100kmeans, lime5000")
    a = ap.parse_args()
    excl = {int(s) for s in a.exclude_seeds.split(",") if s}
    runs, skipped = [], []
    for p in sorted(glob.glob(f"{a.dir}/{a.design}_seed*.json")):
        d = json.load(open(p))
        if (d["seed"] in excl or d.get("penalty", 0.3) != a.penalty or d.get("attacker", "rf") != a.attacker
                or d.get("harvest_scale", 1) != a.harvest_scale or d.get("shuffle", 1.0) != a.shuffle
                or d.get("model", "xgb") != a.model or d.get("audit", "match") != a.audit):
            continue
        if not d.get("complete"):          # a budget-limited call writes its JSON before every cell is in
            skipped.append(d["tag"])
            continue
        runs.append(d)
    print(f"design={a.design}: {len(runs)} complete seeds {[r['seed'] for r in runs]}"
          + (f"; skipped incomplete: {skipped}" if skipped else ""))
    fbg = Path(a.dir) / "fidelity_by_group.csv"     # scripts/fidelity_by_group.py
    FB = pd.read_csv(fbg).set_index(["tag", "explainer", "deployment"]) if fbg.exists() else None
    rows = []
    for d in runs:
        for e in EXPL:
            b = d["cells"].get(f"{e}/biased")
            h = d["cells"].get(f"{e}/honest")
            for m in DEP:
                c = d["cells"].get(f"{e}/{m}")
                if c is None:
                    continue
                r = dict(seed=d["seed"], explainer=e, deployment=m, n_pen=c["n_pen"],
                         trig_abs=c["trig_mean_abs"], top1=c["trig_top1"], top3=c["trig_top3"],
                         rank_of_mean=c["trig_rank_of_mean"], signed_pen=c["trig_signed_pen"],
                         signed_unpen=c["trig_signed_unpen"])
                # trigger in the top three separately on penalised and unpenalised explained homes
                trig_all = ["GEO"] if (e == "geoshapley" and d["design"] == "location") else \
                    (["UTM_X", "UTM_Y"] if d["design"] == "location" else ["pct_black_latino"])
                r.update(top3_split(d, e, m, trig_all, a.dir))
                if b is not None and b["trig_signed_pen"]:
                    r["penalty_share_kept"] = (c["trig_signed_pen"] or 0.0) / b["trig_signed_pen"]
                # same share, net of what the honest model attributes to the trigger on those homes
                # (matters in the location design, where location has a legitimate price effect)
                if b is not None and h is not None and b["trig_signed_pen"] is not None \
                        and abs(b["trig_signed_pen"] - (h["trig_signed_pen"] or 0.0)) > 1e-9:
                    h0 = h["trig_signed_pen"] or 0.0
                    r["penalty_share_kept_excess"] = ((c["trig_signed_pen"] or 0.0) - h0) / (b["trig_signed_pen"] - h0)
                if b is not None and h is not None and abs(b["trig_mean_abs"] - h["trig_mean_abs"]) > 1e-9:
                    r["hiding_H"] = (b["trig_mean_abs"] - c["trig_mean_abs"]) / (b["trig_mean_abs"] - h["trig_mean_abs"])
                if b is not None:
                    trig = ["GEO"] if (e == "geoshapley" and d["design"] != "feature") else \
                        (["pct_black_latino"] if d["design"] != "location" else ["UTM_X", "UTM_Y"])
                    bp, bu = by_feature(d, e, "biased", b, a.dir)
                    cp, cu = by_feature(d, e, m, c, a.dir)
                    if m != "biased" and bp is not None and cp is not None and all(t in bp for t in trig):
                        r.update(relocation(bp, bu, cp, cu, trig))
                if "geo" in c:
                    r.update(geo_abs=c["geo"]["geo_mean_abs"], geo_signed_pen=c["geo"]["geo_signed_pen"],
                             intera_abs_sum=c["geo"]["intera_abs_sum"])
                if m == "gen" or m.startswith("mat"):
                    det = d["detectors"][c["detector"]]
                    op = {"gen": "tau05", "mat": "taustar"}.get(m, f"tau_fid{m[3:]}")
                    rh, qh = det["real_score_hist_explained"], c["query_score_hist"]
                    r.update(tau=c["tau"], leak=c["queries_routed_to_f"],
                             fidelity=det[op]["fidelity_explained_homes"],
                             fid_pen=(FB.loc[(d["tag"], e, m), "fid_pen"]
                                      if FB is not None and (d["tag"], e, m) in FB.index else np.nan),
                             pearson_r=det[op]["pearson_r_scaffold_vs_f"],
                             auc=auc_from_hists(rh, qh),
                             leak_at_fid90=leak_at_fidelity(rh, qh, 0.90),
                             leak_at_fid75=leak_at_fidelity(rh, qh, 0.75),
                             leak_at_fid50=leak_at_fidelity(rh, qh, 0.50))
                rows.append(r)
    sfx = a.design + ("" if a.penalty == 0.30 else f"_pen{a.penalty:g}") + ("" if a.attacker == "rf" else f"_att-{a.attacker}") \
        + ("" if a.harvest_scale == 1 else f"_hs{a.harvest_scale}") + ("" if a.shuffle == 1.0 else f"_lam{a.shuffle:g}") \
        + ("" if a.model == "xgb" else f"_m-{a.model}") + ("" if a.audit == "match" else f"_aud-{a.audit}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no cells yet")
        return
    df.to_csv(f"{a.dir}/per_seed_{sfx}.csv", index=False)
    metrics = [c for c in ["trig_abs", "top1", "top3", "top3_pen", "top3_unpen", "fid_pen", "rank_of_mean", "signed_pen", "penalty_share_kept",
                           "penalty_share_kept_excess", "gap_kept_trigger", "gap_relocated", "gap_total",
                           "hiding_H", "leak", "fidelity", "pearson_r", "tau", "auc", "leak_at_fid90",
                           "leak_at_fid75", "leak_at_fid50", "geo_abs", "geo_signed_pen", "intera_abs_sum"]
               if c in df.columns]
    out = []
    for (e, m), g in df.groupby(["explainer", "deployment"], sort=False):
        r = dict(explainer=e, deployment=m, seeds=len(g))
        if "top_target" in g:
            tt = g["top_target"].dropna()
            r["top_target_mode"] = tt.mode().iloc[0] if len(tt) else None
        for k in metrics:
            mu, hw, n = ci(g[k].tolist())
            r[k], r[k + "_ci"] = mu, hw
        out.append(r)
    S = pd.DataFrame(out)
    S["explainer"] = pd.Categorical(S["explainer"], EXPL)
    S["deployment"] = pd.Categorical(S["deployment"], DEP)
    S = S.sort_values(["explainer", "deployment"])
    S.to_csv(f"{a.dir}/summary_{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["explainer", "deployment", "seeds", "top3", "top3_ci", "signed_pen", "signed_pen_ci",
            "penalty_share_kept", "penalty_share_kept_ci", "gap_kept_trigger", "gap_relocated",
            "leak", "fidelity", "tau", "auc", "auc_ci"]
    print(S[[c for c in show if c in S.columns]].round(3).to_string(index=False))
    # paired contrast: GeoShapley minus KernelSHAP under the matched scaffold
    piv = df[df.deployment == "mat"].pivot(index="seed", columns="explainer")
    for k in ["penalty_share_kept", "penalty_share_kept_excess", "auc", "leak_at_fid90", "top3"]:
        if k in df.columns and {"geoshapley", "kernelshap"} <= set(piv[k].columns):
            diff = (piv[k]["geoshapley"] - piv[k]["kernelshap"]).dropna()
            mu, hw, n = ci(diff.tolist())
            print(f"GeoShapley - KernelSHAP, matched, {k}: {mu:+.3f} +/- {hw:.3f} (n={n} seeds)")


if __name__ == "__main__":
    main()
