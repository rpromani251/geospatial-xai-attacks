"""Mechanism: why a matched scaffold hides the penalty from LIME but only partly from Shapley explainers.

Shapley attribution of the trigger j is an average over coalition sizes, each size weighted 1/M:
    phi_j = (1/M) * sum_{k=0}^{M-1}  E_{|S|=k, j not in S, b ~ background} [ e(x_{S+j}, b) - e(x_S, b) ]
where e is the served model and (x_S, b) is the hybrid that takes players in S from the explained home
x and the rest from background row b. Under the scaffold e = r*f + (1-r)*psi with r = 1{detector
score >= tau}. Three facts follow:
  * the largest coalition (k = M-1, S+j = every player) is the explained home itself, so that layer is
    served by f whenever the attacker keeps fidelity; efficiency then forces the penalty into the
    attribution somewhere;
  * near-complete coalitions differ from a real home in one or two players and are hard to separate
    from real homes, so their routing rate rho_k stays close to the fidelity;
  * the retained share of the penalty is therefore roughly the Shapley average of the routing rates,
    and the attacker can lower it only by lowering rho_k at large k, i.e. by lowering fidelity.
LIME has no efficiency constraint and its samples are far from the instance, so nothing forces the
penalty into its explanation.

For each seed this script rebuilds the deployment exactly as scripts/run_matched_attack.py does,
loads the matched detector from its cache (or rebuilds it deterministically), and estimates by Monte
Carlo, for every coalition size and every attacker threshold (tau*, 75% and 50% fidelity):
    rho_k        share of hybrids with |S+j| = k+1 routed to f
    layer_k      contribution of size k to the trigger's attribution, scaffold and biased reference
    predicted    Monte Carlo trigger attribution, scaffold / biased  (compare with penalty_share_kept)
Writes results/mechanism/routing_by_size.csv and predicted_vs_measured.csv.

  python scripts/mechanism_routing.py --design feature --explainer kernelshap --seeds 0-4
  python scripts/mechanism_routing.py --design location --explainer geoshapley --seeds 0-4
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def seeds_arg(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--design", choices=["feature", "location"], default="feature")
    ap.add_argument("--explainer", choices=["kernelshap", "geoshapley"], default="kernelshap")
    ap.add_argument("--seeds", type=seeds_arg, default=seeds_arg("0-4"))
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--penalty", type=float, default=0.30)
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--bg", type=int, default=100)
    ap.add_argument("--draws", type=int, default=200, help="Monte Carlo hybrids per home per coalition size")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--results", default=None, help="attack results folder (default results/matched_attack)")
    a = ap.parse_args()
    if a.design == "location" and a.explainer == "kernelshap":
        sys.exit("location design: use --explainer geoshapley (the trigger is the joint GEO player)")
    root = Path(a.root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    from sklearn.model_selection import train_test_split
    from run_matched_attack import (BASE, H_CFG, PSI_CFG, THRESH, Biased, Honest, TractLookup,
                                    _atomic, det_harvested, regressor, tract_raster)

    res = Path(a.results) if a.results else root / "results" / "matched_attack"
    out = root / "results" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/processed/seattle_housing_with_demographics.csv")
    feats = BASE + (["pct_black_latino"] if a.design == "feature" else []) + ["UTM_X", "UTM_Y"]
    X, y = df[feats].astype(float), df["log_price"]
    ux, uy = feats.index("UTM_X"), feats.index("UTM_Y")
    # players: columns moved together
    if a.explainer == "kernelshap":
        players = [[i] for i in range(len(feats))]
        trig = feats.index("pct_black_latino")
    else:
        players = [[i] for i, c in enumerate(feats) if c not in ("UTM_X", "UTM_Y")] + [[ux, uy]]
        trig = len(players) - 1 if a.design == "location" else \
            next(q for q, pl in enumerate(players) if pl == [feats.index("pct_black_latino")])
    M = len(players)
    others = [p for p in range(M) if p != trig]
    rows, comp = [], []
    for seed in a.seeds:
        tag = f"{a.design}_seed{seed}" + ("" if a.penalty == 0.30 else f"_pen{a.penalty:g}") \
            + ("" if a.model == "xgb" else f"_m-{a.model}")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        base_idx = [feats.index(c) for c in BASE + ["UTM_X", "UTM_Y"]]
        h = Honest(regressor(a.model, H_CFG, seed, a.jobs).fit(Xtr.values[:, base_idx], ytr.values), base_idx)
        pidx = [feats.index(c) for c in BASE]
        psi = Honest(regressor(a.model, PSI_CFG, seed, a.jobs).fit(Xtr.values[:, pidx], ytr.values), pidx)
        lookup = TractLookup(*tract_raster(root, res / "_cache")) if a.design == "location" else None
        f = Biased(h, a.design, feats, a.penalty, lookup)
        p = res / "_cache" / tag / f"det_{a.explainer}_mat.pkl"
        if p.exists():
            det, _ = pickle.load(open(p, "rb"))
        else:
            print(f"seed {seed}: detector cache missing, rebuilding (deterministic)", flush=True)
            det, info = det_harvested(a.explainer, f, Xtr, a.bg, seed, a.jobs)
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic(p, lambda t: pickle.dump((det, info), open(t, "wb")))
        c1 = list(det.classes_).index(1)

        def score(Z):
            return det.predict_proba(np.asarray(Z, float))[:, c1]

        Xe, Xcal = Xte.values[:a.n_explain], Xte.values[a.n_explain:]
        s_cal = score(Xcal)
        taus = {"taustar": float(min(0.5, np.quantile(s_cal, 0.10, method="lower"))),
                "fid75": float(np.quantile(s_cal, 0.25, method="lower")),
                "fid50": float(np.quantile(s_cal, 0.50, method="lower"))}
        pen_e = f.penalised(Xe)
        homes = Xe[pen_e]
        B = Xtr.sample(n=a.bg, random_state=seed).values       # the auditor's background
        rng = np.random.default_rng(seed + 31337)
        # hybrids: for each home, size k, draw: S (size k from others), background row b
        Zw, Zo, K = [], [], []
        for x in homes:
            for k in range(M):
                for _ in range(a.draws):
                    S = rng.choice(others, k, replace=False) if k else np.array([], int)
                    b = B[rng.integers(len(B))].copy()
                    zo = b.copy()
                    for pl in S:
                        zo[players[pl]] = x[players[pl]]
                    zw = zo.copy()
                    zw[players[trig]] = x[players[trig]]
                    Zo.append(zo); Zw.append(zw); K.append(k)
        Zw, Zo, K = np.array(Zw), np.array(Zo), np.array(K)
        sw, so = score(Zw), score(Zo)
        fw, fo, pw, po = f.predict(Zw), f.predict(Zo), psi.predict(Zw), psi.predict(Zo)
        ref_layer = np.array([(fw - fo)[K == k].mean() for k in range(M)])
        ref_pen = -a.penalty * np.array([(f.penalised(Zw).astype(float) - f.penalised(Zo))[K == k].mean()
                                         for k in range(M)])
        for nm, tau in taus.items():
            rw, ro = sw >= tau, so >= tau
            ew, eo = np.where(rw, fw, pw), np.where(ro, fo, po)
            layer = np.array([(ew - eo)[K == k].mean() for k in range(M)])
            for k in range(M):
                rows.append(dict(seed=seed, tau_name=nm, tau=tau, k_with_trigger=k + 1, M=M,
                                 rho_with=float(rw[K == k].mean()), rho_without=float(ro[K == k].mean()),
                                 layer_scaffold=float(layer[k]), layer_biased=float(ref_layer[k]),
                                 layer_biased_penalty_only=float(ref_pen[k])))
            fid = float((score(Xe) >= tau).mean())
            comp.append(dict(seed=seed, tau_name=nm, tau=tau, fidelity_explained=fid,
                             fidelity_penalised_homes=float((score(homes) >= tau).mean()),
                             predicted_share_kept=float(layer.mean() / ref_layer.mean()),
                             top_layer_share=float(layer[-1] / M / ref_layer.mean()),
                             n_pen=int(len(homes))))
        # measured, from the attack run
        jf = res / f"{tag}.json"
        if jf.exists():
            d = json.load(open(jf))
            for r in comp:
                if r["seed"] != seed:
                    continue
                cell = {"taustar": "mat", "fid75": "mat75", "fid50": "mat50"}[r["tau_name"]]
                c, bref = d["cells"].get(f"{a.explainer}/{cell}"), d["cells"].get(f"{a.explainer}/biased")
                if c and bref and bref.get("trig_signed_pen"):
                    r["measured_share_kept"] = c["trig_signed_pen"] / bref["trig_signed_pen"]
        print(f"seed {seed}: " + "; ".join(f"{r['tau_name']} fid={r['fidelity_explained']:.2f} "
                                           f"pred={r['predicted_share_kept']:.2f} "
                                           f"meas={r.get('measured_share_kept', float('nan')):.2f}"
                                           for r in comp if r["seed"] == seed), flush=True)
    sfx = f"{a.design}_{a.explainer}" + ("" if a.model == "xgb" else f"_m-{a.model}")
    pd.DataFrame(rows).to_csv(out / f"routing_by_size_{sfx}.csv", index=False)
    C = pd.DataFrame(comp)
    C.to_csv(out / f"predicted_vs_measured_{sfx}.csv", index=False)
    R = pd.DataFrame(rows).groupby(["tau_name", "k_with_trigger"])[["rho_with", "layer_scaffold", "layer_biased"]].mean()
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())
    print(C.groupby("tau_name")[[c for c in C.columns if c not in ("seed", "tau_name")]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
