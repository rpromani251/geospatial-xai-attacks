"""Fidelity split by penalised and unpenalised explained homes.

The attack JSONs record fidelity on all 200 explained homes. For unpenalised homes f = h, so serving them f delivers
no bias; the share of *penalised* homes served by f is the share of the bias actually delivered. This script reloads
each run's cached detectors, scores the explained homes, applies the threshold recorded in each cell, and writes

  results/matched_attack/fidelity_by_group.csv
      tag, design, seed, model, penalty, shuffle, explainer, deployment, tau, fid_all, fid_pen, fid_unpen, n_pen

It checks that fid_all reproduces the fidelity stored in the JSON and stops if it does not. Only complete runs with
the auditor's default settings are read (audit-setting runs reuse the default run's detectors and thresholds).

  python scripts/fidelity_by_group.py [--results results]
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OP = {"gen": "tau05", "mat": "taustar", "mat75": "tau_fid75", "mat50": "tau_fid50"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    res = Path(a.results) / "matched_attack"
    sys.path.insert(0, str(ROOT))
    from sklearn.model_selection import train_test_split
    df0 = pd.read_csv(ROOT / "data/processed/seattle_housing_with_demographics.csv")
    old = res / "fidelity_by_group.csv"
    done = pd.read_csv(old) if old.exists() else pd.DataFrame(columns=["tag"])
    rows = [r for r in done.to_dict("records")]
    have = set(done["tag"])
    for p in sorted(glob.glob(str(res / "*_seed*.json"))):
        d = json.load(open(p))
        if d.get("audit", "match") != "match" or not d.get("complete") or d["tag"] in have:
            continue
        df = df0.copy()
        if d["design"] == "feature" and d.get("shuffle", 1.0) < 1.0:       # as in run_matched_attack.py
            srng = np.random.default_rng(d["seed"] + 5000)
            v = df["pct_black_latino"].values.copy()
            swap = srng.random(len(v)) >= d["shuffle"]
            v[swap] = df["pct_black_latino"].values[srng.integers(0, len(v), int(swap.sum()))]
            df["pct_black_latino"] = v
        X = df[d["feats"]].astype(float)
        _, Xte = train_test_split(X, test_size=0.2, random_state=d["seed"])
        Xe = Xte.values[:d["n_explain"]]
        npz = sorted(glob.glob(str(res / f"{d['tag']}__*_biased.npz")))
        if not npz:
            print("no attribution file for", d["tag"])
            continue
        pen = np.load(npz[0])["penalised"].astype(bool)
        dets = {}
        for cname, c in d["cells"].items():
            e, m = cname.split("/")
            if m not in OP:
                continue
            key = c["detector"]
            if key not in dets:
                pk = res / "_cache" / d["tag"] / f"det_{key}.pkl"
                if not pk.exists():
                    print("detector cache missing:", pk)
                    dets[key] = None
                    continue
                det, _ = pickle.load(open(pk, "rb"))
                dets[key] = det.predict_proba(Xe)[:, list(det.classes_).index(1)]
            s = dets[key]
            if s is None:
                continue
            r = s >= c["tau"]
            want = d["detectors"][key][OP[m]]["fidelity_explained_homes"]
            if abs(r.mean() - want) > 1e-9:
                sys.exit(f"{d['tag']} {cname}: recomputed fidelity {r.mean():.4f} != stored {want:.4f}")
            rows.append(dict(tag=d["tag"], design=d["design"], seed=d["seed"], model=d.get("model", "xgb"),
                             penalty=d.get("penalty", 0.3), shuffle=d.get("shuffle", 1.0), explainer=e, deployment=m,
                             tau=c["tau"], fid_all=float(r.mean()), fid_pen=float(r[pen].mean()),
                             fid_unpen=float(r[~pen].mean()), n_pen=int(pen.sum())))
        print(d["tag"], "ok", flush=True)
    pd.DataFrame(rows).to_csv(old, index=False)
    print(f"{len(rows)} rows -> {old}")


if __name__ == "__main__":
    main()
