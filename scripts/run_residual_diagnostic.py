"""E7: can a spatial diagnostic on outcomes find a bias that the explainers are prevented from seeing?

No explainer is involved. For each seed and penalty we take the held-out test homes, serve them
through the deployed model, and look at residuals r = prediction - observed log price.

Deployed models
  honest    h                                  (control: what an unbiased model's residuals look like)
  biased    f = h - penalty * 1{tract share > 0.25}
  scaffold  f behind the LIME-matched detector (real homes are occasionally misrouted to psi)

Statistics, all on the 600 test homes with 8-nearest-neighbour row-standardised weights
  global Moran's I of r, with a permutation p-value
  per-home scores for "is this home penalised?":  -r  and  -lag(r)  (lag includes the home itself)
      -> AUC against the true penalised set
  local Moran (LISA) low-low clusters at conditional-permutation p < 0.05
      -> recall on penalised homes, flag rate on unpenalised homes, flag rate under the honest model

  python scripts/run_residual_diagnostic.py --seeds 10
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "waterfront", "view", "age"]
THRESH = 0.25


def knn_weights(coords, k=8):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[1][:, 1:]


def morans_i(z, nb):
    z = z - z.mean()
    return float((z * z[nb].mean(1)).sum() / (z * z).sum())


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def lisa_lowlow(r, nb, rng, perms=499, alpha=0.05):
    """Conditional-permutation local Moran; returns boolean mask of significant low-low homes."""
    z = (r - r.mean()) / r.std()
    lag = z[nb].mean(1)
    n, k = nb.shape
    ge = np.zeros(n)
    for _ in range(perms):
        # conditional permutation: neighbours drawn from the other homes
        idx = rng.integers(0, n - 1, size=(n, k))
        idx += (idx >= np.arange(n)[:, None])
        ge += (z[idx].mean(1) <= lag)
    p = (ge + 1) / (perms + 1)          # one-sided: lag as low as observed
    return (z < 0) & (lag < 0) & (p < alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--penalties", default="0.02,0.05,0.10,0.20,0.30")
    ap.add_argument("--no-scaffold", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from run_matched_attack import H_CFG, PSI_CFG, det_lime_matched

    out = root / "results" / "residual_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/processed/seattle_housing_with_demographics.csv")
    feats = BASE + ["pct_black_latino", "UTM_X", "UTM_Y"]
    X, y = df[feats].astype(float), df["log_price"]
    bi = [feats.index(c) for c in BASE + ["UTM_X", "UTM_Y"]]
    rows = []
    for seed in range(a.seeds):
        t0 = time.time()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        h = xgb.XGBRegressor(random_state=seed, **H_CFG).fit(Xtr.values[:, bi], ytr.values)
        hp = h.predict(Xte.values[:, bi])
        pen = (Xte["pct_black_latino"].values > THRESH)
        nb = knn_weights(Xte[["UTM_X", "UTM_Y"]].values)
        rng = np.random.default_rng(seed)
        routed = None
        if not a.no_scaffold:
            psi = xgb.XGBRegressor(random_state=seed, **PSI_CFG).fit(
                Xtr.values[:, [feats.index(c) for c in BASE]], ytr.values)
            pp = psi.predict(Xte.values[:, [feats.index(c) for c in BASE]])
            det, _ = det_lime_matched(Xtr.values, seed, -1)
            routed = det.predict_proba(Xte.values)[:, list(det.classes_).index(1)] >= 0.5
        r_h = hp - yte.values

        def stats(r, label, penalty):
            lag_self = (r + r[nb].sum(1)) / (nb.shape[1] + 1)
            I = morans_i(r, nb)
            perm = np.array([morans_i(rng.permutation(r), nb) for _ in range(199)])
            ll = lisa_lowlow(r, nb, rng)
            return dict(seed=seed, penalty=penalty, model=label, n=len(r), n_pen=int(pen.sum()),
                        rmse=float(np.sqrt((r ** 2).mean())), mean_resid_pen=float(r[pen].mean()),
                        mean_resid_unpen=float(r[~pen].mean()), morans_I=I,
                        morans_p=float(((perm >= I).sum() + 1) / 200),
                        auc_resid=auc(-r, pen), auc_lag=auc(-lag_self, pen),
                        lisa_flag_rate=float(ll.mean()), lisa_recall_pen=float(ll[pen].mean()),
                        lisa_flag_unpen=float(ll[~pen].mean()),
                        lisa_precision=float(pen[ll].mean()) if ll.any() else float("nan"))

        rows.append(stats(r_h, "honest", 0.0))
        for p in [float(v) for v in a.penalties.split(",")]:
            fp = hp - p * pen
            rows.append(stats(fp - yte.values, "biased", p))
            if routed is not None:
                rows.append(stats(np.where(routed, fp, pp) - yte.values, "scaffold_lime_matched", p))
                rows[-1]["share_routed_to_f"] = float(routed.mean())
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    D = pd.DataFrame(rows)
    Z = 1.959963984540054
    num = [c for c in D.columns if c not in ("seed", "penalty", "model")]
    g = D.groupby(["model", "penalty"])[num]
    from scipy.stats import t as student_t
    q = g.count().apply(lambda c: student_t.ppf(0.975, np.maximum(c - 1, 1)))
    S = g.mean().join((q * g.std(ddof=1) / np.sqrt(g.count())).add_suffix("_ci")).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S[["model", "penalty", "rmse", "morans_I", "morans_p", "auc_resid", "auc_lag", "auc_lag_ci",
             "lisa_recall_pen", "lisa_flag_unpen", "lisa_precision"]].round(3).to_string(index=False))
    json.dump(dict(seeds=a.seeds, k=8, perms_lisa=499, alpha=0.05), open(out / "meta.json", "w"))


if __name__ == "__main__":
    main()
