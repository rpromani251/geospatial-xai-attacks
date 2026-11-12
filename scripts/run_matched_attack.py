"""Matched-scaffold attack on LIME, KernelSHAP and GeoShapley: hiding versus fidelity.

For one (design, seed) this script explains the first --n-explain test homes with each
explainer under four deployments:

  honest   h                        XGBoost on the 10 baseline features
  biased   f = h - penalty * 1{.}   served directly (the reference the attack must hide)
  gen      scaffold(f, psi, D_gen)  Phase 1 / Slack-recipe detector, threshold 0.5
  mat      scaffold(f, psi, D_mat)  detector trained on the explainer's own queries, threshold
                                    tau* = min(0.5, 10th percentile of D_mat on held-out real homes),
                                    i.e. the attacker keeps >= 90% of unseen real homes on f

Designs
  feature   11 features; penalty when pct_black_latino > 0.25           (Phase 1 trigger)
  location  10 features; penalty when the tract containing (UTM_X, UTM_Y) has
            pct_black_latino > 0.25, decided inside the model from a 25 m tract raster

Detectors
  LIME    gen = src.attacks.ood_detector.train_lime_ood_detector (X + N(0, 0.3), raw units)
          mat = negatives from LimeTabularExplainer's own default sampler
                (same construction as train_lime_faithful_detector in run_probe_sampler_defense.py)
  SHAP    gen = src.attacks.ood_detector.train_shap_ood_detector (k-means substitution)
          mat = negatives harvested from the explainer's real queries on TRAINING homes, with
                background draws different from the auditor's (one detector each for KernelSHAP
                and GeoShapley)

Held-out real homes: test rows [n_explain:] calibrate tau*; test rows [:n_explain] are explained
and give the reported fidelity. Every detector query score is histogrammed (101 bins), so the
leak-versus-fidelity curve can be rebuilt for any threshold without re-running an explainer.

Robustness options (each gets its own tag, so results never mix)
  --model rf|mlp        model family of h and psi (default xgb). Detectors do not depend on it:
                        explainer query points are set by the sampler, not by the model's outputs.
  audit-settings mismatch: the attacker trains its detectors for the DEFAULT explainer settings
  (Shapley background = 100 sampled training rows; LIME 500 samples, quartile sampler); the auditor
  then explains with different settings. Only the biased and mat cells of the affected explainers run.
    --audit-bg N            Shapley background size at audit time
    --audit-bg-kind kmeans  Shapley background = N k-means centroids (common practice) instead of rows
    --audit-lime-samples N  LIME num_samples at audit time
    --audit-lime-sampler gaussian   LIME with discretize_continuous=False (Gaussian sampler)
    --audit-ks-l1 default   KernelSHAP with shap's default l1_reg="num_features(10)" (see below)
  Note: the harvested Shapley detectors already use background draws different from the auditor's,
  so "a fresh background sample" is the default setting, not a mismatch.

Estimator settings (fixed 12 November after the audit)
  KernelSHAP: shap's KernelExplainer defaults to l1_reg="num_features(10)", which keeps at most 10 features per
  home and sets the rest to zero, even when every coalition is enumerated. With 11 features (feature design) that
  zeroes one feature per home, and a weakened trigger is the likeliest to go. The main runs therefore use
  l1_reg=False: with full enumeration (2^M - 2 <= 2M + 2048 coalitions for M <= 11) the result is the exact
  Shapley value. --audit-ks-l1 default reproduces what an auditor running shap's defaults would see.
  GeoShapley: with more than 8 non-location features (feature design: 9) the package samples coalitions with
  numpy's global random generator (all coalitions of size <= 3 and their complements, plus Shapley-weighted
  draws: about 600 of 1,024 here). We seed it from the background seed, so a run is reproducible and the attacker's
  harvest draws differ from the auditor's. The location design (8 non-location features) is exact.

Resumable: every detector and every chunk of explained homes is cached; rerun the same command
until it prints ALL DONE. --budget stops cleanly before a time limit.

  python scripts/run_matched_attack.py --design feature --seed 0 --budget 150
"""
import argparse
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()
BASE = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "waterfront", "view", "age"]
THRESH = 0.25
PSI_CFG = dict(n_estimators=300, max_depth=5, learning_rate=0.1)
H_CFG = dict(n_estimators=500, max_depth=6, learning_rate=0.1)
EXPLAINERS = ["lime", "kernelshap", "geoshapley"]
CELL_ORDER = [("lime", "biased"), ("lime", "mat"), ("lime", "gen"), ("lime", "honest"),
              ("kernelshap", "biased"), ("kernelshap", "mat"),
              ("geoshapley", "biased"), ("geoshapley", "mat"),
              ("kernelshap", "gen"), ("geoshapley", "gen"),
              ("kernelshap", "honest"), ("geoshapley", "honest")]


def _atomic(path, write):
    """Write via a temp file in the same folder, then rename: a killed job never leaves a half file
    that would crash the next run, and parallel jobs never read a half-written shared cache."""
    import os
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def _savez(path, **arrays):
    def w(tmp):
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
    _atomic(path, w)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ geography ----
def tract_raster(root, cache, res=25.0, margin=3000.0):
    """pct_black_latino of the tract containing each 25 m cell (0 outside every tract)."""
    f = cache / "tract_raster.npz"
    if f.exists():
        d = np.load(f)
        return d["grid"], float(d["x0"]), float(d["y0"]), float(d["res"])
    import pyproj
    import shapely
    from shapely.geometry import shape
    from shapely.ops import transform
    g = json.load(open(root / "data/processed/king_county_tracts_demographics.geojson"))
    tr = pyproj.Transformer.from_crs("EPSG:4269", "EPSG:32610", always_xy=True).transform
    df = pd.read_csv(root / "data/processed/seattle_housing_with_demographics.csv")
    x0, x1 = df.UTM_X.min() - margin, df.UTM_X.max() + margin
    y0, y1 = df.UTM_Y.min() - margin, df.UTM_Y.max() + margin
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    grid = np.zeros((ny, nx), np.float32)
    for ft in g["features"]:
        v = ft["properties"].get("pct_black_latino")
        if v is None or not np.isfinite(v):
            continue
        poly = transform(tr, shape(ft["geometry"]))
        bx0, by0, bx1, by1 = poly.bounds
        i0, i1 = max(0, int((bx0 - x0) // res)), min(nx, int((bx1 - x0) // res) + 1)
        j0, j1 = max(0, int((by0 - y0) // res)), min(ny, int((by1 - y0) // res) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs, ys = x0 + (np.arange(i0, i1) + 0.5) * res, y0 + (np.arange(j0, j1) + 0.5) * res
        XX, YY = np.meshgrid(xs, ys)
        shapely.prepare(poly)
        m = shapely.contains_xy(poly, XX, YY)
        sub = grid[j0:j1, i0:i1]
        sub[m] = v
    _savez(f, grid=grid, x0=x0, y0=y0, res=res)
    return grid, float(x0), float(y0), float(res)


class TractLookup:
    def __init__(self, grid, x0, y0, res):
        self.grid, self.x0, self.y0, self.res = grid, x0, y0, res

    def __call__(self, ux, uy):
        i = np.floor((np.asarray(ux) - self.x0) / self.res).astype(np.int64)
        j = np.floor((np.asarray(uy) - self.y0) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.grid.shape[1]) & (j >= 0) & (j < self.grid.shape[0])
        out = np.zeros(len(i))
        out[ok] = self.grid[j[ok], i[ok]]
        return out


# --------------------------------------------------------------------- models ----
class Honest:
    def __init__(self, model, idx):
        self.model, self.idx = model, idx

    def predict(self, X):
        return self.model.predict(np.asarray(X, float)[:, self.idx])


class Biased:
    def __init__(self, h, design, feats, penalty, lookup=None):
        self.h, self.design, self.penalty, self.lookup = h, design, penalty, lookup
        self.ux = feats.index("UTM_X") if "UTM_X" in feats else None
        self.uy = feats.index("UTM_Y") if "UTM_Y" in feats else None
        self.p = feats.index("pct_black_latino") if design in ("feature", "nocoord") else None

    def penalised(self, X):
        X = np.asarray(X, float)
        if self.design in ("feature", "nocoord"):
            return X[:, self.p] > THRESH
        return self.lookup(X[:, self.ux], X[:, self.uy]) > THRESH

    def predict(self, X):
        X = np.asarray(X, float)
        return self.h.predict(X) - self.penalty * self.penalised(X)


class Deployed:
    """What the explainer queries. With a detector: route, and histogram every score."""

    def __init__(self, f, psi=None, det=None, tau=0.5, skip_first=False):
        self.f, self.psi, self.det, self.tau, self.skip_first = f, psi, det, tau, skip_first
        self.hist, self.n, self.real = np.zeros(101, np.int64), 0, 0
        self.c1 = None if det is None else list(det.classes_).index(1)

    def predict(self, X):
        X = np.asarray(X, float)
        if self.det is None:
            return self.f.predict(X)
        s = self.det.predict_proba(X)[:, self.c1]
        real = s >= self.tau
        k = 1 if self.skip_first else 0          # LIME's first row is the instance itself
        self.hist += np.bincount(np.rint(s[k:] * 100).astype(int), minlength=101)
        self.n += len(s) - k
        self.real += int(real[k:].sum())
        out = np.empty(len(X))
        if real.any():
            out[real] = self.f.predict(X[real])
        if (~real).any():
            out[~real] = self.psi.predict(X[~real])
        return out


class Harvester:
    def __init__(self, f, keep, seed):
        self.f, self.keep, self.rng, self.rows = f, keep, np.random.default_rng(seed), []

    def predict(self, X):
        X = np.asarray(X, float)
        m = self.rng.random(len(X)) < self.keep
        if m.any():
            self.rows.append(X[m].copy())
        return self.f.predict(X)


# ----------------------------------------------------------------- explainers ----
def background(Xtr_df, bg, bg_seed, kind="sample"):
    if kind == "sample":
        return Xtr_df.sample(n=bg, random_state=bg_seed).values
    if kind == "kmeans":       # unweighted centroids in raw units, as shap.kmeans would produce
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=bg, random_state=bg_seed, n_init=4).fit(Xtr_df.values).cluster_centers_
    raise ValueError(kind)


def explain(name, predict, Xtr_df, Xe_df, bg, seed, bg_seed=None, lime_samples=500,
            lime_sampler="quartile", bg_kind="sample", ks_l1=False):
    """Returns dict(flat=(n, k)); GeoShapley adds primary / geo / intera.
    seed drives LIME's sampler; bg_seed picks the Shapley background rows.
    The defaults are the settings the attacker assumes; the auditor may change them."""
    bg_seed = seed if bg_seed is None else bg_seed
    if name == "lime":
        from lime.lime_tabular import LimeTabularExplainer
        ex = LimeTabularExplainer(Xtr_df.values, feature_names=list(Xtr_df.columns),
                                  mode="regression", random_state=seed,
                                  discretize_continuous=(lime_sampler == "quartile"))
        W = np.zeros(Xe_df.shape)
        for i, x in enumerate(Xe_df.values):
            e = ex.explain_instance(x, predict, num_features=Xe_df.shape[1], num_samples=lime_samples)
            for j, w in e.as_map()[1]:
                W[i, j] = w
        return dict(flat=W)
    if name == "kernelshap":
        import shap
        B = background(Xtr_df, bg, bg_seed, bg_kind)
        ex = shap.KernelExplainer(predict, B)
        return dict(flat=np.array(ex.shap_values(Xe_df.values, silent=True, l1_reg=ks_l1)))
    from geoshapley import GeoShapleyExplainer
    B = background(Xtr_df, bg, bg_seed, bg_kind)
    np.random.seed(int(bg_seed) % (2 ** 32))   # GeoShapley samples coalitions from the global generator
    r = GeoShapleyExplainer(predict_f=predict, background=B, g=2).explain(Xe_df, n_jobs=1)
    return dict(flat=np.asarray(r.geoshap_to_shap()), primary=np.asarray(r.primary),
                geo=np.asarray(r.geo), intera=np.asarray(r.geo_intera))


def regressor(family, cfg, seed, jobs):
    """h and psi. xgb is the main model; rf and mlp are the model-family robustness check."""
    nj = jobs if jobs and jobs > 0 else None
    if family == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=seed, n_jobs=nj, **cfg)
    if family == "rf":
        from sklearn.ensemble import RandomForestRegressor
        # 100 trees: test R2 0.87 (as with 300 trees, which predict three times slower)
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5,
                                     random_state=seed, n_jobs=nj)
    if family == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # test R2 about 0.81 over seeds 0-4 (XGBoost 0.89, random forest 0.87); a quantile-transformed
        # input layer was unstable across seeds
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                                            early_stopping=True, max_iter=2000,
                                                            n_iter_no_change=30, random_state=seed))
    raise ValueError(family)


# ------------------------------------------------------------------ detectors ----
def rf(seed, jobs, kind="rf"):
    """Classifier behind the MATCHED detectors. 'rf' is the Slack et al. recipe; the other two are
    stronger attackers used only for the attacker-strength check (--attacker)."""
    from sklearn.ensemble import RandomForestClassifier
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=jobs)
    if kind == "rf_leaf5":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=jobs)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, random_state=seed,
                                 n_jobs=jobs if jobs and jobs > 0 else None)
    raise ValueError(kind)


def det_lime_matched(Xtr, seed, jobs, mult=30, kind="rf"):
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(Xtr, mode="regression", random_state=seed)
    pert = np.vstack([ex._LimeTabularExplainer__data_inverse(x, mult + 1)[1][1:] for x in Xtr])
    real = np.repeat(Xtr, mult, axis=0)
    return rf(seed, jobs, kind).fit(np.vstack([real, pert]),
                              np.r_[np.ones(len(real)), np.zeros(len(pert))]), dict(n_neg=len(pert))


def det_harvested(name, f, Xtr_df, bg, seed, jobs, draws=4, homes=10, cap=40000, kind="rf"):
    keep = 0.004 if name == "kernelshap" else 0.015
    rng = np.random.default_rng(seed + 777)
    rows = []
    for b in range(draws):
        idx = rng.choice(len(Xtr_df), homes, replace=False)
        hv = Harvester(f, keep, seed * 100 + b)
        explain(name, hv.predict, Xtr_df, Xtr_df.iloc[idx], bg, seed=10_000 + seed * 10 + b)
        rows += hv.rows
    H = np.vstack(rows)
    if len(H) > cap:
        H = H[rng.choice(len(H), cap, replace=False)]
    Xtr = np.ascontiguousarray(Xtr_df.values, dtype=np.float64)
    keys = {r.tobytes() for r in Xtr}
    yH = np.array([1.0 if r.tobytes() in keys else 0.0 for r in np.ascontiguousarray(H)])
    reps = max(1, int(round(len(H) / len(Xtr) / 2)))
    Xr = np.repeat(Xtr, reps, axis=0)
    det = rf(seed, jobs, kind).fit(np.vstack([Xr, H]), np.r_[np.ones(len(Xr)), yH])
    return det, dict(n_harvested=int(len(H)), harvested_share_exact_real=float(yH.mean()),
                     real_reps=reps)


# -------------------------------------------------------------------- summary ----
def summarise(arr, names, trig, pen):
    W = arr["flat"]
    A = np.abs(W)
    tix = [names.index(t) for t in trig]
    ta = A[:, tix].sum(1)
    rank = 1 + (np.delete(A, tix, axis=1) > ta[:, None]).sum(1)
    signed = W[:, tix].sum(1)
    out = dict(mean_abs={n: float(A[:, i].mean()) for i, n in enumerate(names)},
               trig_mean_abs=float(ta.mean()), trig_share=float(ta.sum() / A.sum()),
               trig_top1=float((rank == 1).mean()), trig_top3=float((rank <= 3).mean()),
               trig_rank_of_mean=int(1 + (np.delete(A.mean(0), tix) > A[:, tix].mean(0).sum()).sum()),
               trig_signed_pen=float(signed[pen].mean()) if pen.any() else None,
               trig_signed_unpen=float(signed[~pen].mean()), n_pen=int(pen.sum()), n=int(len(W)),
               # mechanism: where the penalty sits. For Shapley explainers the row sum is fixed by
               # efficiency (= served prediction - mean served prediction on the background).
               total_signed_pen=float(W[pen].sum(1).mean()) if pen.any() else None,
               total_signed_unpen=float(W[~pen].sum(1).mean()),
               signed_pen_by_feature={n: float(W[pen, i].mean()) for i, n in enumerate(names)} if pen.any() else None,
               signed_unpen_by_feature={n: float(W[~pen, i].mean()) for i, n in enumerate(names)})
    if "geo" in arr:
        out["geo"] = dict(geo_mean_abs=float(np.abs(arr["geo"]).mean()),
                          geo_signed_pen=float(arr["geo"][pen].mean()) if pen.any() else None,
                          geo_signed_unpen=float(arr["geo"][~pen].mean()),
                          intera_abs_sum=float(np.abs(arr["intera"]).sum(1).mean()),
                          primary_abs_sum=float(np.abs(arr["primary"]).sum(1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--design", choices=["feature", "location", "nocoord"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--bg", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--penalty", type=float, default=0.30)
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--speed", type=float, default=1.0, help="multiplier on per-chunk time estimates")
    ap.add_argument("--attacker", choices=["rf", "rf_leaf5", "xgb"], default="rf",
                    help="classifier for the matched detectors (attacker-strength check)")
    ap.add_argument("--harvest-scale", type=int, default=1,
                    help="multiplies the number of harvested background draws and the row cap")
    ap.add_argument("--shuffle", type=float, default=1.0,
                    help="feature design only: share of homes that keep their own trigger value; the "
                         "rest get the value of a random other home (lowers Moran's I, keeps the marginal)")
    ap.add_argument("--skip-honest-shapley", action="store_true",
                    help="feature design: h ignores the trigger, so its Shapley cells are redundant")
    ap.add_argument("--frontier", action="store_true",
                    help="also run matched Shapley scaffolds at 75%% and 50%% fidelity (cells mat75, mat50)")
    ap.add_argument("--clean-cache", action="store_true",
                    help="delete this run's cached chunks once every planned cell is done (detectors are kept)")
    ap.add_argument("--cells", default="all", help="comma list like lime:mat,kernelshap:biased")
    ap.add_argument("--model", choices=["xgb", "rf", "mlp"], default="xgb",
                    help="model family of h and psi (robustness check)")
    ap.add_argument("--audit-bg", type=int, default=None, help="Shapley background size at audit time")
    ap.add_argument("--audit-bg-kind", choices=["sample", "kmeans"], default="sample")
    ap.add_argument("--audit-lime-samples", type=int, default=500)
    ap.add_argument("--audit-lime-sampler", choices=["quartile", "gaussian"], default="quartile")
    ap.add_argument("--audit-ks-l1", choices=["off", "default"], default="off",
                    help="KernelSHAP l1_reg: off = exact Shapley (main); default = shap's num_features(10)")
    a = ap.parse_args()
    root = Path(a.root)
    sys.path.insert(0, str(root))
    import xgboost as xgb
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from src.attacks.ood_detector import train_lime_ood_detector, train_shap_ood_detector

    tag = f"{a.design}_seed{a.seed}" + ("" if a.penalty == 0.30 else f"_pen{a.penalty:g}")
    tag += ("" if a.attacker == "rf" else f"_att-{a.attacker}") + ("" if a.harvest_scale == 1 else f"_hs{a.harvest_scale}")
    tag += "" if a.shuffle == 1.0 else f"_lam{a.shuffle:g}"
    tag += "" if a.model == "xgb" else f"_m-{a.model}"
    audit_bg = a.bg if a.audit_bg is None else a.audit_bg
    shap_mis = audit_bg != a.bg or a.audit_bg_kind != "sample"
    lime_mis = a.audit_lime_samples != 500 or a.audit_lime_sampler != "quartile"
    ks_mis = a.audit_ks_l1 != "off"
    ks_l1 = "num_features(10)" if ks_mis else False
    audit = "match"
    if shap_mis or lime_mis or ks_mis:
        parts = ([f"bg{audit_bg}" + ("kmeans" if a.audit_bg_kind == "kmeans" else "")] if shap_mis else []) + \
                ([f"lime{a.audit_lime_samples}" + ("gauss" if a.audit_lime_sampler == "gaussian" else "")] if lime_mis else []) + \
                (["ksl1"] if ks_mis else [])
        audit = "-".join(parts)            # e.g. bg25, bg100kmeans, lime5000, lime500gauss, ksl1
    det_tag = tag                          # detectors depend on the attacker's assumptions only
    tag += "" if audit == "match" else f"_aud-{audit}"
    out = Path(a.out) if a.out else root / "results" / "matched_attack"
    cache = (Path(a.cache) if a.cache else out / "_cache") / tag
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    det_cache = cache.parent / det_tag     # mismatch runs reuse the matched run's detectors
    det_cache.mkdir(parents=True, exist_ok=True)
    final = out / f"{tag}.json"
    prev = json.load(open(final)) if final.exists() else {}
    frontier_cells = [f"{e}/{m}" for e in ("kernelshap", "geoshapley") for m in ("mat75", "mat50")]
    if a.design == "nocoord":
        frontier_cells = [c for c in frontier_cells if not c.startswith("geoshapley")]
    if prev.get("complete") and (not a.frontier or all(c in prev.get("cells", {}) for c in frontier_cells)):
        print("ALL DONE", tag)
        return

    def over(need=0.0):
        return time.time() - T0 + need > a.budget

    # ---- data, models (cheap; rebuilt every call)
    df = pd.read_csv(root / "data/processed/seattle_housing_with_demographics.csv")
    morans = None
    if a.design == "feature":
        if a.shuffle < 1.0:
            srng = np.random.default_rng(a.seed + 5000)
            v = df["pct_black_latino"].values.copy()
            swap = srng.random(len(v)) >= a.shuffle
            v[swap] = df["pct_black_latino"].values[srng.integers(0, len(v), int(swap.sum()))]
            df["pct_black_latino"] = v
        from sklearn.neighbors import NearestNeighbors
        nb = NearestNeighbors(n_neighbors=9).fit(df[["UTM_X", "UTM_Y"]].values).kneighbors(
            df[["UTM_X", "UTM_Y"]].values)[1][:, 1:]
        z = df["pct_black_latino"].values - df["pct_black_latino"].values.mean()
        morans = float((z * z[nb].mean(1)).sum() / (z * z).sum())
    if a.design == "nocoord":
        feats = BASE + ["pct_black_latino"]
    else:
        feats = BASE + (["pct_black_latino"] if a.design == "feature" else []) + ["UTM_X", "UTM_Y"]
    X, y = df[feats].astype(float), df["log_price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed)
    base_idx = [feats.index(c) for c in BASE + ([] if a.design == "nocoord" else ["UTM_X", "UTM_Y"])]
    hx = regressor(a.model, H_CFG, a.seed, a.jobs)
    hx.fit(Xtr.values[:, base_idx], ytr.values)
    h = Honest(hx, base_idx)
    px = regressor(a.model, PSI_CFG, a.seed, a.jobs)
    px.fit(Xtr.values[:, [feats.index(c) for c in BASE]], ytr.values)
    psi = Honest(px, [feats.index(c) for c in BASE])
    lookup = None
    if a.design == "location":
        lookup = TractLookup(*tract_raster(root, cache.parent))
    f = Biased(h, a.design, feats, a.penalty, lookup)
    Xe, Xcal = Xte.iloc[:a.n_explain], Xte.iloc[a.n_explain:]
    pen = f.penalised(Xe.values)
    geo_names = [c for c in feats if c not in ("UTM_X", "UTM_Y")] + ["GEO"]
    trig_tab = ["pct_black_latino"] if a.design in ("feature", "nocoord") else ["UTM_X", "UTM_Y"]
    trig_geo = ["pct_black_latino"] if a.design == "feature" else ["GEO"]
    log(f"{tag}: h R2={r2_score(yte, h.predict(Xte.values)):.4f} "
        f"psi R2={r2_score(yte, psi.predict(Xte.values)):.4f} "
        f"penalised test={f.penalised(Xte.values).mean():.3f} explained={int(pen.sum())}/{len(pen)}")

    # ---- detectors (each cached)
    def get_det(key, build, need):
        p = det_cache / f"det_{key}.pkl"
        if p.exists():
            d, info = pickle.load(open(p, "rb"))
        else:
            if over(need):
                return None
            t = time.time()
            d, info = build()
            info = dict(info or {}, train_seconds=round(time.time() - t, 1))
            _atomic(p, lambda t: pickle.dump((d, info), open(t, "wb")))
            log(f"detector {key} trained in {info['train_seconds']}s")
        try:
            d.set_params(n_jobs=a.jobs if a.jobs > 0 else None)
        except Exception:
            d.n_jobs = a.jobs
        return d, info

    jobs = a.jobs
    builders = [
        ("lime_gen", lambda: train_lime_ood_detector(Xtr.values, 0.3, 30, 100, random_state=a.seed), 40),
        ("lime_mat", lambda: det_lime_matched(Xtr.values, a.seed, jobs, kind=a.attacker), 50),
        ("shap_gen", lambda: train_shap_ood_detector(Xtr.values, 10, 10, 20000, 100, random_state=a.seed), 60),
        ("kernelshap_mat", lambda: det_harvested("kernelshap", f, Xtr, a.bg, a.seed, jobs,
                                           draws=4 * a.harvest_scale, cap=40000 * a.harvest_scale,
                                           kind=a.attacker), 90 * a.harvest_scale),
        ("geoshapley_mat", lambda: det_harvested("geoshapley", f, Xtr, a.bg, a.seed, jobs,
                                           draws=4 * a.harvest_scale, cap=40000 * a.harvest_scale,
                                           kind=a.attacker), 90 * a.harvest_scale),
    ]
    if a.design == "nocoord":
        builders = [b for b in builders if not b[0].startswith("geoshapley")]
    dets = {}
    for key, build, need in builders:
        r = get_det(key, build, need)
        if r is None:
            print("BUDGET (detectors)", tag)
            return
        dets[key] = r

    def det_for(e, m):
        key = {"lime": "lime", "kernelshap": "shap", "geoshapley": "shap"}[e] + "_gen" if m == "gen" \
            else f"{e}_mat"     # mat, mat75, mat50 share the detector; only the threshold differs
        return key, dets[key][0]

    def scores(det, Z):
        return det.predict_proba(np.asarray(Z, float))[:, list(det.classes_).index(1)]

    setup = dict(tag=tag, design=a.design, seed=a.seed, penalty=a.penalty, n_explain=a.n_explain,
                 bg=a.bg, feats=feats, attacker=a.attacker, harvest_scale=a.harvest_scale,
                 shuffle=a.shuffle, trigger_morans_I=morans, model=a.model, audit=audit,
                 audit_settings=dict(bg=audit_bg, bg_kind=a.audit_bg_kind, lime_samples=a.audit_lime_samples,
                                     lime_sampler=a.audit_lime_sampler, kernelshap_l1_reg=str(ks_l1)),
                 estimator_notes="kernelshap exact (l1_reg=False) unless audit ksl1; geoshapley coalition sampling "
                                 "seeded from the background seed",
                 honest_r2=float(r2_score(yte, h.predict(Xte.values))),
                 psi_r2=float(r2_score(yte, psi.predict(Xte.values))),
                 share_test_penalised=float(f.penalised(Xte.values).mean()),
                 n_explained_penalised=int(pen.sum()),
                 versions=dict(python=platform.python_version(), machine=platform.machine(),
                               xgboost=xgb.__version__, numpy=np.__version__,
                               sklearn=__import__("sklearn").__version__,
                               shap=__import__("shap").__version__),
                 detectors={})
    if a.design == "location":
        csv_pen = (df.loc[Xte.index, "pct_black_latino"] > THRESH).values
        setup["raster_agrees_with_csv_tract"] = float((f.penalised(Xte.values) == csv_pen).mean())
    fb = f.predict(Xte.values)
    tau_star = {}
    for key, (d, info) in dets.items():
        s_cal, s_exp = scores(d, Xcal.values), scores(d, Xe.values)
        ts = float(min(0.5, np.quantile(s_cal, 0.10, method="lower")))
        tau_star[key] = ts
        if key.endswith("_mat"):
            for tgt in (75, 50):   # frontier points: no 0.5 cap, the threshold is set by the target
                tau_star[f"{key}{tgt}"] = float(np.quantile(s_cal, 1 - tgt / 100, method="lower"))
        s_all = scores(d, Xte.values)
        rec = dict(info)
        ops = [("tau05", 0.5), ("taustar", ts)] + [(f"tau_fid{t}", tau_star[f"{key}{t}"]) for t in (75, 50)
                                                     if f"{key}{t}" in tau_star]
        for nm, tau in ops:
            routed = s_all >= tau
            sp = np.where(routed, fb, psi.predict(Xte.values))
            rec[nm] = dict(tau=tau, fidelity_explained_homes=float((s_exp >= tau).mean()),
                           fidelity_calibration_homes=float((s_cal >= tau).mean()),
                           pearson_r_scaffold_vs_f=float(np.corrcoef(sp, fb)[0, 1]))
        rec["real_score_hist_explained"] = np.bincount(np.rint(s_exp * 100).astype(int), minlength=101).tolist()
        rec["real_score_hist_calibration"] = np.bincount(np.rint(s_cal * 100).astype(int), minlength=101).tolist()
        setup["detectors"][key] = rec

    # ---- cells, chunked
    wanted = CELL_ORDER if a.cells == "all" else [tuple(c.split(":")) for c in a.cells.split(",")]
    if a.skip_honest_shapley:
        wanted = [c for c in wanted if not (c[1] == "honest" and c[0] != "lime")]
    if a.frontier and a.cells == "all":
        wanted = wanted + [tuple(c.split("/")) for c in frontier_cells]
    if a.design == "nocoord":
        wanted = [c for c in wanted if c[0] != "geoshapley"]
    if audit != "match":       # mismatch runs: only the reference and the matched scaffold, affected explainers
        hit = ({"kernelshap", "geoshapley"} if shap_mis else set()) | ({"lime"} if lime_mis else set()) \
            | ({"kernelshap"} if ks_mis else set())
        wanted = [c for c in wanted if c[0] in hit and c[1] in ("biased", "mat")]
    old_cells = prev.get("cells", {})
    chunks = [(s, min(s + a.chunk, a.n_explain)) for s in range(0, a.n_explain, a.chunk)]
    est = {"lime": 8, "kernelshap": 70, "geoshapley": 45}
    cells, pending = {}, False
    for e, m in wanted:
        if f"{e}/{m}" in old_cells:
            cells[f"{e}/{m}"] = old_cells[f"{e}/{m}"]
            continue
        parts = []
        for ci, (s0, s1) in enumerate(chunks):
            p = cache / f"cell_{e}_{m}_{ci:02d}.npz"
            if not p.exists():
                scale = (audit_bg / 100.0) if e != "lime" else (a.audit_lime_samples / 500.0)
                if over(a.speed * est[e] * scale * (a.chunk / 25.0) * (1.6 if m != "honest" and m != "biased" else 1.0)):
                    pending = True
                    break
                t = time.time()
                if m == "honest":
                    dep = Deployed(h)
                elif m == "biased":
                    dep = Deployed(f)
                else:
                    key, d = det_for(e, m)
                    tau = 0.5 if m == "gen" else tau_star[key if m == "mat" else key + m[3:]]
                    dep = Deployed(f, psi, d, tau, skip_first=(e == "lime"))
                arr = explain(e, dep.predict, Xtr, Xe.iloc[s0:s1], audit_bg, seed=a.seed * 1000 + ci,
                              bg_seed=a.seed, lime_samples=a.audit_lime_samples,
                              lime_sampler=a.audit_lime_sampler, bg_kind=a.audit_bg_kind, ks_l1=ks_l1)
                _savez(p, hist=dep.hist, n=dep.n, real=dep.real, secs=time.time() - t,
                         tau=dep.tau if dep.det is not None else np.nan, **arr)
                log(f"{e:10s} {m:7s} chunk {ci + 1}/{len(chunks)} {time.time() - t:5.1f}s")
            parts.append(np.load(p))
        if len(parts) < len(chunks):
            if pending:
                break
            continue
        arr = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files
               if k in ("flat", "primary", "geo", "intera")}
        names = geo_names if e == "geoshapley" else feats
        c = summarise(arr, names, trig_geo if e == "geoshapley" else trig_tab, pen)
        c["seconds"] = float(sum(float(q["secs"]) for q in parts))
        if m == "gen" or m.startswith("mat"):
            hist = sum(q["hist"] for q in parts)
            n, real = int(sum(int(q["n"]) for q in parts)), int(sum(int(q["real"]) for q in parts))
            c.update(detector=det_for(e, m)[0], tau=float(parts[0]["tau"]), queries=n,
                     queries_routed_to_f=real / max(n, 1), query_score_hist=hist.tolist())
        cells[f"{e}/{m}"] = c
        _savez(out / f"{tag}__{e}_{m}.npz", penalised=pen, names=np.array(names), **arr)

    done = (a.cells == "all") and all(f"{e}/{m}" in cells for e, m in wanted)
    cells = {**{k: v for k, v in old_cells.items() if k not in cells}, **cells}   # keep earlier extra cells
    if done and a.clean_cache:     # drop chunk files; keep detectors (scripts/mechanism_routing.py reuses them)
        for q in cache.glob("cell_*.npz"):
            q.unlink(missing_ok=True)
    payload = dict(setup, cells_planned=[f"{e}/{m}" for e, m in wanted], cells=cells, complete=bool(done))
    _atomic(final, lambda t: json.dump(payload, open(t, "w"), indent=1))
    for k, c in cells.items():
        r = c.get("queries_routed_to_f")
        log(f"  {k:18s} trig|a|={c['trig_mean_abs']:.4f} top3={c['trig_top3']:.2f} "
            f"signed(pen)={c['trig_signed_pen'] if c['trig_signed_pen'] is None else round(c['trig_signed_pen'], 3)} "
            f"leak={'-' if r is None else round(r, 3)} tau={c.get('tau', '-')}")
    print("ALL DONE" if done else f"PARTIAL {len(cells)}/{len(wanted)} cells", tag)


if __name__ == "__main__":
    main()
