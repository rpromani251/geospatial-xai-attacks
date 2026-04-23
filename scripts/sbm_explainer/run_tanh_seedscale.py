"""Seed-scale the tanh sweep at the signal region + test saturation.

Targets:
- β ∈ {2, 5, 10}: 10 fp seeds (current run had 3). Peng wants error bars.
- β ∈ {20, 50}: 5 fp seeds each. Test saturation prediction — once the
  sigmoid is effectively a step function, refinements depend only on
  sign(x_u + δ) with Bernoulli noise, so m_i takes two values regardless
  of β. Δ should plateau or slightly decrease if the step-limit noise
  floor dominates.

Everything else matches run_tanh_sweep.py: CC/LIME scaffold, σ=1.0, K=20,
cosine 8-NN, SBM K=4, 5 SBM seeds per fingerprint.
"""
from __future__ import annotations
import sys, json, os, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SLACK = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/fooling-lime-shap")
sys.path.insert(0, str(SLACK))

from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import RESULTS_DIR

import warnings
warnings.filterwarnings("ignore")

from adversarial_models import Adversarial_Lime_Model
from utils import Params
from get_data import get_and_preprocess_cc

CELLS = [
    (2.0, 10),
    (5.0, 10),
    (10.0, 10),
    (20.0, 5),
    (50.0, 5),
]
K_FP = 20
K_NEIGHBOURS = 8
SBM_K = 4
N_ITER = 400
BURN_IN = 200
N_SBM_SEEDS = 5
N_NODES = 200
SIGMA = 1.0


def cosine_knn(E, k):
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    Z = E / norms
    S = Z @ Z.T
    n = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    top_k = np.argpartition(-S, k, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.int8)
    rows = np.repeat(np.arange(n), k)
    A[rows, top_k.reshape(-1)] = 1
    A = ((A + A.T) > 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def fit_cell(A, layer):
    pln, pex, rr = [], [], []
    for s in range(N_SBM_SEEDS):
        mp = fit_sbm(A, K=SBM_K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                     gamma=1.0, seed=s, verbose=False)
        me = fit_pex_sbm(A, layer=layer, K=SBM_K, n_iter=N_ITER,
                         burn_in=BURN_IN, thin=5, gamma=1.0, seed=s,
                         verbose=False)
        pln.append(layer_separation(mp.z, layer))
        pex.append(layer_separation(me.z, layer))
        Cp = mp.co_clustering_matrix()
        w, a = layer_coclustering(Cp, layer)
        rr.append(w / a if a > 0 else float("nan"))
    pln = np.asarray(pln); pex = np.asarray(pex); rr = np.asarray(rr)
    return dict(plain=float(pln.mean()), pex=float(pex.mean()),
                delta=float((pex - pln).mean()),
                C_ratio=float(np.nanmean(rr)), n_edges=int(A.sum() // 2))


def fingerprint(scaff, X, sigma, K, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, K + 1))
    E[:, 0] = scaff.predict_proba(X)[:, 1]
    for k in range(K):
        pert = X + rng.normal(size=(n, d)) * sigma
        E[:, k + 1] = scaff.predict_proba(pert)[:, 1]
    return E


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def main():
    out = RESULTS_DIR / "sbm_explainer" / "tanh_seedscale"
    out.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(SLACK)
    try:
        params = Params(str(SLACK / "model_configurations" / "experiment_params.json"))
        np.random.seed(params.seed)
        X, y, cols = get_and_preprocess_cc(params)
    finally:
        os.chdir(cwd)
    X["unrelated_column_one"] = np.random.choice([0, 1], size=X.shape[0])
    X["unrelated_column_two"] = np.random.choice([0, 1], size=X.shape[0])
    features = list(X.columns)
    race_idx = features.index("racePctWhite numeric")
    unrelated_idx = features.index("unrelated_column_one")
    X_np = X.values

    xtr, xte, ytr, yte = train_test_split(X_np, y, test_size=0.1, random_state=0)
    ss = StandardScaler().fit(xtr)
    xtr = ss.transform(xtr); xte = ss.transform(xte)
    pos, neg = params.positive_outcome, params.negative_outcome

    def one_hot(a):
        o = np.zeros((len(a), 2), dtype=float); o[np.arange(len(a)), a] = 1.0; return o

    class RacistF:
        def predict(self, X):
            return np.array([neg if x[race_idx] > 0 else pos for x in X])
        def predict_proba(self, X):
            return one_hot(np.asarray(self.predict(X), dtype=int))

    def make_innocuous(beta):
        class InnocuousTanh:
            def __init__(self, seed=17):
                self.rng = np.random.default_rng(seed)
            def predict_proba(self, X):
                p_pos = sigmoid(beta * X[:, unrelated_idx])
                out = np.stack([1.0 - p_pos, p_pos], axis=1)
                if pos == 1:
                    return out
                else:
                    return out[:, ::-1]
            def predict(self, X):
                p_pos = sigmoid(beta * X[:, unrelated_idx])
                u = self.rng.random(len(X))
                preds = np.where(u < p_pos, pos, neg).astype(int)
                return preds
        return InnocuousTanh()

    cat_idx = [unrelated_idx, features.index("unrelated_column_two")]

    X_explain_full = xte[:N_NODES]
    layer = (X_explain_full[:, race_idx] > 0).astype(int)
    print(f"n={len(layer)}  high={int(layer.sum())}  low={int((layer == 0).sum())}", flush=True)

    results = {"cells": [], "config": {
        "sigma": SIGMA, "K_fp": K_FP, "n_nodes": N_NODES,
        "sbm_K": SBM_K, "n_sbm_seeds": N_SBM_SEEDS,
        "cells": CELLS,
    }}
    for beta, n_fp in CELLS:
        print(f"\n═══ β = {beta}  ({n_fp} fp seeds) ═══", flush=True)
        t0 = time.time()
        f_inn = make_innocuous(beta)
        scaff = Adversarial_Lime_Model(RacistF(), f_inn).train(
            xtr, ytr, feature_names=features,
            categorical_features=cat_idx, perturbation_multiplier=30,
        )
        seed_cells = []
        for fs in range(n_fp):
            E = fingerprint(scaff, X_explain_full, SIGMA, K_FP, fs)
            A = cosine_knn(E, K_NEIGHBOURS)
            c = fit_cell(A, layer)
            seed_cells.append(c)
            print(f"  seed {fs:2d}: plain={c['plain']:.3f}  pex={c['pex']:.3f}  "
                  f"Δ={c['delta']:+.3f}", flush=True)
        plains = np.array([c["plain"] for c in seed_cells])
        pexs = np.array([c["pex"] for c in seed_cells])
        deltas = pexs - plains
        summary = dict(
            beta=beta, n_fp=n_fp,
            plain=float(plains.mean()), plain_std=float(plains.std(ddof=1)),
            pex=float(pexs.mean()), pex_std=float(pexs.std(ddof=1)),
            delta=float(deltas.mean()), delta_std=float(deltas.std(ddof=1)),
            delta_sem=float(deltas.std(ddof=1) / np.sqrt(n_fp)),
            per_seed=seed_cells,
        )
        results["cells"].append(summary)
        dt = time.time() - t0
        print(f"  β={beta}: plain={plains.mean():.3f}±{plains.std(ddof=1):.3f}  "
              f"pex={pexs.mean():.3f}±{pexs.std(ddof=1):.3f}  "
              f"Δ={deltas.mean():+.3f}±{deltas.std(ddof=1):.3f}  "
              f"SEM={summary['delta_sem']:.3f}  ({dt:.0f}s)", flush=True)
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
