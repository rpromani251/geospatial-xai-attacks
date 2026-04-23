"""Rung-3 fingerprint + finite-K pEx on Slack's COMPAS adversarial scaffold.

Loads Slack et al.'s Adversarial_Lime_Model (biased racist_model_f +
innocuous model keyed on unrelated columns), wraps it with the
behavioral-fingerprint encoder, and runs plain SBM vs finite-K pEx
on the cosine 8-NN fingerprint graph. Layer = race.

Predicts: if the rung-3 + finite-K pEx pipeline is a generic
scaffold-attack detector, Δ(pEx − plain) > 0 and C ratio > 2 on this
setup, in direct analogy to the geospatial Gauss+LIME cell.

Runs at a few Gaussian pert σ levels to check detector-manifold
dependence.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Slack repo lives here
SLACK = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/fooling-lime-shap")
sys.path.insert(0, str(SLACK))

# repo imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import RESULTS_DIR

import warnings
warnings.filterwarnings("ignore")

# Slack imports (must happen after sys.path insert)
from adversarial_models import (
    Adversarial_Lime_Model, Adversarial_Kernel_SHAP_Model,
)
from utils import Params
from get_data import get_and_preprocess_compas_data

K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
N_GAUSS = 20
N_NODES = 200
SIGMAS = [0.1, 0.3, 1.0]  # post-StandardScaler std multipliers


def gaussian_fingerprints_proba(scaffold, X, sigma, n_pert=N_GAUSS, seed=0):
    """Fingerprint from class-1 probabilities. Shape (n, n_pert + 1)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, n_pert + 1))
    E[:, 0] = scaffold.predict_proba(X)[:, 1]
    for k in range(n_pert):
        pert = X + rng.normal(size=(n, d)) * sigma
        E[:, k + 1] = scaffold.predict_proba(pert)[:, 1]
    return E


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
    plain_nmis, pex_nmis, ratios = [], [], []
    for s in range(N_SEEDS):
        mp = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                     gamma=1.0, seed=s, verbose=False)
        me = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                         thin=5, gamma=1.0, seed=s, verbose=False)
        plain_nmis.append(layer_separation(mp.z, layer))
        pex_nmis.append(layer_separation(me.z, layer))
        Cp = mp.co_clustering_matrix()
        w, a = layer_coclustering(Cp, layer)
        ratios.append(w / a if a > 0 else float("nan"))
    pln = np.asarray(plain_nmis); pex = np.asarray(pex_nmis)
    r = np.asarray(ratios)
    return dict(
        plain_nmi_mean=float(pln.mean()), plain_nmi_std=float(pln.std(ddof=1)),
        pex_nmi_mean=float(pex.mean()), pex_nmi_std=float(pex.std(ddof=1)),
        delta_mean=float((pex - pln).mean()),
        delta_std=float((pex - pln).std(ddof=1)),
        C_ratio=float(r.mean()), n_edges=int(A.sum() // 2),
    )


def main():
    out = RESULTS_DIR / "sbm_explainer" / "compas_fingerprint"
    out.mkdir(parents=True, exist_ok=True)

    # Slack's exact setup (get_data uses relative path to data/)
    import os
    cwd = os.getcwd()
    os.chdir(SLACK)
    try:
        params = Params(str(SLACK / "model_configurations" / "experiment_params.json"))
        np.random.seed(params.seed)
        X, y, cols = get_and_preprocess_compas_data(params)
    finally:
        os.chdir(cwd)
    X["unrelated_column_one"] = np.random.choice([0, 1], size=X.shape[0])
    X["unrelated_column_two"] = np.random.choice([0, 1], size=X.shape[0])
    features = list(X.columns)
    race_idx = features.index("race")
    unrelated_idx = features.index("unrelated_column_one")
    unrelated_idx2 = features.index("unrelated_column_two")
    X_np = X.values

    xtr, xte, ytr, yte = train_test_split(X_np, y, test_size=0.1, random_state=0)
    ss = StandardScaler().fit(xtr)
    xtr = ss.transform(xtr); xte = ss.transform(xte)
    print(f"COMPAS: train {xtr.shape}, test {xte.shape}, d={xtr.shape[1]}")
    print(f"race_idx={race_idx}, unrelated_idx={unrelated_idx}")

    # Slack's racist + innocuous models (inline — they reference column indices)
    pos = params.positive_outcome; neg = params.negative_outcome

    def one_hot(a):
        out = np.zeros((len(a), 2), dtype=float)
        out[np.arange(len(a)), a] = 1.0
        return out

    class RacistF:
        def predict(self, X):
            return np.array([neg if x[race_idx] > 0 else pos for x in X])
        def predict_proba(self, X):
            return one_hot(np.asarray(self.predict(X), dtype=int))

    class InnocuousPsi:
        def predict_proba(self, X):
            preds = np.array([neg if x[unrelated_idx] > 0 else pos for x in X])
            return one_hot(preds.astype(int))

    # Build LIME-scaffold (this trains the OOD detector internally)
    cat_idx = [features.index(c) for c in (
        "unrelated_column_one", "unrelated_column_two",
        "c_charge_degree_F", "c_charge_degree_M",
        "two_year_recid", "race", "sex_Male", "sex_Female",
    )]
    print("\nTraining LIME-style adversarial scaffold (Slack)...")
    adv_lime = Adversarial_Lime_Model(RacistF(), InnocuousPsi()).train(
        xtr, ytr, feature_names=features,
        categorical_features=cat_idx, perturbation_multiplier=30,
    )
    print("Training SHAP-style adversarial scaffold (Slack)...")
    adv_shap = Adversarial_Kernel_SHAP_Model(RacistF(), InnocuousPsi()).train(
        xtr, ytr, feature_names=features,
    )

    # Pick 200 test points. Layer = race (already 0/1 post-encoding).
    # Make sure both layers present.
    n_take = min(N_NODES, xte.shape[0])
    X_explain = xte[:n_take]
    race_vals = X.values[np.argsort(np.random.default_rng(0).integers(0, 10**9, size=X.shape[0]))]  # placeholder
    # layer from test points' pre-standardization race column:
    # The race column in standardized form is a signed value; after StandardScaler,
    # mean=0. Original was 0/1. Threshold at 0 recovers it.
    layer = (X_explain[:, race_idx] > 0).astype(int)
    n_high = int(layer.sum()); n_low = int((layer == 0).sum())
    print(f"\nn={n_take}  race=1: {n_high}   race=0: {n_low}")
    if n_high == 0 or n_low == 0:
        print("Missing a layer; bumping n_take...")
        X_explain = xte
        layer = (X_explain[:, race_idx] > 0).astype(int)
        n_high = int(layer.sum()); n_low = int((layer == 0).sum())
        print(f"n={len(layer)}  race=1: {n_high}   race=0: {n_low}")

    # Routing diagnostic: fraction of Gaussian perts classified as real
    rng = np.random.default_rng(0)
    probe = X_explain[rng.integers(0, len(X_explain), 2000)]
    for sigma in SIGMAS:
        perts = probe + rng.normal(size=probe.shape) * sigma
        lime_real = float((adv_lime.perturbation_identifier.predict_proba(
            perts[:, adv_lime.numerical_cols] if adv_lime.numerical_cols else perts
        )[:, 1] >= 0.5).mean())
        shap_real = float((adv_shap.perturbation_identifier.predict_proba(
            perts[:, adv_shap.numerical_cols] if adv_shap.numerical_cols else perts
        )[:, 1] >= 0.5).mean())
        print(f"σ={sigma}:  LIME detector {lime_real*100:.1f}% real, "
              f"SHAP detector {shap_real*100:.1f}% real")

    # Run rung-3 at each σ for both scaffolds
    results = {"n": int(len(layer)), "n_high": n_high, "n_low": n_low, "cells": []}
    for scaff_name, scaff in [("lime", adv_lime), ("shap", adv_shap)]:
        for sigma in SIGMAS:
            print(f"\n── {scaff_name.upper()} scaffold × Gauss σ={sigma} ──")
            t0 = time.time()
            E = gaussian_fingerprints_proba(scaff, X_explain, sigma)
            A = cosine_knn(E, K_NEIGHBOURS)
            cell = fit_cell(A, layer)
            dt = time.time() - t0
            print(f"  plain={cell['plain_nmi_mean']:.3f}±{cell['plain_nmi_std']:.3f}  "
                  f"pEx={cell['pex_nmi_mean']:.3f}±{cell['pex_nmi_std']:.3f}  "
                  f"Δ={cell['delta_mean']:+.3f}  ratio={cell['C_ratio']:.2f}  "
                  f"edges={cell['n_edges']}  ({dt:.0f}s)")
            cell.update(scaffold=scaff_name, sigma=sigma)
            results["cells"].append(cell)
            np.save(out / f"fp_{scaff_name}_sigma{sigma}.npy", E)
            np.save(out / f"graph_{scaff_name}_sigma{sigma}.npy", A)

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out / 'results.json'}")


if __name__ == "__main__":
    main()
