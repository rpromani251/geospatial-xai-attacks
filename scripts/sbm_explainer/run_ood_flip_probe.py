"""Probe 1: OOD-detector flip test.

Prediction of the stated rung-3 mechanism: the "Gauss+LIME is the only
non-null cell" result is driven by the OOD detector's training
distribution, not by SHAP vs LIME explainer identity. Swapping detectors
between scaffolds should flip the result:

    Gauss+SHAP_scaffold + LIME-detector  →  predicted non-null (flip)
    Gauss+LIME_scaffold + SHAP-detector  →  predicted null (flip)

If both flips confirm, mechanism established. If either fails, rung-3
story needs a different explanation.

Rebuilds biased/innocuous models identically to run_fingerprint_sbm.py;
only the scaffold's `ood_detector` field differs.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.attacks.scaffold import AdversarialScaffold
from src.attacks.ood_detector import (
    train_lime_ood_detector, train_shap_ood_detector,
)
from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

FEATURE = "sqft_lot"
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
N_GAUSS = 20
GAUSS_SIGMA_FRAC = 0.1


def gaussian_fingerprints(scaffold, X, feature_std, n_pert=N_GAUSS, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, n_pert + 1))
    E[:, 0] = scaffold.predict(X)
    scale = GAUSS_SIGMA_FRAC * feature_std[None, :]
    for k in range(n_pert):
        delta = rng.normal(size=(n, d)) * scale
        E[:, k + 1] = scaffold.predict(X + delta)
    return E


def cosine_knn_graph(E, k):
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
    plain_nmis = np.asarray(plain_nmis)
    pex_nmis = np.asarray(pex_nmis)
    ratios = np.asarray(ratios)
    return dict(
        plain_nmi_mean=float(plain_nmis.mean()),
        plain_nmi_std=float(plain_nmis.std(ddof=1)),
        pex_nmi_mean=float(pex_nmis.mean()),
        pex_nmi_std=float(pex_nmis.std(ddof=1)),
        delta=float((pex_nmis - plain_nmis).mean()),
        C_ratio_mean=float(np.mean(ratios)),
        n_edges=int(A.sum() // 2),
    )


def main():
    out_dir = RESULTS_DIR / "sbm_explainer" / "ood_flip"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    X_full, y = load_housing(dataset="demographics", feature_columns=None,
                              target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
    df_sweep = X_full[list(baseline_features)]
    feature_names = list(df_sweep.columns)
    bias_col_idx = feature_names.index(FEATURE)
    base_feature_indices = [feature_names.index(f) for f in baseline_features]
    threshold = float(np.percentile(df_sweep[FEATURE].values, 75))
    penalty = cfg["biased_model"]["penalty"]

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"])
    X_train_np = X_train.values
    X_explain = X_test.iloc[:200]
    X_explain_np = X_explain.values
    test_indices = X_test.index.values[:200]

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_indices]
    layer = (col > threshold).astype(int)
    feature_std = X_train.std(ddof=1).values.astype(np.float64)

    honest = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    f_biased = BiasedModel(honest, base_feature_indices, bias_col_idx,
                           threshold=threshold, penalty=penalty)
    safe_indices = [i for i, f in enumerate(feature_names)
                    if f not in {feature_names[bias_col_idx], "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, cfg["innocuous_model"])

    print("Training both OOD detectors...")
    lime_detector, lime_diag = train_lime_ood_detector(
        X_train_np, **cfg["lime_ood"])
    print(f"  LIME OOD acc = {lime_diag['accuracy']:.4f}")
    shap_detector, shap_diag = train_shap_ood_detector(
        X_train_np, **cfg["shap_ood"])
    print(f"  SHAP OOD acc = {shap_diag['accuracy']:.4f}")

    # Check routing: what fraction of Gaussian perts does each detector
    # classify as "real" (in-manifold)?
    rng = np.random.default_rng(0)
    probe = X_explain_np.copy()
    n_probe = 2000
    idx = rng.integers(0, probe.shape[0], size=n_probe)
    delta = rng.normal(size=(n_probe, probe.shape[1])) * GAUSS_SIGMA_FRAC * feature_std
    perts = probe[idx] + delta
    lime_real = float((lime_detector.predict_proba(perts)[:, 1] >= 0.5).mean())
    shap_real = float((shap_detector.predict_proba(perts)[:, 1] >= 0.5).mean())
    print(f"\nRouting of Gaussian perts (σ=0.1·feat_std):")
    print(f"  LIME detector classifies {lime_real*100:.1f}% as real → routed to BIASED")
    print(f"  SHAP detector classifies {shap_real*100:.1f}% as real → routed to BIASED")

    # Four scaffolds: {SHAP, LIME} scaffold × {default, swapped} detector
    def make_scaffold(detector):
        sc = AdversarialScaffold(f_biased, psi)
        sc.ood_detector = detector
        sc.ood_diagnostics = {}
        return sc

    scaffolds = {
        "LIME_detector (default on LIME scaffold)": make_scaffold(lime_detector),
        "SHAP_detector (default on SHAP scaffold)": make_scaffold(shap_detector),
        "LIME_detector swapped onto SHAP scaffold": make_scaffold(lime_detector),
        "SHAP_detector swapped onto LIME scaffold": make_scaffold(shap_detector),
    }
    # Scaffolds 0/2 and 1/3 are behaviorally identical (same biased + innocuous
    # + detector). The scaffold "identity" is pure bookkeeping in our code —
    # what actually drives routing is the detector.
    # So we really only have TWO distinct behavioral configurations here.

    # Simplify: run the two distinct configs
    configs = {
        "LIME_OOD detector": lime_detector,
        "SHAP_OOD detector": shap_detector,
    }

    results = {
        "threshold": threshold, "penalty": penalty,
        "n_explain": 200, "n_high": int(layer.sum()),
        "n_low": int((layer == 0).sum()),
        "routing": {"lime_accepts_gauss_pct": lime_real * 100,
                    "shap_accepts_gauss_pct": shap_real * 100},
        "cells": {},
    }
    for name, det in configs.items():
        print(f"\n── {name} × Gaussian fingerprints ──")
        sc = make_scaffold(det)
        E = gaussian_fingerprints(sc, X_explain_np, feature_std, seed=0)
        np.save(out_dir / f"fp_{name.split('_')[0].lower()}_det.npy", E)
        A = cosine_knn_graph(E, K_NEIGHBOURS)
        np.save(out_dir / f"graph_{name.split('_')[0].lower()}_det.npy", A)
        print(f"  edges = {A.sum() // 2}")
        cell = fit_cell(A, layer)
        print(f"  plain NMI = {cell['plain_nmi_mean']:.3f} ± {cell['plain_nmi_std']:.3f}")
        print(f"  pEx   NMI = {cell['pex_nmi_mean']:.3f} ± {cell['pex_nmi_std']:.3f}")
        print(f"  Δ = {cell['delta']:+.3f}   C ratio = {cell['C_ratio_mean']:.2f}")
        results["cells"][name] = cell

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n── Probe 1 summary ──")
    lime_cell = results["cells"]["LIME_OOD detector"]
    shap_cell = results["cells"]["SHAP_OOD detector"]
    print(f"LIME detector:  Δ = {lime_cell['delta']:+.3f}, "
          f"C ratio = {lime_cell['C_ratio_mean']:.2f}, "
          f"plain NMI = {lime_cell['plain_nmi_mean']:.3f}")
    print(f"SHAP detector:  Δ = {shap_cell['delta']:+.3f}, "
          f"C ratio = {shap_cell['C_ratio_mean']:.2f}, "
          f"plain NMI = {shap_cell['plain_nmi_mean']:.3f}")
    print(f"\nPrediction of stated mechanism:")
    print(f"  LIME_detector → non-null (Δ > 0, C ratio 2-4)")
    print(f"  SHAP_detector → null      (Δ ≈ 0, C ratio ≈ 1)")
    print(f"\nSaved {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
