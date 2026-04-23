"""Noisy-layer stress test: flip fraction p of layer labels before fitting pEx.

Plain SBM never sees the layer — its NMI is unchanged across noise levels
(reported once as reference). pEx's prior is now conditioned on a corrupted
layer; the question is how gracefully its NMI degrades.

If pEx still beats plain at p = 0.2 (i.e., a noisy survey of demographic
membership), that's a usable method in real auditing settings.

Output: results/sbm_poc/stress_noisy_layer/summary.json + plot.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import build_knn_similarity_graph, fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


FEATURE = "sqft_lot"
EXPLAINER = "lime"
THRESHOLD = 9361.0
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def load_cell():
    honest = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / f"{EXPLAINER}_values.npy")
    adv = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "adv" / f"adv_{EXPLAINER}_values.npy")
    test_idx = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    return honest, adv, layer


def flip_labels(layer: np.ndarray, p: float, rng) -> np.ndarray:
    mask = rng.random(len(layer)) < p
    out = layer.copy()
    out[mask] = 1 - out[mask]
    return out


def fit_plain_multi(A, K, layer_true, n_seeds):
    nmis = []
    for s in range(n_seeds):
        sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer_true))
    return np.array(nmis)


def fit_pex_multi(A, K, layer_noisy, layer_true, n_seeds):
    nmis = []
    for s in range(n_seeds):
        sbm = fit_pex_sbm(A, layer=layer_noisy, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                          seed=s, verbose=False)
        # NMI is always evaluated vs the TRUE layer — we're testing degradation under input noise
        nmis.append(layer_separation(sbm.z, layer_true))
    return np.array(nmis)


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_noisy_layer"
    out_dir.mkdir(parents=True, exist_ok=True)

    honest, adv, layer = load_cell()
    print(f"cell: {FEATURE} × {EXPLAINER}   n={len(layer)}  "
          f"high={int(layer.sum())}  low={int((layer == 0).sum())}")

    A = build_knn_similarity_graph(adv, k=K_NEIGHBOURS, symmetric=True)

    print("\nBaseline plain SBM (layer-free) × 5 seeds...")
    plain_nmi = fit_plain_multi(A, K, layer, N_SEEDS)
    print(f"  plain NMI = {plain_nmi.mean():.3f} ± {plain_nmi.std():.3f}")

    results = {"K": K, "feature": FEATURE, "explainer": EXPLAINER,
               "plain_nmi_mean": float(plain_nmi.mean()),
               "plain_nmi_std":  float(plain_nmi.std()),
               "noise_levels": []}

    rng_master = np.random.default_rng(0)
    for p in NOISE_LEVELS:
        print(f"\n══ noise p = {p:.2f} ══")
        seed_rngs = [np.random.default_rng(rng_master.integers(0, 10**6)) for _ in range(N_SEEDS)]
        pex_nmis = []
        for s, r in enumerate(seed_rngs):
            noisy = flip_labels(layer, p, r)
            flips = int((noisy != layer).sum())
            sbm = fit_pex_sbm(A, layer=noisy, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                              seed=s, verbose=False)
            nmi = layer_separation(sbm.z, layer)
            pex_nmis.append(nmi)
            print(f"  seed {s}: flipped {flips}/{len(layer)} -> NMI={nmi:.3f}")
        pex_nmis = np.array(pex_nmis)
        print(f"  pEx NMI = {pex_nmis.mean():.3f} ± {pex_nmis.std():.3f} "
              f"(vs plain {plain_nmi.mean():.3f})")
        results["noise_levels"].append({
            "p": p,
            "pex_nmi_mean": float(pex_nmis.mean()),
            "pex_nmi_std":  float(pex_nmis.std()),
            "delta_vs_plain": float(pex_nmis.mean() - plain_nmi.mean()),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    ps = [r["p"] for r in results["noise_levels"]]
    pex_m = [r["pex_nmi_mean"] for r in results["noise_levels"]]
    pex_s = [r["pex_nmi_std"]  for r in results["noise_levels"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(ps, pex_m, yerr=pex_s, marker="s", color="#de2d26",
                label="pEx-SBM (noisy layer)", capsize=3)
    ax.axhline(plain_nmi.mean(), color="#3182bd", ls="--",
               label=f"plain SBM (layer-free, {plain_nmi.mean():.3f})")
    ax.fill_between(ps,
                    plain_nmi.mean() - plain_nmi.std(),
                    plain_nmi.mean() + plain_nmi.std(),
                    color="#3182bd", alpha=0.15)
    ax.set_xlabel("label-flip probability p")
    ax.set_ylabel("layer-separation NMI (vs TRUE layer)")
    ax.set_title(f"Noisy-layer stress — {FEATURE} / {EXPLAINER.upper()} / K={K}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "noisy_layer_plot.png", dpi=160)
    print(f"Saved {out_dir/'noisy_layer_plot.png'}")


if __name__ == "__main__":
    main()
