"""γ-sweep stress test: does pEx show an effect when the prior has teeth?

The α-sweep (weak_signal_stress) showed pEx ≡ plain at every attack strength.
Hypothesis: γ=1 makes the prior too weak — per-node log-prior difference
between pEx and plain is ≤ log(n_high + 1) ≈ 3.9 nats, while the Beta-Binomial
edge term contributes tens to hundreds of nats. The prior is invisible.

This script varies γ on the pEx side only, with the likelihood held fixed
(sqft_lot LIME, full attack, K=4). If pEx can ever dominate plain, we expect
it at γ ≥ 10: the layer-conditional count term begins to rival the edge term.

Output: results/sbm_poc/stress_gamma/summary.json + plot.
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
N_SEEDS = 3
GAMMAS = [0.1, 1.0, 5.0, 20.0, 100.0]


def load_cell():
    honest = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / f"{EXPLAINER}_values.npy")
    adv = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "adv" / f"adv_{EXPLAINER}_values.npy")
    test_idx = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    return adv, layer


def fit_plain(A, K, layer, seed, gamma=1.0):
    sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                  gamma=gamma, seed=seed, verbose=False)
    return layer_separation(sbm.z, layer)


def fit_pex(A, K, layer, seed, gamma=1.0):
    sbm = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                      gamma=gamma, seed=seed, verbose=False)
    return layer_separation(sbm.z, layer)


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_gamma"
    out_dir.mkdir(parents=True, exist_ok=True)
    adv, layer = load_cell()
    A = build_knn_similarity_graph(adv, k=K_NEIGHBOURS, symmetric=True)
    print(f"cell: {FEATURE} × {EXPLAINER}  n={len(layer)}  "
          f"high={int(layer.sum())}  low={int((layer == 0).sum())}  K={K}")

    results = {"K": K, "feature": FEATURE, "explainer": EXPLAINER, "gammas": []}

    for gamma in GAMMAS:
        print(f"\n══ γ = {gamma} ══")
        plain_nmis = [fit_plain(A, K, layer, s, gamma=gamma) for s in range(N_SEEDS)]
        pex_nmis   = [fit_pex(A,   K, layer, s, gamma=gamma) for s in range(N_SEEDS)]
        plain_nmis = np.array(plain_nmis)
        pex_nmis   = np.array(pex_nmis)
        print(f"  plain NMI = {plain_nmis.mean():.3f} ± {plain_nmis.std():.3f}")
        print(f"  pEx   NMI = {pex_nmis.mean():.3f} ± {pex_nmis.std():.3f}")
        print(f"  Δ(pEx − plain) = {pex_nmis.mean() - plain_nmis.mean():+.3f}")
        results["gammas"].append({
            "gamma": gamma,
            "plain_nmi_mean": float(plain_nmis.mean()),
            "plain_nmi_std":  float(plain_nmis.std()),
            "pex_nmi_mean":   float(pex_nmis.mean()),
            "pex_nmi_std":    float(pex_nmis.std()),
            "delta_mean":     float(pex_nmis.mean() - plain_nmis.mean()),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    gs = [r["gamma"] for r in results["gammas"]]
    pm = [r["plain_nmi_mean"] for r in results["gammas"]]
    ps = [r["plain_nmi_std"]  for r in results["gammas"]]
    em = [r["pex_nmi_mean"]   for r in results["gammas"]]
    es = [r["pex_nmi_std"]    for r in results["gammas"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(gs, pm, yerr=ps, marker="o", label="plain SBM", color="#3182bd", capsize=3)
    ax.errorbar(gs, em, yerr=es, marker="s", label="pEx-SBM",   color="#de2d26", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("γ (Dirichlet concentration)")
    ax.set_ylabel("layer-separation NMI")
    ax.set_title(f"γ-sweep — {FEATURE} / {EXPLAINER.upper()} / K={K}, full attack")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(out_dir / "gamma_plot.png", dpi=160)
    print(f"Saved {out_dir/'gamma_plot.png'}")


if __name__ == "__main__":
    main()
