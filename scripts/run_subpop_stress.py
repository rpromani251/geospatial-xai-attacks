"""Subpopulation stress: attacker only biases a fraction p of layer-high nodes.

Rationale
---------
α-sweep scaled the attack strength uniformly. Subpop instead dilutes *coverage*:
fraction p of layer-high nodes get adversarial explanations, the rest get
honest ones. Layer-low nodes always get honest.

The layer label is still "true" for all high nodes — even those whose
explanation was left honest. This is the scenario where pEx could win by
construction: half the high-layer nodes look "honest" to the edge signal, but
the prior still sees them as high.

Output: results/sbm_poc/stress_subpop/{summary.json, subpop_plot.png}
"""

from __future__ import annotations

import sys, json
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
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_matrices():
    base = RESULTS_DIR / f"scaffold_{FEATURE}"
    honest = np.load(base / "honest" / f"{EXPLAINER}_values.npy")
    adv    = np.load(base / "adv"    / f"adv_{EXPLAINER}_values.npy")
    test_idx = np.load(base / "honest" / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    return honest, adv, layer


def mix(honest, adv, layer, p, rng):
    """Return explanation matrix where fraction p of layer-high nodes use adv."""
    X = honest.copy()
    high = np.where(layer == 1)[0]
    n_attack = int(round(p * len(high)))
    if n_attack > 0:
        chosen = rng.choice(high, size=n_attack, replace=False)
        X[chosen] = adv[chosen]
    return X


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_subpop"
    out_dir.mkdir(parents=True, exist_ok=True)
    honest, adv, layer = load_matrices()
    print(f"n={len(layer)}  high={int(layer.sum())}  low={int((layer==0).sum())}")

    results = {"K": K, "feature": FEATURE, "explainer": EXPLAINER, "fractions": []}

    for p in FRACTIONS:
        print(f"\n══ p = {p} ══")
        plain_nmis, pex_nmis = [], []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 + s)
            X = mix(honest, adv, layer, p, rng)
            A = build_knn_similarity_graph(X, k=K_NEIGHBOURS, symmetric=True)
            sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                          gamma=1.0, seed=s, verbose=False)
            pex = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                              thin=5, gamma=1.0, seed=s, verbose=False)
            plain_nmis.append(layer_separation(sbm.z, layer))
            pex_nmis.append(layer_separation(pex.z, layer))
        plain_nmis = np.array(plain_nmis)
        pex_nmis   = np.array(pex_nmis)
        print(f"  plain NMI = {plain_nmis.mean():.4f} ± {plain_nmis.std():.4f}")
        print(f"  pEx   NMI = {pex_nmis.mean():.4f} ± {pex_nmis.std():.4f}")
        print(f"  Δ = {pex_nmis.mean() - plain_nmis.mean():+.4f}")
        results["fractions"].append({
            "p": p,
            "plain_nmi_mean": float(plain_nmis.mean()),
            "plain_nmi_std":  float(plain_nmis.std()),
            "pex_nmi_mean":   float(pex_nmis.mean()),
            "pex_nmi_std":    float(pex_nmis.std()),
            "delta_mean":     float(pex_nmis.mean() - plain_nmis.mean()),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    ps  = [r["p"] for r in results["fractions"]]
    pm  = [r["plain_nmi_mean"] for r in results["fractions"]]
    pstd= [r["plain_nmi_std"]  for r in results["fractions"]]
    em  = [r["pex_nmi_mean"]   for r in results["fractions"]]
    estd= [r["pex_nmi_std"]    for r in results["fractions"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ps, pm, yerr=pstd, marker="o", label="plain SBM", color="#3182bd", capsize=3)
    ax.errorbar(ps, em, yerr=estd, marker="s", label="pEx-SBM",   color="#de2d26", capsize=3)
    ax.set_xlabel("fraction of layer-high nodes with adversarial explanation")
    ax.set_ylabel("layer-separation NMI")
    ax.set_title(f"subpop sweep — {FEATURE} / {EXPLAINER.upper()} / K={K}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "subpop_plot.png", dpi=160)
    print(f"Saved {out_dir/'subpop_plot.png'}")


if __name__ == "__main__":
    main()
