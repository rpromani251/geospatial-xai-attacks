"""Weak-signal stress test: attenuate the attack and trace pEx vs plain.

Rationale (prior-vs-likelihood decomposition)
---------------------------------------------
pEx and plain SBM share the Beta-Binomial edge likelihood. They differ only
in the Dirichlet-Categorical block-prior term (per-layer vs. global).
Their NMI gap can only materialise when the log edge-likelihood ratio
between competing partitions is in the intermediate regime — strong enough
to rule out trivial partitions, weak enough that the prior still moves the
posterior.

Our full-attack battery lives in the saturated regime (NMI ≈ 0.54 at K=4,
0.92 at K=2 on sqft_lot LIME). To test whether pEx can help in the middle
regime, we weaken the edge signal by interpolating the explanation matrix:

    X(α) = α · X_adv + (1 − α) · X_honest,    α ∈ [0, 1]

α = 1 is the full attack; α = 0 is the honest model. Sweeping α traces the
likelihood ratio from saturated (α large) to absent (α → 0), and pEx should
show its effect in the middle band.

Output: results/sbm_poc/stress_weak_signal/summary.json + plot.
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
ALPHAS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.05]


def load_honest_adv():
    honest = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / f"{EXPLAINER}_values.npy")
    adv = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "adv" / f"adv_{EXPLAINER}_values.npy")
    test_idx = np.load(RESULTS_DIR / f"scaffold_{FEATURE}" / "honest" / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    elif honest.shape[1] > adv.shape[1]:
        adv = np.hstack([adv, np.zeros((adv.shape[0], honest.shape[1] - adv.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    return honest, adv, layer


def log_likelihood_ratio_proxy(A: np.ndarray, layer: np.ndarray) -> float:
    """Crude proxy for log Λ: log-odds of within-layer vs cross-layer edge density.

    Under the Bernoulli SBM with K=2 and layer as the target partition,
        p_within  = mean(A[i,j] | l_i = l_j)
        p_across  = mean(A[i,j] | l_i ≠ l_j)
    The per-edge log-odds log(p_within (1 - p_across) / (p_across (1 - p_within)))
    times the number of observed edges approximates the log-Bayes-factor of
    the layer partition vs. a random partition (up to a constant).
    """
    n = len(layer)
    same = layer[:, None] == layer[None, :]
    diag = np.eye(n, dtype=bool)
    within_mask = same & ~diag
    across_mask = ~same
    n_edges = int(A.sum())
    p_w = A[within_mask].mean() if within_mask.any() else 0.0
    p_a = A[across_mask].mean() if across_mask.any() else 0.0
    eps = 1e-12
    logodds = np.log((p_w + eps) * (1 - p_a + eps) / ((p_a + eps) * (1 - p_w + eps)))
    return float(n_edges * logodds / 2), float(p_w), float(p_a)


def fit_plain_multi(A, K, layer, n_seeds):
    nmis = []
    for s in range(n_seeds):
        sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                      alpha=1.0, beta=1.0, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer))
    return np.array(nmis)


def fit_pex_multi(A, K, layer, n_seeds):
    nmis = []
    for s in range(n_seeds):
        sbm = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                          alpha=1.0, beta=1.0, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer))
    return np.array(nmis)


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_weak_signal"
    out_dir.mkdir(parents=True, exist_ok=True)

    honest, adv, layer = load_honest_adv()
    n = len(layer)
    print(f"cell: {FEATURE} × {EXPLAINER}   n={n}  high={int(layer.sum())}  low={int((layer == 0).sum())}")

    results = {"K": K, "alphas": [], "feature": FEATURE, "explainer": EXPLAINER}

    for alpha in ALPHAS:
        print(f"\n══ α = {alpha} ══")
        X = alpha * adv + (1 - alpha) * honest
        A = build_knn_similarity_graph(X, k=K_NEIGHBOURS, symmetric=True)
        logL, p_w, p_a = log_likelihood_ratio_proxy(A, layer)
        print(f"  p_within={p_w:.3f}  p_across={p_a:.3f}  log-Λ proxy ≈ {logL:.1f}")

        plain_nmi = fit_plain_multi(A, K, layer, N_SEEDS)
        pex_nmi   = fit_pex_multi(A, K, layer, N_SEEDS)
        print(f"  plain NMI = {plain_nmi.mean():.3f} ± {plain_nmi.std():.3f}")
        print(f"  pEx   NMI = {pex_nmi.mean():.3f} ± {pex_nmi.std():.3f}")
        print(f"  Δ(pEx − plain) = {pex_nmi.mean() - plain_nmi.mean():+.3f}")

        results["alphas"].append({
            "alpha": alpha,
            "log_lambda_proxy": logL,
            "p_within": p_w,
            "p_across": p_a,
            "plain_nmi_mean": float(plain_nmi.mean()),
            "plain_nmi_std":  float(plain_nmi.std()),
            "pex_nmi_mean":   float(pex_nmi.mean()),
            "pex_nmi_std":    float(pex_nmi.std()),
            "delta_mean":     float(pex_nmi.mean() - plain_nmi.mean()),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    # Plot
    alphas = [r["alpha"] for r in results["alphas"]]
    plain_m = [r["plain_nmi_mean"] for r in results["alphas"]]
    plain_s = [r["plain_nmi_std"] for r in results["alphas"]]
    pex_m   = [r["pex_nmi_mean"] for r in results["alphas"]]
    pex_s   = [r["pex_nmi_std"]  for r in results["alphas"]]
    delta_m = [r["delta_mean"]   for r in results["alphas"]]
    logL    = [r["log_lambda_proxy"] for r in results["alphas"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.errorbar(alphas, plain_m, yerr=plain_s, marker="o", label="plain SBM", color="#3182bd", capsize=3)
    ax.errorbar(alphas, pex_m,   yerr=pex_s,   marker="s", label="pEx-SBM",   color="#de2d26", capsize=3)
    ax.set_xlabel("α (attack strength)")
    ax.set_ylabel("layer-separation NMI")
    ax.set_title(f"Weak-signal stress: {FEATURE} / {EXPLAINER.upper()} / K={K}")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(logL, delta_m, marker="o", color="#31a354")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("log-Λ proxy (edge likelihood strength)")
    ax.set_ylabel("NMI(pEx) − NMI(plain)")
    ax.set_title("pEx advantage vs. likelihood strength")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "stress_plot.png", dpi=160)
    print(f"Saved {out_dir/'stress_plot.png'}")


if __name__ == "__main__":
    main()
