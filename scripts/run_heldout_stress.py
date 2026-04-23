"""Held-out layer stress: does pEx's prior help infer block membership for
nodes whose layer label is hidden at fit time?

This is the only task plain-SBM literally cannot do (no layer channel). We
nonetheless fit both for comparison:
  * plain SBM: layer never used; measure block-vs-layer NMI on held-out set.
  * pEx-SBM: layer used only for the non-held-out subset (held-out nodes
    pass layer = -1, which the sampler treats as plain Dirichlet for those
    nodes while still using per-layer counts for observed ones).

For each hold-out fraction f ∈ {0.0, 0.1, 0.25, 0.5, 0.75}, stratified by
true layer. NMI reported on the held-out subset only (primary metric) and
on the observed subset (sanity).

Output: results/sbm_poc/stress_heldout/{summary.json, heldout_plot.png}
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
HOLDOUT_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75]


def load_cell():
    base = RESULTS_DIR / f"scaffold_{FEATURE}"
    adv = np.load(base / "adv" / f"adv_{EXPLAINER}_values.npy")
    test_idx = np.load(base / "honest" / "test_indices.npy")
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    return adv, layer


def stratified_holdout(layer, f, rng):
    """Return a boolean mask (True = held out), stratified per layer."""
    hold = np.zeros(len(layer), dtype=bool)
    for ell in np.unique(layer):
        idx = np.where(layer == ell)[0]
        n_hold = int(round(f * len(idx)))
        if n_hold > 0:
            chosen = rng.choice(idx, size=n_hold, replace=False)
            hold[chosen] = True
    return hold


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    adv, layer = load_cell()
    A = build_knn_similarity_graph(adv, k=K_NEIGHBOURS, symmetric=True)
    print(f"cell: {FEATURE} × {EXPLAINER}  n={len(layer)}  "
          f"high={int(layer.sum())}  low={int((layer==0).sum())}  K={K}")

    results = {"K": K, "feature": FEATURE, "explainer": EXPLAINER, "fractions": []}

    for f in HOLDOUT_FRACTIONS:
        print(f"\n══ hold-out fraction f = {f} ══")
        plain_all, plain_held, plain_obs = [], [], []
        pex_all,   pex_held,   pex_obs   = [], [], []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(2000 + s)
            hold = stratified_holdout(layer, f, rng)
            obs  = ~hold
            layer_partial = layer.copy()
            layer_partial[hold] = -1

            # plain SBM — identical across held-out fractions (never uses labels)
            sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                          gamma=1.0, seed=s, verbose=False)
            # pEx — sees partial labels
            pex = fit_pex_sbm(A, layer=layer_partial, K=K,
                              n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                              gamma=1.0, seed=s, verbose=False)
            plain_all.append(layer_separation(sbm.z, layer))
            pex_all.append(  layer_separation(pex.z, layer))
            if hold.any():
                plain_held.append(layer_separation(sbm.z[hold], layer[hold]))
                pex_held.append(  layer_separation(pex.z[hold], layer[hold]))
            if obs.any():
                plain_obs.append(layer_separation(sbm.z[obs], layer[obs]))
                pex_obs.append(  layer_separation(pex.z[obs], layer[obs]))

        def stat(xs):
            a = np.array(xs) if xs else np.array([np.nan])
            return float(a.mean()), float(a.std())

        p_all_m, p_all_s = stat(plain_all)
        p_hld_m, p_hld_s = stat(plain_held)
        p_obs_m, p_obs_s = stat(plain_obs)
        e_all_m, e_all_s = stat(pex_all)
        e_hld_m, e_hld_s = stat(pex_held)
        e_obs_m, e_obs_s = stat(pex_obs)

        print(f"  plain  NMI(all/held/obs) = {p_all_m:.4f} / {p_hld_m:.4f} / {p_obs_m:.4f}")
        print(f"  pEx    NMI(all/held/obs) = {e_all_m:.4f} / {e_hld_m:.4f} / {e_obs_m:.4f}")
        print(f"  Δ(pEx − plain) on held   = {e_hld_m - p_hld_m:+.4f}")

        results["fractions"].append({
            "f": f,
            "plain_all_mean": p_all_m, "plain_all_std": p_all_s,
            "plain_heldout_mean": p_hld_m, "plain_heldout_std": p_hld_s,
            "plain_obs_mean": p_obs_m, "plain_obs_std": p_obs_s,
            "pex_all_mean":   e_all_m, "pex_all_std":   e_all_s,
            "pex_heldout_mean": e_hld_m, "pex_heldout_std": e_hld_s,
            "pex_obs_mean":   e_obs_m, "pex_obs_std":   e_obs_s,
            "delta_heldout":  e_hld_m - p_hld_m,
        })

    with open(out_dir / "summary.json", "w") as f_:
        json.dump(results, f_, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    fs  = [r["f"] for r in results["fractions"]]
    p_h = [r["plain_heldout_mean"] for r in results["fractions"]]
    p_hs= [r["plain_heldout_std"]  for r in results["fractions"]]
    e_h = [r["pex_heldout_mean"]   for r in results["fractions"]]
    e_hs= [r["pex_heldout_std"]    for r in results["fractions"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(fs, p_h, yerr=p_hs, marker="o", label="plain SBM (held-out)",
                color="#3182bd", capsize=3)
    ax.errorbar(fs, e_h, yerr=e_hs, marker="s", label="pEx-SBM (held-out)",
                color="#de2d26", capsize=3)
    ax.set_xlabel("held-out fraction f")
    ax.set_ylabel("NMI on held-out subset")
    ax.set_title(f"held-out layer inference — {FEATURE} / {EXPLAINER.upper()} / K={K}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "heldout_plot.png", dpi=160)
    print(f"Saved {out_dir/'heldout_plot.png'}")


if __name__ == "__main__":
    main()
