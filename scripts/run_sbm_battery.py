"""Battery of SBM experiments: multi-seed plain SBM + pEx-SBM + K-sweep.

Cells
-----
- pct_black_latino × {SHAP, LIME}   (layer: pct_black_latino > 0.25)
- sqft_lot         × {SHAP, LIME}   (layer: sqft_lot > 9361, P75)
- view             × {SHAP, LIME}   (layer: view > 0)
- waterfront       × {SHAP, LIME}   (layer: waterfront > 0)

For each cell:
  * Build honest + adversarial k-NN graphs once
  * Fit plain SBM at seeds 0..N-1 (reusing graph)
  * Fit pEx-SBM once (seed 0)
  * Report mean ± std layer-sep NMI (plain, both honest and adv)
    and single pEx-SBM NMI

K-sweep: sqft_lot LIME, K ∈ {2,3,4,6}, single seed.

Output: results/sbm_poc/battery/summary.json + battery_plot.png
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
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


CELLS = [
    dict(feature="pct_black_latino", scaffold="scaffold", threshold=0.25, moran=0.76),
    dict(feature="sqft_lot",         scaffold="scaffold_sqft_lot",        threshold=9361.0, moran=0.47),
    dict(feature="view",             scaffold="scaffold_view",            threshold=0.5,    moran=0.24),
    dict(feature="waterfront",       scaffold="scaffold_waterfront",      threshold=0.5,    moran=0.05),
]
EXPLAINERS = ["shap", "lime"]
N_SEEDS = 5
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
K_SWEEP = [2, 3, 4, 6]


def _default_honest_dir(cell):
    if cell["feature"] == "pct_black_latino":
        return RESULTS_DIR / "explanations"
    return RESULTS_DIR / cell["scaffold"] / "honest"


def _adv_dir(cell):
    if cell["feature"] == "pct_black_latino":
        return RESULTS_DIR / "scaffold"
    return RESULTS_DIR / cell["scaffold"] / "adv"


def load_cell(cell, explainer):
    honest_dir = _default_honest_dir(cell)
    adv_dir = _adv_dir(cell)
    honest = np.load(honest_dir / f"{explainer}_values.npy")
    adv = np.load(adv_dir / f"adv_{explainer}_values.npy")
    test_idx = np.load(honest_dir / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1]-honest.shape[1]))])
    elif honest.shape[1] > adv.shape[1]:
        adv = np.hstack([adv, np.zeros((adv.shape[0], honest.shape[1]-adv.shape[1]))])

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[cell["feature"]].values[test_idx]
    layer = (col > cell["threshold"]).astype(int)
    return honest, adv, layer


def fit_plain_multiseed(A, K, n_seeds, layer):
    nmis = []
    withins = []
    acrosses = []
    for s in range(n_seeds):
        sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                      alpha=1.0, beta=1.0, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer))
        w, a = layer_coclustering(sbm.co_clustering_matrix(), layer)
        withins.append(w); acrosses.append(a)
    return np.array(nmis), np.array(withins), np.array(acrosses)


def fit_pex(A, K, layer, seed=0):
    sbm = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                      alpha=1.0, beta=1.0, gamma=1.0, seed=seed, verbose=False)
    w, a = layer_coclustering(sbm.co_clustering_matrix(), layer)
    return layer_separation(sbm.z, layer), w, a


def fit_pex_multiseed(A, K, layer, n_seeds):
    nmis, withins, acrosses = [], [], []
    for s in range(n_seeds):
        nmi, w, a = fit_pex(A, K, layer, seed=s)
        nmis.append(nmi); withins.append(w); acrosses.append(a)
    return np.array(nmis), np.array(withins), np.array(acrosses)


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "battery"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"cells": [], "k_sweep": []}

    for cell in CELLS:
        for expl in EXPLAINERS:
            label = f"{cell['feature']}_{expl}"
            print(f"\n══ {label}  I={cell['moran']}  thresh>{cell['threshold']} ══")
            honest, adv, layer = load_cell(cell, expl)
            print(f"  n={len(layer)}  high={int(layer.sum())}  low={int((layer==0).sum())}")
            if int(layer.sum()) < 2 or int((layer == 0).sum()) < 2:
                print("  SKIP: degenerate layer split")
                continue

            A_h = build_knn_similarity_graph(honest, k=K_NEIGHBOURS, symmetric=True)
            A_a = build_knn_similarity_graph(adv,    k=K_NEIGHBOURS, symmetric=True)

            print(f"  plain SBM ×{N_SEEDS} seeds (honest)...")
            nmi_h, wh, ah = fit_plain_multiseed(A_h, K, N_SEEDS, layer)
            print(f"    honest NMI = {nmi_h.mean():.3f} ± {nmi_h.std():.3f}")
            print(f"  plain SBM ×{N_SEEDS} seeds (adv)...")
            nmi_a, wa, aa = fit_plain_multiseed(A_a, K, N_SEEDS, layer)
            print(f"    adv    NMI = {nmi_a.mean():.3f} ± {nmi_a.std():.3f}")

            print(f"  pEx-SBM ×{N_SEEDS} seeds (honest)...")
            pex_nmi_h, pex_wh, pex_ah = fit_pex_multiseed(A_h, K, layer, N_SEEDS)
            print(f"    pEx honest NMI = {pex_nmi_h.mean():.3f} ± {pex_nmi_h.std():.3f}")
            print(f"  pEx-SBM ×{N_SEEDS} seeds (adv)...")
            pex_nmi_a, pex_wa, pex_aa = fit_pex_multiseed(A_a, K, layer, N_SEEDS)
            print(f"    pEx adv    NMI = {pex_nmi_a.mean():.3f} ± {pex_nmi_a.std():.3f}")

            results["cells"].append({
                "feature": cell["feature"], "explainer": expl, "moran_I": cell["moran"],
                "n": int(len(layer)),
                "layer_high": int(layer.sum()), "layer_low": int((layer == 0).sum()),
                "plain": {
                    "honest_nmi_mean": float(nmi_h.mean()), "honest_nmi_std": float(nmi_h.std()),
                    "adv_nmi_mean":    float(nmi_a.mean()), "adv_nmi_std":    float(nmi_a.std()),
                    "honest_within_mean": float(wh.mean()), "honest_across_mean": float(ah.mean()),
                    "adv_within_mean":    float(wa.mean()), "adv_across_mean":    float(aa.mean()),
                },
                "pex": {
                    "honest_nmi_mean": float(pex_nmi_h.mean()),
                    "honest_nmi_std":  float(pex_nmi_h.std()),
                    "adv_nmi_mean":    float(pex_nmi_a.mean()),
                    "adv_nmi_std":     float(pex_nmi_a.std()),
                    "honest_within_mean": float(pex_wh.mean()),
                    "honest_across_mean": float(pex_ah.mean()),
                    "adv_within_mean":    float(pex_wa.mean()),
                    "adv_across_mean":    float(pex_aa.mean()),
                },
            })

    # K-sweep on sqft_lot LIME
    print("\n══ K-sweep: sqft_lot LIME ══")
    cell = CELLS[1]  # sqft_lot
    honest, adv, layer = load_cell(cell, "lime")
    A_h = build_knn_similarity_graph(honest, k=K_NEIGHBOURS, symmetric=True)
    A_a = build_knn_similarity_graph(adv,    k=K_NEIGHBOURS, symmetric=True)
    for k_ in K_SWEEP:
        plain_h, _, _ = fit_plain_multiseed(A_h, k_, N_SEEDS, layer)
        plain_a, _, _ = fit_plain_multiseed(A_a, k_, N_SEEDS, layer)
        pex_h,   _, _ = fit_pex_multiseed(A_h, k_, layer, N_SEEDS)
        pex_a,   _, _ = fit_pex_multiseed(A_a, k_, layer, N_SEEDS)
        print(f"  K={k_}: plain honest={plain_h.mean():.3f}±{plain_h.std():.3f} "
              f"adv={plain_a.mean():.3f}±{plain_a.std():.3f} | "
              f"pEx honest={pex_h.mean():.3f}±{pex_h.std():.3f} "
              f"adv={pex_a.mean():.3f}±{pex_a.std():.3f}")
        results["k_sweep"].append({
            "K": k_,
            "plain_honest_nmi_mean": float(plain_h.mean()), "plain_honest_nmi_std": float(plain_h.std()),
            "plain_adv_nmi_mean":    float(plain_a.mean()), "plain_adv_nmi_std":    float(plain_a.std()),
            "pex_honest_nmi_mean":   float(pex_h.mean()),   "pex_honest_nmi_std":   float(pex_h.std()),
            "pex_adv_nmi_mean":      float(pex_a.mean()),   "pex_adv_nmi_std":      float(pex_a.std()),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_dir/'summary.json'}")

    # Plot: separation-NMI bars (plain honest, plain adv, pEx honest, pEx adv) per cell
    cells = results["cells"]
    if cells:
        labels = [f"{c['feature'][:8]}\n{c['explainer']}\nI={c['moran_I']}" for c in cells]
        ph = [c["plain"]["honest_nmi_mean"] for c in cells]
        pa = [c["plain"]["adv_nmi_mean"]    for c in cells]
        ph_err = [c["plain"]["honest_nmi_std"] for c in cells]
        pa_err = [c["plain"]["adv_nmi_std"] for c in cells]
        exh = [c["pex"]["honest_nmi_mean"] for c in cells]
        exa = [c["pex"]["adv_nmi_mean"]    for c in cells]
        exh_err = [c["pex"]["honest_nmi_std"] for c in cells]
        exa_err = [c["pex"]["adv_nmi_std"]    for c in cells]
        x = np.arange(len(cells)); w = 0.2
        fig, ax = plt.subplots(figsize=(max(8, 1.2*len(cells)), 4.5))
        ax.bar(x-1.5*w, ph, w, yerr=ph_err, label="plain honest", color="#9ecae1", capsize=3)
        ax.bar(x-0.5*w, pa, w, yerr=pa_err, label="plain adv",    color="#3182bd", capsize=3)
        ax.bar(x+0.5*w, exh, w, yerr=exh_err, label="pEx honest",  color="#fcae91", capsize=3)
        ax.bar(x+1.5*w, exa, w, yerr=exa_err, label="pEx adv",     color="#de2d26", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("layer-separation NMI")
        ax.set_title(f"SBM vs pEx-SBM — K={K}, {N_SEEDS} seeds each")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "battery_plot.png", dpi=150)
        print(f"Saved {out_dir/'battery_plot.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
