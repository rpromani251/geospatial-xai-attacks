"""Penalty sweep: regenerate scaffold explanations at a range of scaffold
penalties, then fit plain SBM and pEx-SBM on the resulting LIME + SHAP
similarity graphs.

Motivation
----------
α-sweep attenuated *explanation vectors* post-hoc via
    X(α) = α X_adv + (1-α) X_honest.
That's a synthetic mixture, not a genuinely weaker attacker. This sweep
regenerates adversarial explanations by retraining the biased model with
smaller penalties, producing the full "natural regime-3" trajectory:

    penalty ∈ {0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20}
    (0.03 increments in the transition window 0.01–0.19, plus cached
    bracket points 0.05 and 0.20 for continuity with the prior sweep)

Output: results/sbm_poc/stress_penalty/{summary.json, penalty_plot.png}

For each penalty:
  * regenerate scaffold → results/scaffold_sqft_lot_p{penalty:.2f}/
  * build 8-NN similarity graph for LIME and for SHAP
  * fit plain SBM (5 seeds, K=4) and pEx-SBM (5 seeds, K=4)
  * record honest + adv NMI means ± stds, Δ(pEx − plain)
"""

from __future__ import annotations

import sys, json, subprocess
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
THRESHOLD = 9361.0
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
PENALTIES = [0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20]
EXPLAINERS = ["lime", "shap"]


def penalty_dir(p: float) -> Path:
    return RESULTS_DIR / f"scaffold_{FEATURE}_p{p:.2f}"


def ensure_scaffold(p: float) -> Path:
    out = penalty_dir(p)
    adv_lime = out / "adv" / "adv_lime_values.npy"
    adv_shap = out / "adv" / "adv_shap_values.npy"
    if adv_lime.exists() and adv_shap.exists():
        print(f"  [penalty={p:.2f}] cached at {out}")
        return out
    print(f"  [penalty={p:.2f}] regenerating scaffold → {out}")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "generate_scaffold_explanations.py"),
        "--bias-feature", FEATURE,
        "--n-explain", "200",
        "--penalty", f"{p}",
        "--out-dir", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return out


def load_pair(out: Path, explainer: str):
    honest = np.load(out / "honest" / f"{explainer}_values.npy")
    adv    = np.load(out / "adv"    / f"adv_{explainer}_values.npy")
    test_idx = np.load(out / "honest" / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1] - honest.shape[1]))])
    return honest, adv, test_idx


def fit_multiseed(A, layer, K, method, n_seeds=N_SEEDS):
    nmis = []
    for s in range(n_seeds):
        if method == "plain":
            m = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                        gamma=1.0, seed=s, verbose=False)
        else:
            m = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                            thin=5, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(m.z, layer))
    return float(np.mean(nmis)), float(np.std(nmis))


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "stress_penalty"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    results = {"K": K, "feature": FEATURE, "penalties": []}

    for p in PENALTIES:
        print(f"\n══ penalty = {p:.2f} ══")
        scaffold = ensure_scaffold(p)

        row = {"penalty": p, "cells": {}}
        for expl in EXPLAINERS:
            honest, adv, test_idx = load_pair(scaffold, expl)
            col = df[FEATURE].values[test_idx]
            layer = (col > THRESHOLD).astype(int)
            A_h = build_knn_similarity_graph(honest, k=K_NEIGHBOURS, symmetric=True)
            A_a = build_knn_similarity_graph(adv,    k=K_NEIGHBOURS, symmetric=True)

            ph_m, ph_s = fit_multiseed(A_h, layer, K, "plain")
            pa_m, pa_s = fit_multiseed(A_a, layer, K, "plain")
            eh_m, eh_s = fit_multiseed(A_h, layer, K, "pex")
            ea_m, ea_s = fit_multiseed(A_a, layer, K, "pex")

            print(f"  {expl.upper():4s}  honest: plain={ph_m:.3f}±{ph_s:.3f}  "
                  f"pEx={eh_m:.3f}±{eh_s:.3f}")
            print(f"  {expl.upper():4s}  adv:    plain={pa_m:.3f}±{pa_s:.3f}  "
                  f"pEx={ea_m:.3f}±{ea_s:.3f}  Δ={ea_m-pa_m:+.4f}")

            row["cells"][expl] = {
                "plain_honest_mean": ph_m, "plain_honest_std": ph_s,
                "plain_adv_mean":    pa_m, "plain_adv_std":    pa_s,
                "pex_honest_mean":   eh_m, "pex_honest_std":   eh_s,
                "pex_adv_mean":      ea_m, "pex_adv_std":      ea_s,
                "delta_adv":         ea_m - pa_m,
            }
        results["penalties"].append(row)

        # incremental save so partial progress isn't lost
        with open(out_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {out_dir/'summary.json'}")

    # Plot: adv NMI vs penalty, per explainer, plain vs pEx
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, expl in zip(axes, EXPLAINERS):
        ps  = [r["penalty"] for r in results["penalties"]]
        pm  = [r["cells"][expl]["plain_adv_mean"] for r in results["penalties"]]
        ps_ = [r["cells"][expl]["plain_adv_std"]  for r in results["penalties"]]
        em  = [r["cells"][expl]["pex_adv_mean"]   for r in results["penalties"]]
        es_ = [r["cells"][expl]["pex_adv_std"]    for r in results["penalties"]]
        hm  = [r["cells"][expl]["plain_honest_mean"] for r in results["penalties"]]
        ax.errorbar(ps, pm, yerr=ps_, marker="o", label="plain SBM (adv)",
                    color="#3182bd", capsize=3)
        ax.errorbar(ps, em, yerr=es_, marker="s", label="pEx-SBM (adv)",
                    color="#de2d26", capsize=3)
        ax.plot(ps, hm, marker=".", linestyle="--", color="#999999",
                label="plain SBM (honest baseline)")
        ax.set_xlabel("scaffold penalty")
        ax.set_title(f"{expl.upper()} — {FEATURE} / K={K}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("layer-separation NMI")
    plt.tight_layout()
    fig.savefig(out_dir / "penalty_plot.png", dpi=160)
    print(f"Saved {out_dir/'penalty_plot.png'}")


if __name__ == "__main__":
    main()
