"""Regime (a) vs (b) diagnostic: is the SBM *posterior* layer-aware in the
transition region, or is the signal missing from the posterior itself?

Context
-------
Fix 3 (Beta-Bernoulli layer-density test) shows the k-NN graph carries
monotone layer-aligned edge evidence across the penalty sweep — log BF
rises from −7 at p=0.01 to +390 at p=0.19, passing through +50, +130,
+259 at intermediate p. Plain-SBM MAP-partition NMI stays at noise
(0.010–0.035) until p=0.19, then jumps to 0.398.

Two possible causes:

  (a) MAP collapse. Posterior P(z | A) has mass on layer-aligned
      configurations, but the modal / argmax readout picks configurations
      that don't line up with the layer labels. The co-clustering
      matrix C_{ij} = P(z_i = z_j | A) would still show within-layer
      separation in the transition region.

  (b) Likelihood blindness. The posterior itself doesn't encode layer,
      because no discrete K-block partition concentrates the pair-level
      signal. Within-layer vs across-layer C are indistinguishable.

Diagnostic
----------
For each penalty p ∈ {0.01, …, 0.20} and each explainer:
  1. Fit plain SBM (5 seeds, K=4) on the adversarial 8-NN graph.
  2. Average the posterior co-clustering matrix across seeds and draws.
  3. Report layer-separation NMI (MAP readout) vs mean C[within-layer]
     and mean C[across-layer] (posterior readout).

If C[within] >> C[across] in cells where NMI is noise, the MAP is
collapsing a signal the posterior carries — regime (a), readout-fix
path is viable.

If C[within] ≈ C[across] in those cells, the posterior doesn't encode
layer — regime (b), model itself needs revision (layer-conditional
likelihood, degree correction, weighted edges).

Output: results/sbm_poc/coclustering_diagnostic/{summary.json, table.md}
"""

from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import build_knn_similarity_graph, fit_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


FEATURE = "sqft_lot"
THRESHOLD = 9361.0
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
EXPLAINERS = ["lime", "shap"]
PENALTIES = [0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20]


def load_adv(p: float, explainer: str):
    base = RESULTS_DIR / f"scaffold_{FEATURE}_p{p:.2f}"
    adv = np.load(base / "adv" / f"adv_{explainer}_values.npy")
    test_idx = np.load(base / "honest" / "test_indices.npy")
    return adv, test_idx


def fit_and_average_C(A, layer, n_seeds=N_SEEDS):
    """Fit plain SBM n_seeds times, average the posterior co-clustering
    matrix across all seeds, and return (mean NMI, within_C, across_C)."""
    C_sum = np.zeros_like(A, dtype=float)
    nmis = []
    for s in range(n_seeds):
        sbm = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                      gamma=1.0, seed=s, verbose=False)
        C_sum += sbm.co_clustering_matrix()
        nmis.append(layer_separation(sbm.z, layer))
    C = C_sum / n_seeds
    within, across = layer_coclustering(C, layer)
    return float(np.mean(nmis)), float(np.std(nmis)), within, across


def load_bf(p: float, expl: str):
    """Pull Fix-3 log BF for comparison."""
    path = RESULTS_DIR / "sbm_poc" / "layer_density" / "summary.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    for r in d.get("penalty_sweep", []):
        if round(r["penalty"], 3) == round(p, 3) and r["explainer"] == expl:
            return r["fix3"]["log_bayes_factor"]
    return None


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "coclustering_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    rows = []

    print(f"{'p':>5} {'expl':>4} {'NMI':>14} {'C_within':>10} {'C_across':>10} "
          f"{'gap':>8} {'ratio':>6} {'logBF':>8}")
    print("─" * 78)
    for p in PENALTIES:
        for expl in EXPLAINERS:
            adv, idx = load_adv(p, expl)
            col = df[FEATURE].values[idx]
            layer = (col > THRESHOLD).astype(int)
            A = build_knn_similarity_graph(adv, k=K_NEIGHBOURS, symmetric=True)
            nmi_m, nmi_s, within, across = fit_and_average_C(A, layer)
            gap = within - across
            ratio = within / across if across > 0 else float("nan")
            logbf = load_bf(p, expl)
            rows.append(dict(
                penalty=p, explainer=expl,
                nmi_mean=nmi_m, nmi_std=nmi_s,
                C_within=within, C_across=across,
                gap=gap, ratio=ratio,
                log_bayes_factor=logbf,
            ))
            print(f"{p:>5.2f} {expl:>4} {nmi_m:>6.3f}±{nmi_s:.3f} {within:>10.4f} "
                  f"{across:>10.4f} {gap:>+8.4f} {ratio:>6.2f} "
                  f"{(f'{logbf:+.1f}' if logbf is not None else '—'):>8}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"K": K, "rows": rows}, f, indent=2, default=float)
    print(f"\nSaved {out_dir/'summary.json'}")

    # Markdown comparison
    md = ["# Regime (a) vs (b): posterior co-clustering gap across penalty sweep\n",
          "Plain SBM, K=4, 5 seeds, averaged posterior co-clustering matrix.\n",
          "| p | Expl. | MAP NMI | C within-layer | C across-layer | gap | ratio | Fix-3 log BF |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bf = f"{r['log_bayes_factor']:+.1f}" if r['log_bayes_factor'] is not None else "—"
        md.append(f"| {r['penalty']:.2f} | {r['explainer'].upper()} | "
                  f"{r['nmi_mean']:.3f} ± {r['nmi_std']:.3f} | "
                  f"{r['C_within']:.4f} | {r['C_across']:.4f} | "
                  f"{r['gap']:+.4f} | {r['ratio']:.2f} | {bf} |")
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    print(f"Saved {out_dir/'table.md'}")


if __name__ == "__main__":
    main()
