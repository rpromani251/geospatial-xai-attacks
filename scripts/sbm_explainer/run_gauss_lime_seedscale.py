"""Probe 2: 20-seed run of plain SBM + finite-K pEx on the Gauss+LIME graph.

Reuses the saved graph from run_fingerprint_sbm.py; no need to rebuild
scaffolds or fingerprints. Reports mean, std, and 95% CI for NMI and
Δ(pEx − plain).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import RESULTS_DIR

N_SEEDS = 20
K = 4
N_ITER = 400
BURN_IN = 200

fp_dir = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"
hdp_dir = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
out_dir = RESULTS_DIR / "sbm_explainer" / "seedscale_gauss_lime"
out_dir.mkdir(parents=True, exist_ok=True)

A = np.load(fp_dir / "graph_gaussian_lime.npy")
# layer.csv in hdp_pex/ is REORDERED for the R sampler; the saved graph is
# in original order. Reconstruct the unreordered layer by inverting order.
layer_re = np.load(hdp_dir / "layer_reordered.npy")  # reordered, 0/1
order = np.load(hdp_dir / "order.npy")               # reordered → original
inv = np.argsort(order)
layer = layer_re[inv]                                 # back in original order
assert A.shape == (200, 200)
assert layer.shape == (200,) and int(layer.sum()) == 50, \
    f"layer sum {int(layer.sum())} != 50"

print(f"n={len(layer)} edges={A.sum()//2} high={layer.sum()} low={(layer==0).sum()}")
print(f"N_SEEDS={N_SEEDS}, K={K}, iter={N_ITER} (burn {BURN_IN})")

plain_nmis = []
pex_nmis = []
plain_ratios = []
t0 = time.time()

for s in range(N_SEEDS):
    mp = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                 gamma=1.0, seed=s, verbose=False)
    me = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                     thin=5, gamma=1.0, seed=s, verbose=False)
    pn = layer_separation(mp.z, layer)
    en = layer_separation(me.z, layer)
    Cp = mp.co_clustering_matrix()
    within, across = layer_coclustering(Cp, layer)
    ratio = within / across if across > 0 else float("nan")
    plain_nmis.append(pn); pex_nmis.append(en); plain_ratios.append(ratio)
    dt = time.time() - t0
    print(f"  seed {s:2d}: plain={pn:.3f} pex={en:.3f} Δ={en-pn:+.3f} "
          f"ratio={ratio:.2f}  (elapsed {dt:.0f}s)")

plain = np.asarray(plain_nmis)
pex = np.asarray(pex_nmis)
delta = pex - plain
ratios = np.asarray(plain_ratios)


def summary(a: np.ndarray) -> dict:
    return dict(
        mean=float(a.mean()), std=float(a.std(ddof=1)),
        sem=float(a.std(ddof=1) / np.sqrt(len(a))),
        ci95=[float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))],
        min=float(a.min()), max=float(a.max()),
    )


out = dict(
    n_seeds=N_SEEDS, K=K, n_iter=N_ITER, burn_in=BURN_IN,
    plain=summary(plain), pex=summary(pex), delta=summary(delta),
    C_ratio=summary(ratios),
    plain_per_seed=[float(v) for v in plain],
    pex_per_seed=[float(v) for v in pex],
    ratio_per_seed=[float(v) for v in ratios],
)

with open(out_dir / "results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n── Summary ──")
print(f"plain NMI  : {out['plain']['mean']:.4f} ± {out['plain']['std']:.4f}  "
      f"(SEM {out['plain']['sem']:.4f})  95% CI "
      f"[{out['plain']['ci95'][0]:.3f}, {out['plain']['ci95'][1]:.3f}]")
print(f"pEx NMI    : {out['pex']['mean']:.4f} ± {out['pex']['std']:.4f}  "
      f"(SEM {out['pex']['sem']:.4f})  95% CI "
      f"[{out['pex']['ci95'][0]:.3f}, {out['pex']['ci95'][1]:.3f}]")
print(f"Δ (pex-pln): {out['delta']['mean']:.4f} ± {out['delta']['std']:.4f}  "
      f"(SEM {out['delta']['sem']:.4f})  95% CI "
      f"[{out['delta']['ci95'][0]:+.3f}, {out['delta']['ci95'][1]:+.3f}]")
# One-sample t vs 0 for Δ
from scipy import stats
t_stat, p = stats.ttest_1samp(delta, 0.0)
print(f"Δ > 0      : t={t_stat:.2f}, p={p:.2e}  "
      f"(vs null Δ=0, two-sided)")
print(f"\nSaved {out_dir / 'results.json'}")
