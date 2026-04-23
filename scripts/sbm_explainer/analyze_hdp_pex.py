"""Score H-DP pEx output: NMI vs layer + C ratio, compare to plain + finite-K pEx."""

from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sbm.compare import layer_coclustering
from src.utils.paths import RESULTS_DIR

in_dir = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
z_hats = np.loadtxt(in_dir / "z_hats.csv", delimiter=",").astype(int)  # (n, n_seeds)
layer_re = np.load(in_dir / "layer_reordered.npy")
order = np.load(in_dir / "order.npy")

n, n_seeds = z_hats.shape
# Layer in the reordered frame
nmis = [nmi(layer_re, z_hats[:, s]) for s in range(n_seeds)]
print(f"H-DP pEx: NMI per seed = {[f'{v:.3f}' for v in nmis]}")
print(f"          NMI mean = {np.mean(nmis):.3f} ± {np.std(nmis):.3f}")

# Reference numbers from fingerprint_sbm for the same Gauss+LIME cell
fp = json.load(open(RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm" / "summary.json"))
gauss_lime = next(c for c in fp["cells"]
                  if c["variant"] == "gaussian" and c["scaffold"] == "lime")
plain_nmi = gauss_lime["plain"]["nmi_mean"]; plain_std = gauss_lime["plain"]["nmi_std"]
pexK_nmi  = gauss_lime["pex"]["nmi_mean"];   pexK_std  = gauss_lime["pex"]["nmi_std"]
plain_ratio = gauss_lime["plain"]["ratio"]

# Compute C ratio for H-DP by averaging z_hat indicator matrices (not ideal
# since z_hat is VI-optimal point estimate, not posterior C — we'd need to
# average across posterior samples per seed). Use indicator-avg as a proxy.
C_avg = np.zeros((n, n), dtype=np.float64)
for s in range(n_seeds):
    z = z_hats[:, s]
    C_avg += (z[:, None] == z[None, :]).astype(np.float64)
C_avg /= n_seeds
within, across = layer_coclustering(C_avg, layer_re)
hdp_ratio = within / across if across > 0 else float("nan")

print(f"\n── Comparison on Gauss+LIME fingerprint graph ──")
print(f"{'method':<25s} {'NMI':>18s} {'Δ vs plain':>12s} {'C ratio':>10s}")
print(f"{'plain SBM (K=4)':<25s} {plain_nmi:.3f} ± {plain_std:.3f}{'':>3s} {'baseline':>12s} {plain_ratio:>10.2f}")
print(f"{'finite-K pEx (K=4)':<25s} {pexK_nmi:.3f} ± {pexK_std:.3f}{'':>3s} {pexK_nmi - plain_nmi:+.3f}      {'(same)':>10s}")
print(f"{'H-DP pEx (K inferred)':<25s} {np.mean(nmis):.3f} ± {np.std(nmis):.3f}{'':>3s} {np.mean(nmis) - plain_nmi:+.3f}      {hdp_ratio:>10.2f}")

# Save
out = dict(
    hdp_nmi_per_seed=[float(v) for v in nmis],
    hdp_nmi_mean=float(np.mean(nmis)), hdp_nmi_std=float(np.std(nmis)),
    hdp_C_ratio_proxy=float(hdp_ratio),
    plain=dict(nmi_mean=plain_nmi, nmi_std=plain_std, ratio=plain_ratio),
    finite_K_pex=dict(nmi_mean=pexK_nmi, nmi_std=pexK_std),
    delta_hdp_minus_plain=float(np.mean(nmis) - plain_nmi),
    delta_hdp_minus_finiteK=float(np.mean(nmis) - pexK_nmi),
    n_clusters_inferred=[int(len(np.unique(z_hats[:, s]))) for s in range(n_seeds)],
)
with open(in_dir / "results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {in_dir/'results.json'}")
