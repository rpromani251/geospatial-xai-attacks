"""Plot tanh seed-scaled sweep (10 seeds at knee, 5 at saturation)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sbm_explainer" / "tanh_seedscale" / "results.json"
OUT = ROOT / "docs" / "sbm_explainer" / "figures"


def ci95(xs):
    n = len(xs)
    if n < 2:
        return xs.mean(), xs.mean(), xs.mean()
    m = xs.mean()
    se = xs.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h


def main():
    data = json.loads(RES.read_text())
    betas = np.array([c["beta"] for c in data["cells"]])
    plain_seeds = [np.array([s["plain"] for s in c["per_seed"]]) for c in data["cells"]]
    pex_seeds = [np.array([s["pex"] for s in c["per_seed"]]) for c in data["cells"]]
    delta_seeds = [np.array([s["delta"] for s in c["per_seed"]]) for c in data["cells"]]
    n_per = [c["n_fp"] for c in data["cells"]]

    p_ci = np.array([ci95(x) for x in plain_seeds])
    e_ci = np.array([ci95(x) for x in pex_seeds])
    d_ci = np.array([ci95(x) for x in delta_seeds])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(betas, p_ci[:, 0],
                yerr=[p_ci[:, 0] - p_ci[:, 1], p_ci[:, 2] - p_ci[:, 0]],
                fmt="s-", color="tab:red", capsize=4, lw=1.6, label="plain SBM")
    ax.errorbar(betas, e_ci[:, 0],
                yerr=[e_ci[:, 0] - e_ci[:, 1], e_ci[:, 2] - e_ci[:, 0]],
                fmt="o-", color="tab:blue", capsize=4, lw=1.6, label="pEx SBM")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Layer NMI")
    ax.set_title("Plain collapses under anchor dilution\npEx holds until the step-function limit")
    ax.legend(loc="lower left", fontsize=10)
    for b, n in zip(betas, n_per):
        ax.annotate(f"n={n}", (b, 0.04), xytext=(0, -3),
                    textcoords="offset points", fontsize=7, ha="center", color="dimgray")

    ax = axes[1]
    for i, (x, seeds) in enumerate(zip(betas, delta_seeds)):
        ax.scatter([x] * len(seeds), seeds, alpha=0.4, color="steelblue",
                   s=28, zorder=2, label="per-seed" if i == 0 else None)
    ax.errorbar(betas, d_ci[:, 0],
                yerr=[d_ci[:, 0] - d_ci[:, 1], d_ci[:, 2] - d_ci[:, 0]],
                fmt="o-", color="black", lw=1.4, capsize=5, zorder=3,
                label="mean ± 95% CI")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Delta = \mathrm{NMI}_{\mathrm{pEx}} - \mathrm{NMI}_{\mathrm{plain}}$")
    ax.set_title("Δ: zero at β≤2, positive knee at β=5,\nplateau at β∈[10,50]")
    ax.legend(loc="upper left", fontsize=10)

    fig.suptitle(
        "Tanh-Bernoulli sweep, 10+5 seeds — CC/LIME σ=1.0, K=20",
        fontsize=10)
    fig.tight_layout()
    path = OUT / "tanh_seedscale.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")

    print("\n── seed-scaled table ──")
    print(f"{'β':>6}  {'n':>3}  {'plain':>15}  {'pEx':>15}  {'Δ mean':>8}  {'Δ 95% CI':>18}")
    for b, n, p, e, d in zip(betas, n_per, p_ci, e_ci, d_ci):
        print(f"{b:6.1f}  {n:3d}  {p[0]:.3f} [{p[1]:.3f},{p[2]:.3f}]  "
              f"{e[0]:.3f} [{e[1]:.3f},{e[2]:.3f}]  "
              f"{d[0]:+8.3f}  [{d[1]:+6.3f}, {d[2]:+6.3f}]")


if __name__ == "__main__":
    main()
