"""Plot tanh-probe sweep: plain vs pEx NMI across effective Lipschitz."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sbm_explainer" / "tanh_sweep"
OUT = ROOT / "docs" / "sbm_explainer" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def ci95(xs):
    n = len(xs)
    if n < 2:
        return xs.mean(), xs.mean(), xs.mean()
    m = xs.mean()
    se = xs.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h


def main():
    data = json.loads((RES / "results.json").read_text())
    betas = np.array(data["betas"])

    plain = [np.array([s["plain"] for s in c["per_seed"]]) for c in data["cells"]]
    pex = [np.array([s["pex"] for s in c["per_seed"]]) for c in data["cells"]]
    delta = [np.array([s["delta"] for s in c["per_seed"]]) for c in data["cells"]]

    p_mean = np.array([x.mean() for x in plain])
    p_ci = np.array([ci95(x) for x in plain])
    e_mean = np.array([x.mean() for x in pex])
    e_ci = np.array([ci95(x) for x in pex])
    d_mean = np.array([x.mean() for x in delta])
    d_ci = np.array([ci95(x) for x in delta])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.errorbar(betas, p_mean,
                yerr=[p_mean - p_ci[:, 1], p_ci[:, 2] - p_mean],
                fmt="s-", color="tab:red", capsize=3, label="plain SBM")
    ax.errorbar(betas, e_mean,
                yerr=[e_mean - e_ci[:, 1], e_ci[:, 2] - e_mean],
                fmt="o-", color="tab:blue", capsize=3, label="pEx SBM")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$  (effective Lipschitz $= \beta/4$)")
    ax.set_ylabel("Layer NMI")
    ax.set_title("Recovery vs innocuous Lipschitz")
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    for i, (x, seeds) in enumerate(zip(betas, delta)):
        ax.scatter([x] * len(seeds), seeds, alpha=0.45, color="steelblue",
                   s=30, zorder=2, label="per-seed" if i == 0 else None)
    ax.errorbar(betas, d_mean,
                yerr=[d_mean - d_ci[:, 1], d_ci[:, 2] - d_mean],
                fmt="o-", color="black", lw=1.4, capsize=4, zorder=3,
                label="mean ± 95% CI")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Delta = \mathrm{NMI}_{\mathrm{pEx}} - \mathrm{NMI}_{\mathrm{plain}}$")
    ax.set_title("pEx gain grows with Lipschitz smoothness")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Tanh probe: Bernoulli(sigmoid(β·x_u)) innocuous, CC/LIME σ=1.0, K=20\n"
        "Clean monotone Δ(β): saturated-plain at low β, plain fails + pEx holds at high β",
        fontsize=10)
    fig.tight_layout()
    path = OUT / "tanh_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")

    print("\n── tanh sweep table ──")
    print(f"{'β':>6}  {'plain':>6}  {'pEx':>6}  {'Δ mean':>8}  {'Δ 95% CI':>20}")
    for b, p, e, dm, dc in zip(betas, p_mean, e_mean, d_mean, d_ci):
        print(f"{b:6.2f}  {p:6.3f}  {e:6.3f}  {dm:+8.3f}  [{dc[1]:+6.3f}, {dc[2]:+6.3f}]")


if __name__ == "__main__":
    main()
