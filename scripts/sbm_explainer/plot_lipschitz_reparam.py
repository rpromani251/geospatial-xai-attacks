"""Reparameterize Lipschitz sweep results on the physically meaningful axis.

X-axis: E[n_flips] = beta * sigma * sqrt(2/pi) / pi  (expected sign-flips of
sign(sin(beta*x)) over a Gaussian perturbation of scale sigma).

Shows per-seed scatter + mean + 95% CI so the seed noise is visible in the
figure itself. Supports the claim "Delta > 0 across the full decorrelation
range; 3-seed noise prevents resolving functional form."
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sbm_explainer" / "lipschitz_sweep"
OUT = ROOT / "docs" / "sbm_explainer" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def n_flips_expected(beta, sigma):
    return beta * sigma * np.sqrt(2.0 / np.pi) / np.pi


def p_at_least_one(beta, sigma):
    lam = n_flips_expected(beta, sigma)
    return 1.0 - np.exp(-lam)


def main():
    data = json.loads((RES / "results.json").read_text())
    sigma = data["sigma"]
    betas = np.array(data["betas"])
    n_flips = n_flips_expected(betas, sigma)
    p_ge1 = p_at_least_one(betas, sigma)

    mean_delta = np.array([c["delta"] for c in data["cells"]])
    per_seed = [np.array([s["delta"] for s in c["per_seed"]]) for c in data["cells"]]
    per_seed_plain = [np.array([s["plain"] for s in c["per_seed"]]) for c in data["cells"]]
    per_seed_pex = [np.array([s["pex"] for s in c["per_seed"]]) for c in data["cells"]]

    # 95% CI via t-distribution (n=3 seeds)
    def ci95(xs):
        n = len(xs)
        m = xs.mean()
        se = xs.std(ddof=1) / np.sqrt(n)
        h = se * stats.t.ppf(0.975, n - 1)
        return m, m - h, m + h

    ci = np.array([ci95(xs) for xs in per_seed])
    means, lo, hi = ci[:, 0], ci[:, 1], ci[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Delta vs E[n_flips] (log x)
    ax = axes[0]
    for i, (x, seeds) in enumerate(zip(n_flips, per_seed)):
        ax.scatter([x] * len(seeds), seeds, alpha=0.45, color="steelblue",
                   s=30, zorder=2, label="per-seed" if i == 0 else None)
    ax.errorbar(n_flips, means, yerr=[means - lo, hi - means],
                fmt="o-", color="black", lw=1.4, capsize=4, zorder=3,
                label="mean ± 95% CI")
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathbb{E}[n_{\mathrm{flips}}] = \beta\sigma\sqrt{2/\pi}/\pi$")
    ax.set_ylabel(r"$\Delta = \mathrm{NMI}_{\mathrm{pEx}} - \mathrm{NMI}_{\mathrm{plain}}$")
    ax.set_title("pEx gain vs expected sign-flip density")
    ax.legend(loc="lower right", fontsize=9)
    for i, b in enumerate(betas):
        ax.annotate(f"β={b:g}", (n_flips[i], means[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7, color="dimgray")

    # Panel 2: plain vs pEx vs decorrelation probability (linear)
    ax = axes[1]
    plain_means = np.array([xs.mean() for xs in per_seed_plain])
    pex_means = np.array([xs.mean() for xs in per_seed_pex])
    plain_ci = np.array([ci95(xs) for xs in per_seed_plain])
    pex_ci = np.array([ci95(xs) for xs in per_seed_pex])

    ax.errorbar(p_ge1, plain_means,
                yerr=[plain_means - plain_ci[:, 1], plain_ci[:, 2] - plain_means],
                fmt="s-", color="tab:red", capsize=3, label="plain SBM")
    ax.errorbar(p_ge1, pex_means,
                yerr=[pex_means - pex_ci[:, 1], pex_ci[:, 2] - pex_means],
                fmt="o-", color="tab:blue", capsize=3, label="pEx SBM")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(r"$P(\geq 1\ \mathrm{flip\ in\ perturbation}) = 1 - e^{-\lambda}$")
    ax.set_ylabel("Layer NMI")
    ax.set_title("Recovery across decorrelation regime")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-0.05, max(pex_ci[:, 2].max(), 0.7) + 0.05)

    fig.suptitle(
        "Lipschitz sweep reparameterized: sinusoidal-sign innocuous, σ=1.0, K=20\n"
        "Δ > 0 across full range; 3-seed CIs overlap, so β-axis trend is unresolved",
        fontsize=10)
    fig.tight_layout()
    path = OUT / "lipschitz_reparam.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")

    print("\n── reparam table ──")
    print(f"{'β':>6}  {'E[n_flips]':>10}  {'P(≥1)':>7}  "
          f"{'Δ mean':>8}  {'Δ 95% CI':>18}")
    for b, nf, p, m, l, h in zip(betas, n_flips, p_ge1, means, lo, hi):
        print(f"{b:6.2f}  {nf:10.4f}  {p:7.3f}  {m:+8.3f}  [{l:+6.3f}, {h:+6.3f}]")


if __name__ == "__main__":
    main()
