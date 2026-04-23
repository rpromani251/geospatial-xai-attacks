"""Plot K-scaling sweep: plain/pEx NMI and Δ across K for all 4 datasets.

Emphasizes the CC/XOR and German cells as the clearest mechanism
signature (plain monotonically collapses with K while pEx holds).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sbm_explainer" / "k_scaling" / "results.json"
OUT = ROOT / "docs" / "sbm_explainer" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "housing_lime": "Housing (LIME σ=0.1·std)",
    "compas_shap": "COMPAS (SHAP σ=0.1)",
    "german_shap": "German (SHAP σ=1.0)",
    "cc_psi2_lime": "CC psi_two (LIME σ=1.0)",
}


def main():
    data = json.loads(RES.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"housing_lime": "tab:green", "compas_shap": "tab:orange",
              "german_shap": "tab:purple", "cc_psi2_lime": "tab:blue"}

    # Left: plain vs pEx NMI trajectories
    ax = axes[0]
    for key, label in LABELS.items():
        if key not in data:
            continue
        cells = data[key]
        Ks = sorted(int(k) for k in cells.keys())
        plain = [cells[str(K)]["plain_mean"] for K in Ks]
        pex = [cells[str(K)]["pex_mean"] for K in Ks]
        plain_std = [cells[str(K)].get("plain_std", 0) for K in Ks]
        pex_std = [cells[str(K)].get("pex_std", 0) for K in Ks]
        c = colors[key]
        ax.errorbar(Ks, plain, yerr=plain_std, fmt="s--", color=c,
                    alpha=0.6, capsize=3, label=f"{label} plain")
        ax.errorbar(Ks, pex, yerr=pex_std, fmt="o-", color=c,
                    capsize=3, label=f"{label} pEx")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("K (fingerprint dimension)")
    ax.set_ylabel("Layer NMI")
    ax.set_title("Plain SBM collapses with K; pEx holds")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # Right: Δ trajectories
    ax = axes[1]
    for key, label in LABELS.items():
        if key not in data:
            continue
        cells = data[key]
        Ks = sorted(int(k) for k in cells.keys())
        delta = [cells[str(K)]["delta_mean"] for K in Ks]
        delta_std = [cells[str(K)].get("delta_std", 0) for K in Ks]
        ax.errorbar(Ks, delta, yerr=delta_std, fmt="o-",
                    color=colors[key], capsize=3, label=label)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\Delta = \mathrm{NMI}_{\mathrm{pEx}} - \mathrm{NMI}_{\mathrm{plain}}$")
    ax.set_title("pEx gain grows with anchor dilution")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "K-sweep across 4 datasets: anchor-dominance erosion\n"
        "CC psi_two (XOR) gives the sharpest signature: plain 0.333→0.020, pEx 0.33→0.43",
        fontsize=10)
    fig.tight_layout()
    path = OUT / "k_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
