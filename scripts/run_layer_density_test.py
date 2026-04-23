"""Fix 3 — Beta-Bernoulli layer-density sanity check.

Answers one prerequisite question per cell: does the k-NN similarity graph
have any layer-aligned edge structure at all, independent of any block
decomposition? If yes, SBMs have a lower-bound signal to work with. If
no, the null is not the SBM's fault.

Model (per cell):
    A_ij | ell_i, ell_j ~ Bernoulli(theta_{ell_i ell_j})
    theta ~ Beta(1, 1)

With binary layers, three parameters: theta_00, theta_11, theta_01.
Closed-form posteriors via Beta-Binomial conjugacy:
    theta_{ll'} | A ~ Beta(1 + M_{ll'}, 1 + T_{ll'} - M_{ll'})

Readouts per cell:
    * posterior means + 95% CIs for theta_00, theta_11, theta_01
    * within/across ratios theta_11/theta_01 and theta_00/theta_01
    * log Bayes factor for M1 (three separate thetas) vs M0 (single shared theta)

Comparison anchor:
    Alongside each BF we report the plain-SBM and pEx-SBM adv NMI for the
    same cell (from the main battery and penalty-sweep summaries). Fix 3
    is a pair-level test; the SBMs fit a full block decomposition on the
    same graph. Together they answer: "is there layer-aligned edge mass,
    and does any block-level method exploit it?"

Output:
    results/sbm_poc/layer_density/{summary.json, comparison.md}
"""

from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import build_knn_similarity_graph
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR


K_NEIGHBOURS = 8

# Main battery cells — match scripts/run_sbm_battery.py
CELLS = [
    dict(feature="pct_black_latino", scaffold="scaffold",              threshold=0.25,   moran=0.76),
    dict(feature="sqft_lot",         scaffold="scaffold_sqft_lot",     threshold=9361.0, moran=0.47),
    dict(feature="view",             scaffold="scaffold_view",         threshold=0.5,    moran=0.24),
    dict(feature="waterfront",       scaffold="scaffold_waterfront",   threshold=0.5,    moran=0.05),
]
EXPLAINERS = ["shap", "lime"]

# Penalty-sweep cells — same feature, varying scaffold penalty
PENALTY_GRID = [0.01, 0.04, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.20]


def _honest_dir(cell):
    if cell["feature"] == "pct_black_latino":
        return RESULTS_DIR / "explanations"
    return RESULTS_DIR / cell["scaffold"] / "honest"


def _adv_dir(cell):
    if cell["feature"] == "pct_black_latino":
        return RESULTS_DIR / "scaffold"
    return RESULTS_DIR / cell["scaffold"] / "adv"


def load_cell(cell, explainer):
    hd, ad = _honest_dir(cell), _adv_dir(cell)
    honest = np.load(hd / f"{explainer}_values.npy")
    adv    = np.load(ad / f"adv_{explainer}_values.npy")
    test_idx = np.load(hd / "test_indices.npy")
    if honest.shape[1] < adv.shape[1]:
        honest = np.hstack([honest, np.zeros((honest.shape[0], adv.shape[1]-honest.shape[1]))])
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[cell["feature"]].values[test_idx]
    layer = (col > cell["threshold"]).astype(int)
    return honest, adv, layer


def layer_density_test(A: np.ndarray, layers: np.ndarray) -> dict:
    """Beta-Bernoulli posterior on (low-low, high-high, across) edge rates
    plus log Bayes factor M1 (three thetas) vs M0 (one shared theta)."""
    n = len(layers)
    iu = np.triu_indices(n, k=1)
    edges = A[iu].astype(np.int64)
    pair_sum = layers[iu[0]].astype(np.int64) + layers[iu[1]].astype(np.int64)

    cells = {}
    log_m1 = 0.0
    tot_M, tot_T = 0, 0
    for k, name in [(0, "low_low"), (2, "high_high"), (1, "across")]:
        mask = pair_sum == k
        M = int(edges[mask].sum())
        T = int(mask.sum())
        a, b = 1 + M, 1 + T - M
        mean = a / (a + b)
        ci_lo = float(beta_dist.ppf(0.025, a, b)) if T > 0 else np.nan
        ci_hi = float(beta_dist.ppf(0.975, a, b)) if T > 0 else np.nan
        log_marg = betaln(a, b) - betaln(1, 1)
        log_m1 += log_marg
        tot_M += M
        tot_T += T
        cells[name] = dict(M=M, T=T, posterior_mean=mean,
                           ci_95=(ci_lo, ci_hi), log_marginal=log_marg)

    # Null: one shared theta across all pair types
    a0, b0 = 1 + tot_M, 1 + tot_T - tot_M
    log_m0 = betaln(a0, b0) - betaln(1, 1)
    log_bf = log_m1 - log_m0

    th11 = cells["high_high"]["posterior_mean"]
    th00 = cells["low_low"]["posterior_mean"]
    th01 = cells["across"]["posterior_mean"]

    return dict(
        cells=cells,
        log_bayes_factor=log_bf,
        theta_high_over_across=(th11 / th01) if th01 > 0 else np.nan,
        theta_low_over_across=(th00 / th01) if th01 > 0 else np.nan,
        n=int(n),
        n_high=int(layers.sum()),
        n_low=int((layers == 0).sum()),
        pair_T_hh=cells["high_high"]["T"],
        pair_T_ll=cells["low_low"]["T"],
        pair_T_ax=cells["across"]["T"],
    )


def _load_sbm_nmi_lookup():
    """Pull plain/pEx adv NMI for the main cells from the battery summary,
    plus penalty-sweep adv NMI from stress_penalty/summary.json."""
    battery = {}
    bp = RESULTS_DIR / "sbm_poc" / "battery" / "summary.json"
    if bp.exists():
        d = json.load(open(bp))
        for c in d.get("cells", []):
            battery[(c["feature"], c["explainer"])] = dict(
                plain_adv=c["plain"]["adv_nmi_mean"],
                pex_adv=c["pex"]["adv_nmi_mean"],
                plain_honest=c["plain"]["honest_nmi_mean"],
                pex_honest=c["pex"]["honest_nmi_mean"],
            )
    penalty = {}
    pp = RESULTS_DIR / "sbm_poc" / "stress_penalty" / "summary.json"
    if pp.exists():
        d = json.load(open(pp))
        for r in d.get("penalties", []):
            for e, c in r["cells"].items():
                penalty[(round(r["penalty"], 3), e)] = dict(
                    plain_adv=c["plain_adv_mean"],
                    pex_adv=c["pex_adv_mean"],
                    plain_honest=c["plain_honest_mean"],
                )
    return battery, penalty


def main():
    out_dir = RESULTS_DIR / "sbm_poc" / "layer_density"
    out_dir.mkdir(parents=True, exist_ok=True)

    battery_nmi, penalty_nmi = _load_sbm_nmi_lookup()
    results = {"main_battery": [], "penalty_sweep": []}

    # ── Main 8 cells (honest + adv graphs) ──────────────────────────────
    print("══ MAIN BATTERY (8 cells × {honest, adv}) ══")
    for cell in CELLS:
        for expl in EXPLAINERS:
            try:
                honest, adv, layer = load_cell(cell, expl)
            except FileNotFoundError as e:
                print(f"  skip {cell['feature']}/{expl}: {e}")
                continue
            if layer.sum() < 2 or (layer == 0).sum() < 2:
                print(f"  skip {cell['feature']}/{expl}: degenerate layer")
                continue

            A_h = build_knn_similarity_graph(honest, k=K_NEIGHBOURS, symmetric=True)
            A_a = build_knn_similarity_graph(adv,    k=K_NEIGHBOURS, symmetric=True)
            fix3_h = layer_density_test(A_h, layer)
            fix3_a = layer_density_test(A_a, layer)

            nmi = battery_nmi.get((cell["feature"], expl), {})

            print(f"\n  {cell['feature']:18s} / {expl.upper():4s}  "
                  f"I={cell['moran']:.2f}  n_high={fix3_h['n_high']}")
            print(f"    HONEST  logBF={fix3_h['log_bayes_factor']:+7.2f}  "
                  f"θ_11={fix3_h['cells']['high_high']['posterior_mean']:.3f}  "
                  f"θ_01={fix3_h['cells']['across']['posterior_mean']:.3f}  "
                  f"ratio={fix3_h['theta_high_over_across']:.2f}"
                  + (f"  plain NMI={nmi.get('plain_honest'):.3f}" if nmi else ""))
            print(f"    ADV     logBF={fix3_a['log_bayes_factor']:+7.2f}  "
                  f"θ_11={fix3_a['cells']['high_high']['posterior_mean']:.3f}  "
                  f"θ_01={fix3_a['cells']['across']['posterior_mean']:.3f}  "
                  f"ratio={fix3_a['theta_high_over_across']:.2f}"
                  + (f"  plain NMI={nmi.get('plain_adv'):.3f}  pEx NMI={nmi.get('pex_adv'):.3f}" if nmi else ""))

            results["main_battery"].append(dict(
                feature=cell["feature"], explainer=expl,
                moran_I=cell["moran"], threshold=cell["threshold"],
                honest=fix3_h, adv=fix3_a,
                sbm_nmi=nmi,
            ))

    # ── Penalty sweep (sqft_lot only, adv graphs) ───────────────────────
    print("\n══ PENALTY SWEEP (sqft_lot × {SHAP, LIME}, adv only) ══")
    df_sqft = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    for p in PENALTY_GRID:
        base = RESULTS_DIR / f"scaffold_sqft_lot_p{p:.2f}"
        if not base.exists():
            print(f"  skip p={p:.2f}: {base} missing")
            continue
        for expl in EXPLAINERS:
            adv_path = base / "adv" / f"adv_{expl}_values.npy"
            idx_path = base / "honest" / "test_indices.npy"
            if not adv_path.exists() or not idx_path.exists():
                continue
            adv = np.load(adv_path)
            test_idx = np.load(idx_path)
            col = df_sqft["sqft_lot"].values[test_idx]
            layer = (col > 9361.0).astype(int)
            A_a = build_knn_similarity_graph(adv, k=K_NEIGHBOURS, symmetric=True)
            fix3 = layer_density_test(A_a, layer)
            nmi = penalty_nmi.get((round(p, 3), expl), {})
            print(f"  p={p:.2f}  {expl.upper():4s}  logBF={fix3['log_bayes_factor']:+8.2f}  "
                  f"θ_11={fix3['cells']['high_high']['posterior_mean']:.3f}  "
                  f"θ_01={fix3['cells']['across']['posterior_mean']:.3f}  "
                  f"ratio={fix3['theta_high_over_across']:.2f}"
                  + (f"  plain NMI={nmi.get('plain_adv'):.3f}" if nmi else ""))
            results["penalty_sweep"].append(dict(
                penalty=p, explainer=expl, fix3=fix3, sbm_nmi=nmi,
            ))

    # ── Save ────────────────────────────────────────────────────────────
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out_dir/'summary.json'}")

    # ── Comparison markdown table ───────────────────────────────────────
    md = ["# Fix 3: Layer-density Bayes factor vs SBM NMI\n"]
    md.append("## Main battery (adversarial graphs)\n")
    md.append("| Feature | Explainer | I | n_high | log BF (adv) | θ_11 | θ_01 | ratio | plain NMI | pEx NMI |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results["main_battery"]:
        f3 = r["adv"]; nmi = r["sbm_nmi"]
        md.append(f"| {r['feature']} | {r['explainer'].upper()} | {r['moran_I']:.2f} | "
                  f"{f3['n_high']} | {f3['log_bayes_factor']:+.2f} | "
                  f"{f3['cells']['high_high']['posterior_mean']:.3f} | "
                  f"{f3['cells']['across']['posterior_mean']:.3f} | "
                  f"{f3['theta_high_over_across']:.2f} | "
                  f"{nmi.get('plain_adv', float('nan')):.3f} | "
                  f"{nmi.get('pex_adv', float('nan')):.3f} |")
    md.append("\n## Main battery (honest graphs — baseline)\n")
    md.append("| Feature | Explainer | log BF (honest) | θ_11 | θ_01 | ratio | plain NMI | pEx NMI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in results["main_battery"]:
        f3 = r["honest"]; nmi = r["sbm_nmi"]
        md.append(f"| {r['feature']} | {r['explainer'].upper()} | {f3['log_bayes_factor']:+.2f} | "
                  f"{f3['cells']['high_high']['posterior_mean']:.3f} | "
                  f"{f3['cells']['across']['posterior_mean']:.3f} | "
                  f"{f3['theta_high_over_across']:.2f} | "
                  f"{nmi.get('plain_honest', float('nan')):.3f} | "
                  f"{nmi.get('pex_honest', float('nan')):.3f} |")
    md.append("\n## Penalty sweep (sqft_lot, adversarial)\n")
    md.append("| p | Explainer | log BF | θ_11 | θ_01 | ratio | plain NMI |")
    md.append("|---|---|---|---|---|---|---|")
    for r in results["penalty_sweep"]:
        f3 = r["fix3"]; nmi = r["sbm_nmi"]
        md.append(f"| {r['penalty']:.2f} | {r['explainer'].upper()} | {f3['log_bayes_factor']:+.2f} | "
                  f"{f3['cells']['high_high']['posterior_mean']:.3f} | "
                  f"{f3['cells']['across']['posterior_mean']:.3f} | "
                  f"{f3['theta_high_over_across']:.2f} | "
                  f"{nmi.get('plain_adv', float('nan')):.3f} |")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")
    print(f"Saved {out_dir/'comparison.md'}")


if __name__ == "__main__":
    main()
