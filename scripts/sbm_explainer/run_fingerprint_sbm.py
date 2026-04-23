"""pEx-SBM-as-explainer — behavioral-fingerprint encoder under the scaffolded model.

Third rung of the encoder ladder from docs/sbm_explainer/design.md:

    rung 1: SHAP / LIME attribution vectors         (plain-sbm-poc.md, regime b)
    rung 2: XGBoost leaf indices on HONEST booster  (run_activation_sbm.py)
    rung 3: behavioral fingerprint under SCAFFOLD   ← this script

Rung 2's weakness: the honest booster's leaves don't reflect the scaffold gate
(BiasedModel applies the penalty *post-prediction*, no tree structure changes).
Rung 3 encodes gate-awareness explicitly: e_i = [f_scaffold(x_i),
f_scaffold(x_i + δ_1), …, f_scaffold(x_i + δ_K)]. Two samples get similar
fingerprints iff the scaffold routes them (and small neighbourhoods of them)
similarly — which for the sqft_lot scaffold means "same side of the P75 gate
under small perturbations".

Two encoder variants run end-to-end:

    (A) Gaussian     — K random perturbations, δ ~ N(0, 0.1 σ_j) per feature.
                       Fingerprint length = K+1. Ambient gate-sensitivity.
    (B) Targeted     — replace only sqft_lot with a grid around P75:
                       {P10, P25, P50, 0.99·P75, 1.01·P75, P90, P99}.
                       Fingerprint length = 7. Dense gate response curve.

For each variant + each scaffold (LIME-OOD primary, SHAP-OOD secondary):
    (1) cosine 8-NN graph on fingerprint vectors
    (2) Fix 3 Beta-Bernoulli log BF on (low-low, high-high, across) edges
    (3) plain SBM + pEx-SBM, K=4, 5 seeds; MAP NMI vs layer
    (4) posterior co-clustering ratio (C_within / C_across) — regime (a) vs (b)

Compared head-to-head against the activation run and the SHAP/LIME auditor
numbers in a single results/sbm_explainer/fingerprint_sbm/comparison.md.

Output: results/sbm_explainer/fingerprint_sbm/
    fingerprints_{variant}_{scaffold}.npy
    graph_{variant}_{scaffold}.npy
    summary.json, comparison.md
"""

from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import betaln
from scipy.stats import beta as beta_dist
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.attacks.scaffold import AdversarialScaffold
from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


FEATURE = "sqft_lot"
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
SCAFFOLD_REF = "scaffold_sqft_lot"
N_GAUSS = 20           # number of Gaussian perturbations per point
GAUSS_SIGMA_FRAC = 0.1 # δ_j ~ N(0, 0.1 · σ_j)
TARGETED_GRID_QUANTILES = [0.10, 0.25, 0.50, 0.745, 0.755, 0.90, 0.99]
VARIANTS = ("gaussian", "targeted")


# ── Encoders ────────────────────────────────────────────────────────────────

def gaussian_fingerprints(scaffold, X: np.ndarray, feature_std: np.ndarray,
                          n_pert: int = N_GAUSS, seed: int = 0) -> np.ndarray:
    """Fingerprint e_i = [f(x_i), f(x_i + δ_1), …, f(x_i + δ_K)]. Shape (n, K+1)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, n_pert + 1), dtype=np.float64)
    E[:, 0] = scaffold.predict(X)
    scale = GAUSS_SIGMA_FRAC * feature_std[None, :]        # (1, d)
    for k in range(n_pert):
        delta = rng.normal(size=(n, d)) * scale
        E[:, k + 1] = scaffold.predict(X + delta)
    return E


def targeted_fingerprints(scaffold, X: np.ndarray, feature_names: list[str],
                          bias_feature: str, grid_values: np.ndarray) -> np.ndarray:
    """Fingerprint e_i = [f(x_i with bias←v) for v in grid]. Shape (n, len(grid))."""
    j = feature_names.index(bias_feature)
    n = X.shape[0]
    E = np.empty((n, len(grid_values)), dtype=np.float64)
    Xj = X.copy()
    for k, v in enumerate(grid_values):
        Xj[:, j] = v
        E[:, k] = scaffold.predict(Xj)
    return E


# ── Graph builder — 8-NN on cosine similarity ───────────────────────────────

def cosine_knn_graph(E: np.ndarray, k: int) -> np.ndarray:
    # Row-normalise, then S = E @ E^T. (n, d) with d small → dense is fine.
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    Z = E / norms
    S = Z @ Z.T
    n = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    top_k = np.argpartition(-S, k, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.int8)
    rows = np.repeat(np.arange(n), k)
    A[rows, top_k.reshape(-1)] = 1
    A = ((A + A.T) > 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


# ── Fix 3 ────────────────────────────────────────────────────────────────────

def fix3(A: np.ndarray, layers: np.ndarray) -> dict:
    n = len(layers)
    iu = np.triu_indices(n, k=1)
    edges = A[iu].astype(np.int64)
    pair_sum = layers[iu[0]].astype(np.int64) + layers[iu[1]].astype(np.int64)
    cells = {}
    log_m1, tot_M, tot_T = 0.0, 0, 0
    for kk, name in [(0, "low_low"), (2, "high_high"), (1, "across")]:
        mask = pair_sum == kk
        M = int(edges[mask].sum()); T = int(mask.sum())
        a, b = 1 + M, 1 + T - M
        ci = (float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b))) if T > 0 else (np.nan, np.nan)
        log_m1 += betaln(a, b) - betaln(1, 1)
        tot_M += M; tot_T += T
        cells[name] = dict(M=M, T=T, posterior_mean=a / (a + b), ci_95=ci)
    log_m0 = betaln(1 + tot_M, 1 + tot_T - tot_M) - betaln(1, 1)
    th11, th00, th01 = (cells["high_high"]["posterior_mean"],
                        cells["low_low"]["posterior_mean"],
                        cells["across"]["posterior_mean"])
    return dict(
        cells=cells,
        log_bayes_factor=float(log_m1 - log_m0),
        theta_high_over_across=float(th11 / th01) if th01 > 0 else float("nan"),
        theta_low_over_across=float(th00 / th01) if th01 > 0 else float("nan"),
    )


# ── SBM fits ────────────────────────────────────────────────────────────────

def fit_plain_multiseed(A: np.ndarray, layer: np.ndarray):
    C_sum = np.zeros_like(A, dtype=np.float64)
    nmis = []
    for s in range(N_SEEDS):
        m = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                    gamma=1.0, seed=s, verbose=False)
        C_sum += m.co_clustering_matrix()
        nmis.append(layer_separation(m.z, layer))
    C = C_sum / N_SEEDS
    within, across = layer_coclustering(C, layer)
    return dict(nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)),
                C_within=float(within), C_across=float(across),
                gap=float(within - across),
                ratio=float(within / across) if across > 0 else float("nan"))


def fit_pex_multiseed(A: np.ndarray, layer: np.ndarray):
    nmis = []
    for s in range(N_SEEDS):
        m = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                        thin=5, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(m.z, layer))
    return dict(nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)))


# ── Reference pulls ────────────────────────────────────────────────────────

def pull_references():
    ref = {"shap_lime": {"battery": {}, "fix3": {}, "coclust": {}},
           "activation": {}}
    # SHAP/LIME (auditor thread, sqft_lot cells, penalty 0.20)
    bp = RESULTS_DIR / "sbm_poc" / "battery" / "summary.json"
    if bp.exists():
        for c in json.load(open(bp)).get("cells", []):
            if c["feature"] == FEATURE:
                ref["shap_lime"]["battery"][c["explainer"]] = dict(
                    plain_adv=c["plain"]["adv_nmi_mean"],
                    pex_adv=c["pex"]["adv_nmi_mean"])
    ld = RESULTS_DIR / "sbm_poc" / "layer_density" / "summary.json"
    if ld.exists():
        for r in json.load(open(ld)).get("main_battery", []):
            if r["feature"] == FEATURE:
                ref["shap_lime"]["fix3"][r["explainer"]] = r["adv"]["log_bayes_factor"]
    cd = RESULTS_DIR / "sbm_poc" / "coclustering_diagnostic" / "summary.json"
    if cd.exists():
        for r in json.load(open(cd))["rows"]:
            if r["explainer"] in ("lime", "shap") and abs(r["penalty"] - 0.20) < 1e-6:
                ref["shap_lime"]["coclust"][r["explainer"]] = dict(
                    nmi=r["nmi_mean"], ratio=r["ratio"], gap=r["gap"])
    # Activation thread
    ap = RESULTS_DIR / "sbm_explainer" / "activation_sbm" / "summary.json"
    if ap.exists():
        a = json.load(open(ap))
        ref["activation"] = dict(
            log_bayes_factor=a["fix3"]["log_bayes_factor"],
            plain_nmi=a["plain"]["nmi_mean"],
            pex_nmi=a["pex"]["nmi_mean"],
            ratio=a["plain"]["ratio"])
    return ref


# ── Scaffold builder (rebuilds the scaffold used by generate_scaffold_explanations.py) ──

def build_scaffolds(X_train_np, y_train_vals, feature_names, bias_col_idx,
                    base_feature_indices, threshold, penalty, cfg):
    honest = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    f_biased = BiasedModel(honest, base_feature_indices, bias_col_idx,
                            threshold=threshold, penalty=penalty)
    safe_indices = [i for i, f in enumerate(feature_names)
                    if f not in {feature_names[bias_col_idx], "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train_vals, cfg["innocuous_model"])
    print("Innocuous model fitted.")
    print("Training LIME OOD detector...")
    lime_sc = AdversarialScaffold.for_lime(f_biased, psi, X_train_np, cfg["lime_ood"])
    print(f"  LIME OOD acc = {lime_sc.ood_diagnostics['accuracy']:.4f}")
    print("Training SHAP OOD detector...")
    shap_sc = AdversarialScaffold.for_shap(f_biased, psi, X_train_np, cfg["shap_ood"])
    print(f"  SHAP OOD acc = {shap_sc.ood_diagnostics['accuracy']:.4f}")
    return lime_sc, shap_sc


def main():
    out_dir = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    X_full, y = load_housing(dataset="demographics", feature_columns=None,
                              target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
    sweep_cols = list(baseline_features)  # sqft_lot is already in baseline
    df_sweep = X_full[sweep_cols]
    feature_names = list(df_sweep.columns)
    bias_col_idx = feature_names.index(FEATURE)
    base_feature_indices = [feature_names.index(f) for f in baseline_features]
    threshold = float(np.percentile(df_sweep[FEATURE].values, 75))
    penalty = cfg["biased_model"]["penalty"]
    print(f"bias={FEATURE}  P75={threshold:.1f}  penalty={penalty}")

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"])
    X_train_np = X_train.values
    X_test_np = X_test.values

    # Replicate generate_scaffold_explanations.py row selection: FIRST 200 rows of X_test
    n = 200
    X_explain = X_test.iloc[:n]
    X_explain_np = X_explain.values
    test_indices = X_test.index.values[:n]

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_indices]
    layer = (col > threshold).astype(int)
    print(f"n={n}  high={int(layer.sum())}  low={int((layer == 0).sum())}")

    # Targeted grid: percentiles of sqft_lot on full column (not just test pts)
    grid = np.quantile(df_sweep[FEATURE].values, TARGETED_GRID_QUANTILES)
    print(f"targeted grid (quantiles {TARGETED_GRID_QUANTILES}): "
          f"[{', '.join(f'{v:.0f}' for v in grid)}]")

    feature_std = X_train.std(ddof=1).values.astype(np.float64)

    lime_sc, shap_sc = build_scaffolds(
        X_train_np, y_train.values, feature_names, bias_col_idx,
        base_feature_indices, threshold, penalty, cfg)

    scaffolds = {"lime": lime_sc, "shap": shap_sc}
    ref = pull_references()
    results = {
        "feature": FEATURE, "threshold": threshold, "penalty": penalty,
        "K": K, "k_neighbours": K_NEIGHBOURS, "n": n,
        "n_high": int(layer.sum()), "n_low": int((layer == 0).sum()),
        "n_gaussian_perturbations": N_GAUSS,
        "gaussian_sigma_fraction": GAUSS_SIGMA_FRAC,
        "targeted_grid_quantiles": TARGETED_GRID_QUANTILES,
        "targeted_grid_values": [float(v) for v in grid],
        "cells": [],
        "references": ref,
    }

    for scaff_name, scaff in scaffolds.items():
        for variant in VARIANTS:
            print(f"\n── {variant.upper()} × {scaff_name.upper()} scaffold ──")
            if variant == "gaussian":
                E = gaussian_fingerprints(scaff, X_explain_np, feature_std, seed=0)
            else:
                E = targeted_fingerprints(scaff, X_explain_np, feature_names,
                                          FEATURE, grid)
            np.save(out_dir / f"fingerprints_{variant}_{scaff_name}.npy", E)
            print(f"fingerprints shape = {E.shape}")

            A = cosine_knn_graph(E, k=K_NEIGHBOURS)
            np.save(out_dir / f"graph_{variant}_{scaff_name}.npy", A)
            print(f"edges = {A.sum() // 2}")

            f3 = fix3(A, layer)
            plain = fit_plain_multiseed(A, layer)
            pex = fit_pex_multiseed(A, layer)
            print(f"  logBF = {f3['log_bayes_factor']:+.2f}   "
                  f"θ_11/θ_01 = {f3['theta_high_over_across']:.2f}")
            print(f"  plain NMI = {plain['nmi_mean']:.3f} ± {plain['nmi_std']:.3f}   "
                  f"ratio = {plain['ratio']:.2f}")
            print(f"  pEx   NMI = {pex['nmi_mean']:.3f} ± {pex['nmi_std']:.3f}   "
                  f"Δ = {pex['nmi_mean'] - plain['nmi_mean']:+.3f}")
            results["cells"].append(dict(
                variant=variant, scaffold=scaff_name,
                fingerprint_dim=int(E.shape[1]),
                n_edges=int(A.sum() // 2),
                fix3=f3, plain=plain, pex=pex,
            ))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out_dir / 'summary.json'}")

    # Comparison markdown
    md = [
        "# Fingerprint-encoder pEx-SBM — rung 3 of the encoder ladder",
        "",
        f"Cell: `{FEATURE}` (layer = >P75 = {threshold:.0f}), n={n}, "
        f"K={K}, 8-NN cosine.",
        "",
        "## Cells",
        "",
        "| encoder | scaffold | dim | log BF | plain NMI | pEx NMI | Δ (pex−plain) | C ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in results["cells"]:
        md.append(
            f"| {c['variant']} | {c['scaffold'].upper()} | {c['fingerprint_dim']} | "
            f"{c['fix3']['log_bayes_factor']:+.1f} | "
            f"{c['plain']['nmi_mean']:.3f} ± {c['plain']['nmi_std']:.3f} | "
            f"{c['pex']['nmi_mean']:.3f} ± {c['pex']['nmi_std']:.3f} | "
            f"{c['pex']['nmi_mean'] - c['plain']['nmi_mean']:+.3f} | "
            f"{c['plain']['ratio']:.2f} |"
        )
    md += ["",
           "## Ladder comparison (sqft_lot, scaffold penalty 0.20)",
           "",
           "| rung | encoder | log BF | plain NMI | pEx NMI | C ratio |",
           "|---|---|---|---|---|---|"]
    for expl in ("lime", "shap"):
        bf = ref["shap_lime"]["fix3"].get(expl, float("nan"))
        bat = ref["shap_lime"]["battery"].get(expl, {})
        co = ref["shap_lime"]["coclust"].get(expl, {})
        md.append(
            f"| 1 | {expl.upper()} attr | {bf:+.1f} | "
            f"{bat.get('plain_adv', float('nan')):.3f} | "
            f"{bat.get('pex_adv', float('nan')):.3f} | "
            f"{co.get('ratio', float('nan')):.2f} |"
        )
    act = ref.get("activation") or {}
    if act:
        md.append(
            f"| 2 | honest leaves | {act.get('log_bayes_factor', float('nan')):+.1f} | "
            f"{act.get('plain_nmi', float('nan')):.3f} | "
            f"{act.get('pex_nmi', float('nan')):.3f} | "
            f"{act.get('ratio', float('nan')):.2f} |"
        )
    for c in results["cells"]:
        md.append(
            f"| 3 | fp:{c['variant']}+{c['scaffold']} | "
            f"{c['fix3']['log_bayes_factor']:+.1f} | "
            f"{c['plain']['nmi_mean']:.3f} | {c['pex']['nmi_mean']:.3f} | "
            f"{c['plain']['ratio']:.2f} |"
        )
    md += ["",
           "## Reading the table",
           "",
           "- **C ratio** (posterior co-clustering within/across) decides regime "
           "(a) MAP-collapse vs (b) posterior-blindness at matched log BF. "
           "Regime (b) shows C ratio ≈ 1 despite large log BF.",
           "- **Δ (pex − plain)** = prior effect. If the activation thread's "
           "Δ ≈ 0 carries over to fingerprint graphs, the pEx null is "
           "encoder-invariant and the Dirichlet-Categorical prior genuinely "
           "cannot differ from plain under any edge graph in this family.",
           "- Targeted vs Gaussian: targeted is the cleanest gate-probe (all "
           "high points should have near-identical response curves). Gaussian "
           "is the ambient-sensitivity version.",
           ""]
    (out_dir / "comparison.md").write_text("\n".join(md))
    print(f"Saved {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
