"""pEx-SBM-as-explainer pipeline on activation-space graphs.

End-to-end run on the 200 sqft_lot scaffold test points:

    (1) Encoder       : biased XGBoost booster leaf indices (n, T=500 trees)
    (2) Similarity    : pairwise leaf-match fraction (Hamming-agreement / T)
    (3) Graph         : 8-NN, symmetric OR, binary — same hyperparameters
                        as the SHAP/LIME auditor pipeline for direct
                        comparability to plain-sbm-poc.md.
    (4) Diagnostics   : Fix 3 Beta-Bernoulli log BF + §13d posterior
                        co-clustering gap. These decide regime (a) vs (b)
                        on the activation graph.
    (5) Fits          : plain SBM and pEx-SBM, K=4, 5 seeds, coloring =
                        sqft_lot above-P75 indicator.
    (6) Side-by-side  : activation-graph numbers vs the SHAP/LIME numbers
                        from results/sbm_poc/battery & layer_density.

Hypothesis tested (design.md): activation graphs shift regime (b) → (a)
because gated computations produce piecewise-constant representations
downstream of the gate. Leaf-index vectors are the XGBoost equivalent of
penultimate-layer activations.

Output: results/sbm_explainer/activation_sbm/{leaf_vectors.npy, graph.npy,
summary.json, comparison.md}
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
from src.sbm import build_knn_similarity_graph, fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


FEATURE = "sqft_lot"
THRESHOLD = 9361.0          # P75 on the scaffold sweep, matches plain-sbm-poc.md
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 5
SCAFFOLD_REF = "scaffold_sqft_lot"       # test_indices source


# ── (1) Encoder — biased booster leaf indices ───────────────────────────────

def leaf_vectors(model, X) -> np.ndarray:
    """XGBRegressor.apply(X) returns (n, T) leaf indices — one int per tree.
    Two rows agreeing in leaf t means the model routed both samples down the
    same path in tree t; identical rows = identical computation."""
    return model.apply(X).astype(np.int64)


# ── (2) Similarity — leaf-match fraction ────────────────────────────────────

def leaf_match_similarity(leaves: np.ndarray) -> np.ndarray:
    """S[i, j] = (1/T) * sum_t 1[leaf_t(i) == leaf_t(j)]. Symmetric, in [0, 1]."""
    n, T = leaves.shape
    # Pairwise Hamming agreement: row-wise equality count over trees.
    # O(n^2 T) in space; n=200, T=500 → 20M comparisons, fine.
    S = np.zeros((n, n), dtype=np.float64)
    for t in range(T):
        col = leaves[:, t]
        S += (col[:, None] == col[None, :]).astype(np.float64)
    S /= T
    return S


# ── (3) Graph — 8-NN symmetric OR, binary ───────────────────────────────────
# Use build_knn_similarity_graph with the similarity matrix treated as an
# embedding's inner-product proxy. The existing helper expects an (n, d)
# array and computes cosine internally; here we already have (n, n) S, so
# we hand-build the k-NN graph to stay equivalent.

def knn_graph_from_similarity(S: np.ndarray, k: int) -> np.ndarray:
    n = S.shape[0]
    A = np.zeros_like(S, dtype=np.int8)
    # Zero self-similarity before ranking.
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    top_k = np.argpartition(-S2, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = top_k.reshape(-1)
    A[rows, cols] = 1
    A = ((A + A.T) > 0).astype(np.int8)      # symmetric OR
    np.fill_diagonal(A, 0)
    return A


# ── (4a) Fix 3 Beta-Bernoulli layer-density test ────────────────────────────

def fix3(A: np.ndarray, layers: np.ndarray) -> dict:
    n = len(layers)
    iu = np.triu_indices(n, k=1)
    edges = A[iu].astype(np.int64)
    pair_sum = layers[iu[0]].astype(np.int64) + layers[iu[1]].astype(np.int64)
    cells = {}
    log_m1 = 0.0
    tot_M, tot_T = 0, 0
    for kk, name in [(0, "low_low"), (2, "high_high"), (1, "across")]:
        mask = pair_sum == kk
        M = int(edges[mask].sum())
        T = int(mask.sum())
        a, b = 1 + M, 1 + T - M
        mean = a / (a + b)
        ci = (float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b))) if T > 0 else (np.nan, np.nan)
        log_m1 += betaln(a, b) - betaln(1, 1)
        tot_M += M; tot_T += T
        cells[name] = dict(M=M, T=T, posterior_mean=mean, ci_95=ci)
    log_m0 = betaln(1 + tot_M, 1 + tot_T - tot_M) - betaln(1, 1)
    th11, th00, th01 = (cells["high_high"]["posterior_mean"],
                        cells["low_low"]["posterior_mean"],
                        cells["across"]["posterior_mean"])
    return dict(
        cells=cells,
        log_bayes_factor=float(log_m1 - log_m0),
        theta_high_over_across=float(th11 / th01) if th01 > 0 else float("nan"),
        theta_low_over_across=float(th00 / th01)  if th01 > 0 else float("nan"),
    )


# ── (4b) Posterior co-clustering gap, averaged over seeds ───────────────────

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
    return dict(
        nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)),
        C_within=float(within), C_across=float(across),
        gap=float(within - across), ratio=float(within / across) if across > 0 else float("nan"),
    )


def fit_pex_multiseed(A: np.ndarray, layer: np.ndarray):
    nmis = []
    for s in range(N_SEEDS):
        m = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                        thin=5, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(m.z, layer))
    return dict(nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)))


# ── (5) Pull SHAP/LIME numbers from the auditor thread for comparison ───────

def pull_shaplime_reference():
    out = {"battery": {}, "fix3": {}, "coclust": {}}
    bp = RESULTS_DIR / "sbm_poc" / "battery" / "summary.json"
    if bp.exists():
        for c in json.load(open(bp)).get("cells", []):
            if c["feature"] == FEATURE:
                out["battery"][c["explainer"]] = dict(
                    plain_adv=c["plain"]["adv_nmi_mean"], pex_adv=c["pex"]["adv_nmi_mean"],
                    plain_honest=c["plain"]["honest_nmi_mean"])
    ld = RESULTS_DIR / "sbm_poc" / "layer_density" / "summary.json"
    if ld.exists():
        for r in json.load(open(ld)).get("main_battery", []):
            if r["feature"] == FEATURE:
                out["fix3"][r["explainer"]] = r["adv"]["log_bayes_factor"]
    cd = RESULTS_DIR / "sbm_poc" / "coclustering_diagnostic" / "summary.json"
    if cd.exists():
        for r in json.load(open(cd))["rows"]:
            # reference cell = penalty closest to the scaffold default (0.35 was original default, not in new grid; use 0.20 as strongest tested)
            if r["explainer"] in ("lime", "shap") and abs(r["penalty"] - 0.20) < 1e-6:
                out["coclust"][r["explainer"]] = dict(
                    nmi=r["nmi_mean"], ratio=r["ratio"], gap=r["gap"])
    return out


def main():
    out_dir = RESULTS_DIR / "sbm_explainer" / "activation_sbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline features (match honest model training order)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    # Reconstruct the exact 200 test points used by generate_scaffold_explanations.py
    X_full, y = load_housing(dataset="demographics", feature_columns=None,
                              target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
    sweep_cols = list(baseline_features)     # bias feature already ∈ baseline
    df_sweep = X_full[sweep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=0.2, random_state=42)
    test_idx = np.load(RESULTS_DIR / SCAFFOLD_REF / "honest" / "test_indices.npy")
    n = len(test_idx)
    X_explain = X_test.loc[test_idx, baseline_features]
    assert len(X_explain) == n, f"mismatch: {len(X_explain)} vs {n}"

    # Layer labels (same definition as auditor thread: sqft_lot > P75)
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_idx]
    layer = (col > THRESHOLD).astype(int)
    print(f"n={n}  high={int(layer.sum())}  low={int((layer==0).sum())}")

    # (1) Encoder
    model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    leaves = leaf_vectors(model, X_explain)
    print(f"leaves shape = {leaves.shape}  (unique leaves/tree avg = "
          f"{np.mean([len(np.unique(leaves[:, t])) for t in range(leaves.shape[1])]):.1f})")
    np.save(out_dir / "leaf_vectors.npy", leaves)

    # (2) Similarity + (3) 8-NN graph
    S = leaf_match_similarity(leaves)
    A = knn_graph_from_similarity(S, k=K_NEIGHBOURS)
    print(f"similarity: within-high mean = {S[np.ix_(layer==1, layer==1)].mean():.4f}  "
          f"within-low mean = {S[np.ix_(layer==0, layer==0)].mean():.4f}  "
          f"across mean = {S[np.ix_(layer==1, layer==0)].mean():.4f}")
    print(f"graph edges = {A.sum() // 2}  (n_pairs = {n*(n-1)//2})")
    np.save(out_dir / "graph.npy", A)

    # (4a) Fix 3
    f3 = fix3(A, layer)
    print(f"\nFix 3  logBF = {f3['log_bayes_factor']:+.2f}   "
          f"θ_11/θ_01 = {f3['theta_high_over_across']:.2f}")

    # (4b) + (5) posterior co-clustering + plain SBM + pEx-SBM
    plain = fit_plain_multiseed(A, layer)
    pex = fit_pex_multiseed(A, layer)
    print(f"\nplain SBM  NMI = {plain['nmi_mean']:.3f} ± {plain['nmi_std']:.3f}   "
          f"C within/across = {plain['C_within']:.4f}/{plain['C_across']:.4f}   "
          f"ratio = {plain['ratio']:.2f}")
    print(f"pEx-SBM    NMI = {pex['nmi_mean']:.3f} ± {pex['nmi_std']:.3f}")

    # (6) SHAP/LIME reference
    ref = pull_shaplime_reference()

    results = dict(
        feature=FEATURE, threshold=THRESHOLD, K=K, k_neighbours=K_NEIGHBOURS,
        n=n, n_high=int(layer.sum()), n_low=int((layer==0).sum()),
        similarity_summary=dict(
            within_high=float(S[np.ix_(layer==1, layer==1)].mean()),
            within_low=float(S[np.ix_(layer==0, layer==0)].mean()),
            across=float(S[np.ix_(layer==1, layer==0)].mean()),
        ),
        n_edges=int(A.sum() // 2),
        fix3=f3,
        plain=plain,
        pex=pex,
        shap_lime_reference=ref,
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out_dir/'summary.json'}")

    # Comparison markdown
    md = ["# Activation-space pEx-SBM vs SHAP/LIME auditor",
          "",
          f"Cell: `{FEATURE}` (layer = >P75 = {THRESHOLD}), n={n}, K={K}, 8-NN.",
          "",
          "## Regime (a)/(b) diagnostic on activation graph",
          "",
          f"- Fix 3 log BF: **{f3['log_bayes_factor']:+.2f}**  (θ_11/θ_01 = {f3['theta_high_over_across']:.2f})",
          f"- Plain SBM MAP NMI: **{plain['nmi_mean']:.3f} ± {plain['nmi_std']:.3f}**",
          f"- Posterior C within/across ratio: **{plain['ratio']:.2f}**  (gap {plain['gap']:+.4f})",
          f"- pEx-SBM MAP NMI: **{pex['nmi_mean']:.3f} ± {pex['nmi_std']:.3f}**",
          "",
          "## Side-by-side vs SHAP/LIME (sqft_lot adv at scaffold penalty 0.20)",
          "",
          "| Graph source | log BF | plain NMI | pEx NMI | C ratio |",
          "|---|---|---|---|---|",
          f"| **Activation (leaves)** | {f3['log_bayes_factor']:+.1f} | {plain['nmi_mean']:.3f} | {pex['nmi_mean']:.3f} | {plain['ratio']:.2f} |"]
    for expl in ("lime", "shap"):
        bf = ref["fix3"].get(expl)
        bat = ref["battery"].get(expl, {})
        co = ref["coclust"].get(expl, {})
        md.append(f"| SHAP/LIME adv — {expl.upper()} | "
                  f"{bf:+.1f} | {bat.get('plain_adv', float('nan')):.3f} | "
                  f"{bat.get('pex_adv', float('nan')):.3f} | "
                  f"{co.get('ratio', float('nan')):.2f} |")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")
    print(f"Saved {out_dir/'comparison.md'}")


if __name__ == "__main__":
    main()
