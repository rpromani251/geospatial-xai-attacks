"""Sanity check: does the Gauss fingerprint pipeline give Δ > 0 even on the
honest (non-scaffolded) model?

If yes: rung-3's pEx delta is detecting feature-space geometry correlated
with the layer, not the scaffold gate. The "scaffold-attack detector"
framing would need retraction.

If no: rung-3 does require the scaffold to produce the effect, and the
original interpretation survives (even if the mechanism detail needs fixing).

Runs three fingerprint configs for comparison:
  (a) honest XGBoost only             — no scaffold, no gate
  (b) BiasedModel only (no routing)   — gate always on, no InnocuousModel
  (c) InnocuousModel only             — gate absent, but safe-feature regression

For each, K=4 plain SBM + finite-K pEx on Gauss cosine 8-NN graph, 10 seeds.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

FEATURE = "sqft_lot"
K = 4
K_NEIGHBOURS = 8
N_ITER = 400
BURN_IN = 200
N_SEEDS = 10
N_GAUSS = 20
GAUSS_SIGMA_FRAC = 0.1


class PredictWrap:
    def __init__(self, f):
        self._f = f
    def predict(self, X):
        return self._f(X)


def gaussian_fp(model, X, feature_std, n_pert=N_GAUSS, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, n_pert + 1))
    E[:, 0] = model.predict(X)
    scale = GAUSS_SIGMA_FRAC * feature_std[None, :]
    for k in range(n_pert):
        E[:, k + 1] = model.predict(X + rng.normal(size=(n, d)) * scale)
    return E


def cosine_knn(E, k):
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


def fit_cell(A, layer, label):
    plain_nmis, pex_nmis, ratios = [], [], []
    for s in range(N_SEEDS):
        mp = fit_sbm(A, K=K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                     gamma=1.0, seed=s, verbose=False)
        me = fit_pex_sbm(A, layer=layer, K=K, n_iter=N_ITER, burn_in=BURN_IN,
                         thin=5, gamma=1.0, seed=s, verbose=False)
        plain_nmis.append(layer_separation(mp.z, layer))
        pex_nmis.append(layer_separation(me.z, layer))
        Cp = mp.co_clustering_matrix()
        w, a = layer_coclustering(Cp, layer)
        ratios.append(w / a if a > 0 else float("nan"))
    pln = np.asarray(plain_nmis); pex = np.asarray(pex_nmis)
    r = np.asarray(ratios)
    print(f"{label:>25s}  plain={pln.mean():.3f}±{pln.std(ddof=1):.3f}  "
          f"pEx={pex.mean():.3f}±{pex.std(ddof=1):.3f}  "
          f"Δ={pex.mean() - pln.mean():+.3f}  ratio={r.mean():.2f}  "
          f"edges={int(A.sum() // 2)}")
    return dict(
        plain_nmi_mean=float(pln.mean()), plain_nmi_std=float(pln.std(ddof=1)),
        pex_nmi_mean=float(pex.mean()), pex_nmi_std=float(pex.std(ddof=1)),
        delta_mean=float((pex - pln).mean()), delta_std=float((pex - pln).std(ddof=1)),
        C_ratio=float(r.mean()), n_edges=int(A.sum() // 2),
    )


def main():
    out = RESULTS_DIR / "sbm_explainer" / "honest_check"
    out.mkdir(parents=True, exist_ok=True)

    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    X_full, y = load_housing(dataset="demographics", feature_columns=None,
                              target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
    df_sweep = X_full[list(baseline_features)]
    feature_names = list(df_sweep.columns)
    bias_col_idx = feature_names.index(FEATURE)
    base_feature_indices = [feature_names.index(f) for f in baseline_features]
    threshold = float(np.percentile(df_sweep[FEATURE].values, 75))
    penalty = cfg["biased_model"]["penalty"]

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"])
    X_explain = X_test.iloc[:200]
    X_explain_np = X_explain.values
    test_indices = X_test.index.values[:200]

    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df[FEATURE].values[test_indices]
    layer = (col > threshold).astype(int)
    feature_std = X_train.std(ddof=1).values.astype(np.float64)

    honest = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    f_biased = BiasedModel(honest, base_feature_indices, bias_col_idx,
                           threshold=threshold, penalty=penalty)
    safe_indices = [i for i, f in enumerate(feature_names)
                    if f not in {feature_names[bias_col_idx], "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train.values, y_train.values, cfg["innocuous_model"])

    honest_wrap = PredictWrap(lambda X: honest.predict(X))
    biased_wrap = PredictWrap(lambda X: f_biased.predict(X))
    innoc_wrap  = PredictWrap(lambda X: psi.predict(X))

    print(f"n=200  low={int((layer == 0).sum())}  high={int(layer.sum())}")
    print(f"{'model':>25s}  plain                pEx                   Δ           ratio")
    results = {}
    for label, mdl in [
        ("honest XGBoost",    honest_wrap),
        ("BiasedModel only",  biased_wrap),
        ("InnocuousModel only", innoc_wrap),
    ]:
        E = gaussian_fp(mdl, X_explain_np, feature_std, seed=0)
        np.save(out / f"fp_{label.replace(' ', '_')}.npy", E)
        A = cosine_knn(E, K_NEIGHBOURS)
        results[label] = fit_cell(A, layer, label)

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out / 'results.json'}")

    # Reference: the scaffolded Gauss+LIME number we're comparing against
    print("\nFor reference:  scaffold (LIME detector) Gauss fingerprint "
          "→ plain=0.287, pEx=0.334, Δ=+0.048, ratio=2.50")


if __name__ == "__main__":
    main()
