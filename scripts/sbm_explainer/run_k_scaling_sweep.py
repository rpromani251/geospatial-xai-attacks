"""K-scaling sweep: Δ(pEx − plain) as a function of K (# refinement perts).

Mechanism prediction (anchor-vs-refinement cosine decomposition):
  - Smooth innocuous f_inn: refinement cross-term ≈ O(K·h(x_i,x_j))
      → layer signal drowns as O(1/K)
  - XOR / non-smooth f_inn: refinement cross-term concentrates at a
      point-independent constant; fluctuations ~ O(√K)
      → layer signal decays only as O(1/√K)

Sweep K ∈ {5, 10, 20, 40, 80} on the headline cell of each of the four
datasets and compare the decay rate of Δ.
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SLACK = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/fooling-lime-shap")
sys.path.insert(0, str(SLACK))

from src.sbm import fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.utils.paths import CONFIGS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

import warnings
warnings.filterwarnings("ignore")

KS = [5, 10, 20, 40, 80]
FP_SEEDS = [0, 1, 2]           # fingerprint (δ draw) seeds
K_NEIGHBOURS = 8
SBM_K = 4
N_ITER = 400
BURN_IN = 200
N_SBM_SEEDS = 5
N_NODES = 200


# ── Shared utilities ────────────────────────────────────────────────────────

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


def fit_cell(A, layer):
    pln, pex, rr = [], [], []
    for s in range(N_SBM_SEEDS):
        mp = fit_sbm(A, K=SBM_K, n_iter=N_ITER, burn_in=BURN_IN, thin=5,
                     gamma=1.0, seed=s, verbose=False)
        me = fit_pex_sbm(A, layer=layer, K=SBM_K, n_iter=N_ITER,
                         burn_in=BURN_IN, thin=5, gamma=1.0, seed=s,
                         verbose=False)
        pln.append(layer_separation(mp.z, layer))
        pex.append(layer_separation(me.z, layer))
        Cp = mp.co_clustering_matrix()
        w, a = layer_coclustering(Cp, layer)
        rr.append(w / a if a > 0 else float("nan"))
    pln = np.asarray(pln); pex = np.asarray(pex); rr = np.asarray(rr)
    return dict(plain=float(pln.mean()), plain_std=float(pln.std(ddof=1)),
                pex=float(pex.mean()), pex_std=float(pex.std(ddof=1)),
                delta=float((pex - pln).mean()),
                delta_std=float((pex - pln).std(ddof=1)),
                C_ratio=float(rr.mean()), n_edges=int(A.sum() // 2))


# ── Fingerprint builders (dataset-specific) ────────────────────────────────

def housing_fp(scaff, X, feature_std, K, seed, sigma_frac=0.1):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, K + 1))
    E[:, 0] = scaff.predict(X)
    scale = sigma_frac * feature_std[None, :]
    for k in range(K):
        E[:, k + 1] = scaff.predict(X + rng.normal(size=(n, d)) * scale)
    return E


def slack_fp(scaff, X, sigma, K, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E = np.empty((n, K + 1))
    E[:, 0] = scaff.predict_proba(X)[:, 1]
    for k in range(K):
        pert = X + rng.normal(size=(n, d)) * sigma
        E[:, k + 1] = scaff.predict_proba(pert)[:, 1]
    return E


# ── Dataset setups ──────────────────────────────────────────────────────────

def setup_housing():
    """Returns: (label, builder) where builder(K, seed) -> E. Also (layer,)."""
    from src.data.load_housing import load_housing
    from src.attacks.biased_model import BiasedModel, InnocuousModel
    from src.attacks.scaffold import AdversarialScaffold
    from src.utils.io import load_model

    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    X_full, y = load_housing(dataset="demographics", feature_columns=None,
                              target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
    df_sweep = X_full[list(baseline_features)]
    feature_names = list(df_sweep.columns)
    bias_idx = feature_names.index("sqft_lot")
    base_idx = [feature_names.index(f) for f in baseline_features]
    threshold = float(np.percentile(df_sweep["sqft_lot"].values, 75))

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"])
    X_explain = X_test.iloc[:N_NODES].values
    test_indices = X_test.index.values[:N_NODES]

    import pandas as pd
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
    col = df["sqft_lot"].values[test_indices]
    layer = (col > threshold).astype(int)
    feature_std = X_train.std(ddof=1).values.astype(np.float64)

    honest = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    f_biased = BiasedModel(honest, base_idx, bias_idx, threshold=threshold,
                           penalty=cfg["biased_model"]["penalty"])
    safe = [i for i, f in enumerate(feature_names)
            if f not in {"sqft_lot", "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe)
    psi.fit(X_train.values, y_train.values, cfg["innocuous_model"])
    lime_sc = AdversarialScaffold.for_lime(f_biased, psi, X_train.values,
                                           cfg["lime_ood"])
    print(f"  housing LIME OOD acc = {lime_sc.ood_diagnostics['accuracy']:.4f}")

    def builder(K, seed):
        return housing_fp(lime_sc, X_explain, feature_std, K, seed)

    return "housing_lime", builder, layer


def _slack_common(get_preprocess_fn, ProtectedFn, InnocuousFn,
                   protected_col, cat_names, scaff_type, sigma, extra=None):
    sys.path.insert(0, str(SLACK))
    from adversarial_models import Adversarial_Lime_Model, Adversarial_Kernel_SHAP_Model
    from utils import Params

    cwd = os.getcwd()
    os.chdir(SLACK)
    try:
        params = Params(str(SLACK / "model_configurations" / "experiment_params.json"))
        np.random.seed(params.seed)
        X, y, cols = get_preprocess_fn(params)
    finally:
        os.chdir(cwd)

    if extra is not None:
        extra(X)
    features = list(X.columns)
    prot_idx = features.index(protected_col)
    X_np = X.values

    xtr, xte, ytr, yte = train_test_split(X_np, y, test_size=0.1, random_state=0)
    ss = StandardScaler().fit(xtr)
    xtr = ss.transform(xtr); xte = ss.transform(xte)

    pos, neg = params.positive_outcome, params.negative_outcome
    f_biased = ProtectedFn(features, pos, neg)
    f_inn = InnocuousFn(features, pos, neg, xtr)

    cat_idx = [features.index(c) for c in cat_names if c in features]

    if scaff_type == "lime":
        scaff = Adversarial_Lime_Model(f_biased, f_inn).train(
            xtr, ytr, feature_names=features,
            categorical_features=cat_idx, perturbation_multiplier=30)
    else:
        scaff = Adversarial_Kernel_SHAP_Model(f_biased, f_inn).train(
            xtr, ytr, feature_names=features)

    n_take = min(N_NODES, xte.shape[0])
    X_explain = xte[:n_take]
    layer = (X_explain[:, prot_idx] > 0).astype(int)

    def builder(K, seed):
        return slack_fp(scaff, X_explain, sigma, K, seed)

    return builder, layer


def setup_compas():
    from get_data import get_and_preprocess_compas_data

    def one_hot(a):
        o = np.zeros((len(a), 2), dtype=float); o[np.arange(len(a)), a] = 1.0; return o

    class RacistF:
        def __init__(self, features, pos, neg):
            self.race_idx = features.index("race")
            self.pos, self.neg = pos, neg
        def predict(self, X):
            return np.array([self.neg if x[self.race_idx] > 0 else self.pos for x in X])
        def predict_proba(self, X):
            return one_hot(np.asarray(self.predict(X), dtype=int))

    class InnocuousPsi:
        def __init__(self, features, pos, neg, _xtr):
            self.idx = features.index("unrelated_column_one")
            self.pos, self.neg = pos, neg
        def predict_proba(self, X):
            p = np.array([self.neg if x[self.idx] > 0 else self.pos for x in X])
            return one_hot(p.astype(int))

    def add_cols(X):
        X["unrelated_column_one"] = np.random.choice([0, 1], size=X.shape[0])
        X["unrelated_column_two"] = np.random.choice([0, 1], size=X.shape[0])

    cats = ["unrelated_column_one", "unrelated_column_two",
            "c_charge_degree_F", "c_charge_degree_M",
            "two_year_recid", "race", "sex_Male", "sex_Female"]

    builder, layer = _slack_common(
        get_and_preprocess_compas_data, RacistF, InnocuousPsi,
        "race", cats, "shap", sigma=0.1, extra=add_cols)
    return "compas_shap", builder, layer


def setup_german():
    from get_data import get_and_preprocess_german

    def one_hot(a):
        o = np.zeros((len(a), 2), dtype=float); o[np.arange(len(a)), a] = 1.0; return o

    class RacistF:
        def __init__(self, features, pos, neg):
            self.gidx = features.index("Gender")
            self.pos, self.neg = pos, neg
        def predict(self, X):
            return np.array([self.neg if x[self.gidx] < 0 else self.pos for x in X])
        def predict_proba(self, X):
            return one_hot(np.asarray(self.predict(X), dtype=int))

    class InnocuousPsi:
        def __init__(self, features, pos, neg, xtr):
            self.lidx = features.index("LoanRateAsPercentOfIncome")
            self.pos, self.neg = pos, neg
            self.mean = float(np.mean(xtr[:, self.lidx]))
        def predict_proba(self, X):
            p = np.array([self.neg if x[self.lidx] > self.mean else self.pos for x in X])
            return one_hot(p.astype(int))

    cats = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone',
            'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200',
            'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
            'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere',
            'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
            'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
            'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']

    builder, layer = _slack_common(
        get_and_preprocess_german, RacistF, InnocuousPsi,
        "Gender", cats, "shap", sigma=1.0)
    return "german_shap", builder, layer


def setup_cc_psi_two():
    from get_data import get_and_preprocess_cc

    def one_hot(a):
        o = np.zeros((len(a), 2), dtype=float); o[np.arange(len(a)), a] = 1.0; return o

    class RacistF:
        def __init__(self, features, pos, neg):
            self.idx = features.index("racePctWhite numeric")
            self.pos, self.neg = pos, neg
        def predict(self, X):
            return np.array([self.neg if x[self.idx] > 0 else self.pos for x in X])
        def predict_proba(self, X):
            return one_hot(np.asarray(self.predict(X), dtype=int))

    class InnocuousPsiTwo:
        def __init__(self, features, pos, neg, _xtr):
            self.i1 = features.index("unrelated_column_one")
            self.i2 = features.index("unrelated_column_two")
            self.pos, self.neg = pos, neg
        def predict_proba(self, X):
            A = np.where(X[:, self.i1] > 0.5, self.pos, self.neg)
            B = np.where(X[:, self.i2] < -0.5, self.pos, self.neg)
            preds = np.logical_xor(A.astype(bool), B.astype(bool)).astype(int)
            return one_hot(preds)

    def add_cols(X):
        X["unrelated_column_one"] = np.random.choice([0, 1], size=X.shape[0])
        X["unrelated_column_two"] = np.random.choice([0, 1], size=X.shape[0])

    cats = ["unrelated_column_one", "unrelated_column_two"]

    builder, layer = _slack_common(
        get_and_preprocess_cc, RacistF, InnocuousPsiTwo,
        "racePctWhite numeric", cats, "lime", sigma=1.0, extra=add_cols)
    return "cc_psi_two_lime", builder, layer


# ── Main sweep ──────────────────────────────────────────────────────────────

def run_dataset(label, builder, layer):
    print(f"\n{'═' * 60}\n  {label}  (n={len(layer)}, layer.sum()={int(layer.sum())})\n{'═' * 60}")
    per_K = {}
    for K in KS:
        t0 = time.time()
        # Average across fingerprint seeds
        seed_cells = []
        for fs in FP_SEEDS:
            E = builder(K, fs)
            A = cosine_knn(E, K_NEIGHBOURS)
            c = fit_cell(A, layer)
            seed_cells.append(c)
        # Aggregate
        deltas = np.array([c["delta"] for c in seed_cells])
        plains = np.array([c["plain"] for c in seed_cells])
        pexs = np.array([c["pex"] for c in seed_cells])
        ratios = np.array([c["C_ratio"] for c in seed_cells])
        summary = dict(
            K=K,
            plain_mean=float(plains.mean()), plain_std=float(plains.std(ddof=1)),
            pex_mean=float(pexs.mean()),   pex_std=float(pexs.std(ddof=1)),
            delta_mean=float(deltas.mean()), delta_std=float(deltas.std(ddof=1)),
            C_ratio=float(ratios.mean()),
            per_seed=[dict(fp_seed=fs, **c) for fs, c in zip(FP_SEEDS, seed_cells)],
        )
        per_K[K] = summary
        dt = time.time() - t0
        print(f"  K={K:3d}  plain={plains.mean():.3f}  pex={pexs.mean():.3f}  "
              f"Δ={deltas.mean():+.3f}±{deltas.std(ddof=1):.3f}  "
              f"ratio={ratios.mean():.2f}  ({dt:.0f}s)")
    return per_K


def main():
    out = RESULTS_DIR / "sbm_explainer" / "k_scaling"
    out.mkdir(parents=True, exist_ok=True)

    setups = [setup_housing, setup_compas, setup_german, setup_cc_psi_two]
    all_results = {}
    for setup in setups:
        label, builder, layer = setup()
        per_K = run_dataset(label, builder, layer)
        all_results[label] = per_K
        with open(out / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved {out / 'results.json'}")

    # Scaling fit: log Δ vs log K, slope = -α. Expect α≈1 (smooth) or α≈0.5 (XOR).
    print("\n── Scaling fits (log Δ = −α log K + c, only positive Δ) ──")
    import math
    for label, per_K in all_results.items():
        Ks = np.array(sorted(per_K.keys()))
        Ds = np.array([per_K[int(k)]["delta_mean"] for k in Ks])
        mask = Ds > 0
        if mask.sum() < 2:
            print(f"  {label:>25s}  too few positive Δ for fit (Δ={Ds.tolist()})")
            continue
        lx = np.log(Ks[mask]); ly = np.log(Ds[mask])
        slope, intercept = np.polyfit(lx, ly, 1)
        predicted = np.exp(intercept) * Ks ** slope
        rss = float(np.sum((Ds[mask] - predicted[mask]) ** 2))
        print(f"  {label:>25s}  α = {-slope:+.2f}   c = {intercept:+.2f}   "
              f"K→Δ: {dict(zip(Ks.tolist(), np.round(Ds, 3).tolist()))}")


if __name__ == "__main__":
    main()
