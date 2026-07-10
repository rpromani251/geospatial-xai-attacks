"""Fix TreeSHAP for COMPAS/German and complete CC psi_two pipeline.

Patches:
- explain_treeshap_adapted: robust handling of shap 0.51.0 (n,d,2) output
  AND legacy (2,n,d) list-flattened output from older shap versions.
- CC psi_two: full 2x3 matrix from scratch (model+splits already saved).

Bug diagnosed: prior shap version returned list [arr_class0, arr_class1],
np.array() flattened to (2,n,d). Then sv[:,:,1] yielded (2,n) instead of
(n,d), causing spearman_robustness to fail on shape mismatch -> silent NaN.
Fix: detect (2,n,d) transpose to (n,d,2), then take [:,:,1].
Current shap 0.51.0 returns (n,d,2) directly, handled by existing ndim==3
branch. Both cases now handled.
"""

from __future__ import annotations
import os, sys, json, pickle, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SLACK_REPO = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/fooling-lime-shap")
ROOT = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/geospatial-xai-attacks")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SLACK_REPO))

from src.evaluation.neighborhoods import (
    generate_neighbors_random,
    generate_neighbors_medoid,
    filter_same_prediction,
)
from src.evaluation.robustness import spearman_robustness

RESULTS_DIR = ROOT / "results" / "slack_robustness"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_EXPLAIN = 30
N_NEIGHBORS = 50
N_BACKGROUND = 50
DELTA_CLASS = 0.15
DELTA_REG = 0.15
RANDOM_STATE = 42
LIME_SAMPLES = 500
LEVELS = ["random", "medoid"]
EXPLAINERS = ["TreeSHAP", "KernelSHAP", "LIME"]


# ── FIXED TreeSHAP ──────────────────────────────────────────────────────────
def explain_treeshap_adapted(model, X_combined_df, task):
    """Robust TreeSHAP: handles shap>=0.41 (n,d,2) AND legacy (2,n,d) outputs."""
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_combined_df.values.astype(float))

    if isinstance(sv_raw, list):
        # Legacy shap: list of per-class arrays, each shape (n, d)
        sv = np.stack(sv_raw, axis=-1)  # -> (n, d, n_classes)
    else:
        sv = np.array(sv_raw)

    if task == "classification":
        if sv.ndim == 3:
            n0, n1, n2 = sv.shape
            if n2 == 2:
                # Modern shap: (n, d, 2) — take class-1
                sv = sv[:, :, 1]
            elif n0 == 2:
                # Old flattened: (2, n, d) — transpose then take class-1
                sv = sv.transpose(1, 2, 0)[:, :, 1]
            else:
                # Unknown: try last axis
                sv = sv[:, :, -1]
    # regression: sv already (n, d)
    return sv


# ── KernelSHAP ──────────────────────────────────────────────────────────────
def explain_kernelshap_adapted(model, X_train_vals, X_combined_df, task, n_background=N_BACKGROUND):
    bg = shap.sample(pd.DataFrame(X_train_vals), n_background)
    if task == "classification":
        predict_fn = lambda x: model.predict_proba(x.astype(float))[:, 1]
    else:
        predict_fn = lambda x: model.predict(x.astype(float))
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_combined_df.values.astype(float))
    return np.array(sv)


# ── LIME ────────────────────────────────────────────────────────────────────
def explain_lime_adapted(model, X_train_vals, feature_names, X_test_df, task,
                          num_samples=LIME_SAMPLES, random_state=RANDOM_STATE):
    n_features = len(feature_names)
    if task == "classification":
        explainer = LimeTabularExplainer(
            X_train_vals, feature_names=feature_names,
            mode="classification", random_state=random_state,
        )
        def predict_fn_lime(x):
            return model.predict_proba(x.astype(float))

        lime_values = np.zeros((len(X_test_df), n_features))
        for i, (_, row) in enumerate(X_test_df.iterrows()):
            exp = explainer.explain_instance(
                row.values.astype(float), predict_fn_lime,
                num_features=n_features, num_samples=num_samples, labels=(1,),
            )
            fw = dict(exp.as_map()[1])
            for j in range(n_features):
                lime_values[i, j] = fw.get(j, 0.0)
    else:
        explainer = LimeTabularExplainer(
            X_train_vals, feature_names=feature_names,
            mode="regression", random_state=random_state,
        )
        def predict_fn_reg(x):
            return model.predict(x.astype(float))

        lime_values = np.zeros((len(X_test_df), n_features))
        for i, (_, row) in enumerate(X_test_df.iterrows()):
            exp = explainer.explain_instance(
                row.values.astype(float), predict_fn_reg,
                num_features=n_features, num_samples=num_samples,
            )
            fw = dict(exp.as_map()[1])
            for j in range(n_features):
                lime_values[i, j] = fw.get(j, 0.0)
    return lime_values


# ── Per-point loop ──────────────────────────────────────────────────────────
def run_treeshap_only(name, feature_names, model, X_train_vals, X_test_vals, task, out_dir, n_points=30):
    """Fill in TreeSHAP cells of existing rho_matrix.npy; leave other cells intact."""
    rho_matrix = np.load(out_dir / "rho_matrix.npy")
    X_train_df = pd.DataFrame(X_train_vals, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_vals, columns=feature_names)
    if task == "classification":
        predict_fn = lambda X_df: model.predict_proba(X_df.values.astype(float))[:, 1]
    else:
        predict_fn = lambda X_df: model.predict(X_df.values.astype(float))
    delta = DELTA_CLASS if task == "classification" else DELTA_REG

    n_points = min(n_points, len(X_test_df))
    X_explain = X_test_df.iloc[:n_points]

    for i in range(n_points):
        if i % 5 == 0:
            print(f"  [TreeSHAP] {name} point {i}/{n_points}...")
        X_ref = X_explain.iloc[i]
        ref_pred = float(predict_fn(X_ref.to_frame().T)[0])

        for l_idx, level in enumerate(LEVELS):
            if level == "random":
                neighbors = generate_neighbors_random(X_ref, X_train_df, n=N_NEIGHBORS, random_state=i)
            else:
                neighbors = generate_neighbors_medoid(X_ref, X_train_df, n=N_NEIGHBORS, random_state=i)

            neighbors = filter_same_prediction(neighbors, predict_fn, ref_pred, delta=delta)
            if len(neighbors) == 0:
                continue

            X_ref_df = X_ref.to_frame().T
            X_combined = pd.concat([X_ref_df, neighbors], ignore_index=True)

            try:
                tree_all = explain_treeshap_adapted(model, X_combined, task)
                ref_tree = tree_all[0]
                nbr_tree = tree_all[1:]
                rho_matrix[i, l_idx, 0] = spearman_robustness(ref_tree, nbr_tree)
            except Exception as e:
                print(f"    TreeSHAP error i={i} {level}: {e}")

    # Recompute results from rho_matrix
    results = {}
    for l_idx, level in enumerate(LEVELS):
        results[level] = {}
        for e_idx, exp in enumerate(EXPLAINERS):
            vals = rho_matrix[:, l_idx, e_idx]
            valid = vals[~np.isnan(vals)]
            results[level][exp] = {
                "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
                "std": float(np.std(valid)) if len(valid) > 0 else float("nan"),
                "n": int(len(valid)),
            }

    np.save(out_dir / "rho_matrix.npy", rho_matrix)
    with open(out_dir / "results_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved updated {name} results to {out_dir}/")
    return results


def run_full_dataset(name, feature_names, model, X_train_vals, X_test_vals, task, out_dir, n_points=30):
    """Run full 2x3 matrix from scratch (for CC psi_two)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train_df = pd.DataFrame(X_train_vals, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_vals, columns=feature_names)
    if task == "classification":
        predict_fn = lambda X_df: model.predict_proba(X_df.values.astype(float))[:, 1]
    else:
        predict_fn = lambda X_df: model.predict(X_df.values.astype(float))
    delta = DELTA_CLASS if task == "classification" else DELTA_REG

    n_points = min(n_points, len(X_test_df))
    X_explain = X_test_df.iloc[:n_points]

    rho_matrix = np.full((n_points, 2, 3), np.nan)

    for i in range(n_points):
        if i % 5 == 0:
            print(f"  [{name}] point {i}/{n_points}...")
        X_ref = X_explain.iloc[i]
        ref_pred = float(predict_fn(X_ref.to_frame().T)[0])

        for l_idx, level in enumerate(LEVELS):
            if level == "random":
                neighbors = generate_neighbors_random(X_ref, X_train_df, n=N_NEIGHBORS, random_state=i)
            else:
                neighbors = generate_neighbors_medoid(X_ref, X_train_df, n=N_NEIGHBORS, random_state=i)

            neighbors = filter_same_prediction(neighbors, predict_fn, ref_pred, delta=delta)
            if len(neighbors) == 0:
                continue

            X_ref_df = X_ref.to_frame().T
            X_combined = pd.concat([X_ref_df, neighbors], ignore_index=True)

            # TreeSHAP
            try:
                tree_all = explain_treeshap_adapted(model, X_combined, task)
                rho_matrix[i, l_idx, 0] = spearman_robustness(tree_all[0], tree_all[1:])
            except Exception as e:
                print(f"    TreeSHAP error i={i} {level}: {e}")

            # KernelSHAP
            try:
                kernel_all = explain_kernelshap_adapted(model, X_train_vals, X_combined, task)
                rho_matrix[i, l_idx, 1] = spearman_robustness(kernel_all[0], kernel_all[1:])
            except Exception as e:
                print(f"    KernelSHAP error i={i} {level}: {e}")

            # LIME
            try:
                lime_all = explain_lime_adapted(model, X_train_vals, feature_names, X_combined, task)
                rho_matrix[i, l_idx, 2] = spearman_robustness(lime_all[0], lime_all[1:])
            except Exception as e:
                print(f"    LIME error i={i} {level}: {e}")

    results = {}
    for l_idx, level in enumerate(LEVELS):
        results[level] = {}
        for e_idx, exp in enumerate(EXPLAINERS):
            vals = rho_matrix[:, l_idx, e_idx]
            valid = vals[~np.isnan(vals)]
            results[level][exp] = {
                "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
                "std": float(np.std(valid)) if len(valid) > 0 else float("nan"),
                "n": int(len(valid)),
                "n_test": n_points,
            }

    print(f"\n  2x3 Spearman rho — {name}:")
    print(f"  {'':>10}" + "".join(f"{e:>14}" for e in EXPLAINERS))
    for level in LEVELS:
        row = f"  {level:>10}"
        for exp in EXPLAINERS:
            row += f"{results[level][exp]['mean']:14.4f}"
        print(row)

    np.save(out_dir / "rho_matrix.npy", rho_matrix)
    with open(out_dir / "results_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_dir}/")
    return results, n_points


def load_slack_data():
    cwd = os.getcwd()
    os.chdir(SLACK_REPO)
    try:
        from utils import Params
        from get_data import (
            get_and_preprocess_compas_data,
            get_and_preprocess_german,
            get_and_preprocess_cc,
        )
        params_path = str(SLACK_REPO / "model_configurations" / "experiment_params.json")
        params = Params(params_path)
        np.random.seed(params.seed)

        X_compas, y_compas, _ = get_and_preprocess_compas_data(params)
        X_compas["unrelated_column_one"] = np.random.choice([0, 1], size=X_compas.shape[0])
        X_compas["unrelated_column_two"] = np.random.choice([0, 1], size=X_compas.shape[0])

        np.random.seed(params.seed)
        X_german, y_german, _ = get_and_preprocess_german(params)
        X_german["unrelated_column"] = np.random.choice([0, 1], size=X_german.shape[0])

        np.random.seed(params.seed)
        X_cc, y_cc, _ = get_and_preprocess_cc(params)
        X_cc["unrelated_column_one"] = np.random.choice([0, 1], size=X_cc.shape[0])
        X_cc["unrelated_column_two"] = np.random.choice([0, 1], size=X_cc.shape[0])

    finally:
        os.chdir(cwd)

    return (X_compas, y_compas), (X_german, y_german), (X_cc, y_cc)


def main():
    t_start = time.time()
    print("Loading Slack datasets...")
    (X_compas, y_compas), (X_german, y_german), (X_cc, y_cc) = load_slack_data()

    # ── Fix COMPAS TreeSHAP ─────────────────────────────────────────────────
    print("\n=== Fixing TreeSHAP: COMPAS ===")
    with open(RESULTS_DIR / "compas" / "honest_model.pkl", "rb") as f:
        compas_model = pickle.load(f)
    with open(RESULTS_DIR / "compas" / "splits.pkl", "rb") as f:
        c_Xtr, c_Xte, c_ytr, c_yte = pickle.load(f)
    compas_results = run_treeshap_only(
        "COMPAS", list(X_compas.columns), compas_model, c_Xtr, c_Xte,
        "classification", RESULTS_DIR / "compas", N_EXPLAIN
    )

    # ── Fix German TreeSHAP ─────────────────────────────────────────────────
    print("\n=== Fixing TreeSHAP: German ===")
    with open(RESULTS_DIR / "german" / "honest_model.pkl", "rb") as f:
        german_model = pickle.load(f)
    with open(RESULTS_DIR / "german" / "splits.pkl", "rb") as f:
        g_Xtr, g_Xte, g_ytr, g_yte = pickle.load(f)
    german_results = run_treeshap_only(
        "German", list(X_german.columns), german_model, g_Xtr, g_Xte,
        "classification", RESULTS_DIR / "german", N_EXPLAIN
    )

    # ── CC psi_two full run ─────────────────────────────────────────────────
    print("\n=== CC psi_two: Full 2x3 matrix ===")
    with open(RESULTS_DIR / "cc_psi_two" / "honest_model.pkl", "rb") as f:
        cc_model = pickle.load(f)
    with open(RESULTS_DIR / "cc_psi_two" / "splits.pkl", "rb") as f:
        cc_Xtr, cc_Xte, cc_ytr, cc_yte = pickle.load(f)

    # Determine task: check if model has predict_proba
    cc_task = "classification" if hasattr(cc_model, "predict_proba") else "regression"
    print(f"  CC model type: {type(cc_model).__name__}, task={cc_task}")

    cc_results, n_cc = run_full_dataset(
        "CC_psi_two", list(X_cc.columns), cc_model, cc_Xtr, cc_Xte,
        cc_task, RESULTS_DIR / "cc_psi_two", N_EXPLAIN
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # Reload all results (in case run_treeshap_only returned partial)
    with open(RESULTS_DIR / "compas" / "results_matrix.json") as f:
        compas_results = json.load(f)
    with open(RESULTS_DIR / "german" / "results_matrix.json") as f:
        german_results = json.load(f)
    with open(RESULTS_DIR / "cc_psi_two" / "results_matrix.json") as f:
        cc_results = json.load(f)

    housing_ref = {
        "random": {"TreeSHAP": {"mean": 0.845, "std": 0.0, "n": 50},
                   "KernelSHAP": {"mean": 0.847, "std": 0.0, "n": 50},
                   "LIME": {"mean": 0.523, "std": 0.0, "n": 50}},
        "medoid":  {"TreeSHAP": {"mean": 0.515, "std": 0.0, "n": 50},
                    "KernelSHAP": {"mean": 0.531, "std": 0.0, "n": 50},
                    "LIME": {"mean": 0.473, "std": 0.0, "n": 50}},
    }

    all_datasets = [
        ("Housing", housing_ref),
        ("COMPAS",  compas_results),
        ("German",  german_results),
        ("CC psi_two", cc_results),
    ]

    print("\n" + "="*65)
    print("FULL 4-DATASET TABLE")
    print("="*65)
    print(f"{'Dataset':<14} {'Level':<8} {'TreeSHAP':>10} {'KernelSHAP':>12} {'LIME':>8}")
    print("-"*55)
    for ds_name, ds_results in all_datasets:
        for level in ["random", "medoid"]:
            ts = ds_results[level]["TreeSHAP"]["mean"]
            ks = ds_results[level]["KernelSHAP"]["mean"]
            lm = ds_results[level]["LIME"]["mean"]
            print(f"{ds_name:<14} {level:<8} {ts:10.3f} {ks:12.3f} {lm:8.3f}")
    print()

    return all_datasets, compas_results, german_results, cc_results, n_cc


if __name__ == "__main__":
    main()
