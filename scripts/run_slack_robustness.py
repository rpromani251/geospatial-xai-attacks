"""Experiment 5b — Robustness Replication on Slack Suite.

2×3 Spearman ρ matrix (Random/Medoid × TreeSHAP/KernelSHAP/LIME)
for COMPAS, German Credit, and CC psi_two datasets.

Replicates Exp 5 (housing) pattern: does Random > Medoid generalize?

Output:
  results/slack_robustness/<dataset>/honest_model.pkl
  results/slack_robustness/<dataset>/rho_matrix.npy   (50×2×3)
  results/slack_robustness/<dataset>/results_matrix.json
  docs/slack_robustness_comparison.md
  docs/figures/fig_robustness_slack_vs_housing.png
"""

from __future__ import annotations
import sys
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
SLACK_REPO = Path("/Users/rpromani/Desktop/Vault/02 Research/XAI UROP/geo-xai/fooling-lime-shap")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SLACK_REPO))

from src.evaluation.neighborhoods import (
    generate_neighbors_random,
    generate_neighbors_medoid,
    filter_same_prediction,
)
from src.evaluation.robustness import spearman_robustness

RESULTS_DIR = ROOT / "results" / "slack_robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── config ───────────────────────────────────────────────────────────────────
N_EXPLAIN = 30        # reduced to 30 to keep KernelSHAP within budget
N_NEIGHBORS = 50
N_BACKGROUND = 50
DELTA_CLASS = 0.15    # probability-space window (classification)
DELTA_REG = 0.15      # target-unit window (regression)
RANDOM_STATE = 42
LIME_SAMPLES = 500
LEVELS = ["random", "medoid"]
EXPLAINERS = ["TreeSHAP", "KernelSHAP", "LIME"]


# ── helper: load Slack data ───────────────────────────────────────────────────
def load_slack_data():
    """Load all three Slack datasets. Must chdir to SLACK_REPO first."""
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

        # COMPAS
        X_compas, y_compas, cols_compas = get_and_preprocess_compas_data(params)
        X_compas["unrelated_column_one"] = np.random.choice([0, 1], size=X_compas.shape[0])
        X_compas["unrelated_column_two"] = np.random.choice([0, 1], size=X_compas.shape[0])

        # German
        np.random.seed(params.seed)
        X_german, y_german, cols_german = get_and_preprocess_german(params)
        X_german["unrelated_column"] = np.random.choice([0, 1], size=X_german.shape[0])

        # CC
        np.random.seed(params.seed)
        X_cc, y_cc, cols_cc = get_and_preprocess_cc(params)
        X_cc["unrelated_column_one"] = np.random.choice([0, 1], size=X_cc.shape[0])
        X_cc["unrelated_column_two"] = np.random.choice([0, 1], size=X_cc.shape[0])

    finally:
        os.chdir(cwd)

    return (X_compas, y_compas), (X_german, y_german), (X_cc, y_cc)


# ── helper: train honest model ────────────────────────────────────────────────
def train_honest_model(X_df, y, task, random_state=RANDOM_STATE):
    """Train an RF classifier (classification) or RF regressor (regression)."""
    X_vals = X_df.values.astype(float)
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    Xtr, Xte, ytr, yte = train_test_split(
        X_vals, y, test_size=0.2, random_state=random_state
    )
    model.fit(Xtr, ytr)
    return model, Xtr, Xte, ytr, yte, X_df.columns.tolist()


# ── helper: make predict_proba-based predict_fn for classification ─────────────
def make_predict_fn(model, task):
    if task == "classification":
        # predict returns class label; for delta filter use class-1 probability
        return lambda X_df: model.predict_proba(X_df.values.astype(float))[:, 1]
    else:
        return lambda X_df: model.predict(X_df.values.astype(float))


# ── LIME wrapper adapted for classification (class-1 proba output) ─────────────
def explain_lime_adapted(model, X_train_vals, feature_names, X_test_df, task,
                          num_samples=LIME_SAMPLES, random_state=RANDOM_STATE):
    n_features = len(feature_names)
    if task == "classification":
        explainer = LimeTabularExplainer(
            X_train_vals,
            feature_names=feature_names,
            mode="classification",
            random_state=random_state,
        )
        def predict_fn_lime(x):
            p = model.predict_proba(x.astype(float))
            return p  # shape (n, 2)

        lime_values = np.zeros((len(X_test_df), n_features))
        for i, (_, row) in enumerate(X_test_df.iterrows()):
            exp = explainer.explain_instance(
                row.values.astype(float),
                predict_fn_lime,
                num_features=n_features,
                num_samples=num_samples,
                labels=(1,),
            )
            fw = dict(exp.as_map()[1])
            for j in range(n_features):
                lime_values[i, j] = fw.get(j, 0.0)
    else:
        # regression
        explainer = LimeTabularExplainer(
            X_train_vals,
            feature_names=feature_names,
            mode="regression",
            random_state=random_state,
        )
        def predict_fn_reg(x):
            return model.predict(x.astype(float))

        lime_values = np.zeros((len(X_test_df), n_features))
        for i, (_, row) in enumerate(X_test_df.iterrows()):
            exp = explainer.explain_instance(
                row.values.astype(float),
                predict_fn_reg,
                num_features=n_features,
                num_samples=num_samples,
            )
            fw = dict(exp.as_map()[1])
            for j in range(n_features):
                lime_values[i, j] = fw.get(j, 0.0)

    return lime_values


# ── helper: TreeSHAP (works for RF classification too) ────────────────────────
def explain_treeshap_adapted(model, X_combined_df, task):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_combined_df.values.astype(float))
    sv = np.array(sv)
    if task == "classification":
        # SHAP returns (n_points, n_features, n_classes) for RF classifiers
        # OR a list of (n_points, n_features) arrays — handle both
        if sv.ndim == 3:
            # shape (n_points, n_features, n_classes) — take class-1 slice
            sv = sv[:, :, 1]
        elif sv.ndim == 2 and sv.shape[0] == 2:
            # legacy list-of-arrays flattened to (2, n_points, n_features)? unlikely but guard
            sv = sv[1]
    return sv


# ── helper: KernelSHAP ────────────────────────────────────────────────────────
def explain_kernelshap_adapted(model, X_train_vals, X_combined_df, task, n_background=N_BACKGROUND):
    bg = shap.sample(pd.DataFrame(X_train_vals), n_background)
    if task == "classification":
        predict_fn = lambda x: model.predict_proba(x.astype(float))[:, 1]
    else:
        predict_fn = lambda x: model.predict(x.astype(float))
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_combined_df.values.astype(float))
    return np.array(sv)


# ── main per-dataset routine ──────────────────────────────────────────────────
def run_dataset(name, X_df, y, task, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | task={task} | shape={X_df.shape}")
    print('='*60)

    feature_names = list(X_df.columns)

    # 1. Train or load model
    model_path = out_dir / "honest_model.pkl"
    split_path = out_dir / "splits.pkl"
    if model_path.exists() and split_path.exists():
        print("Loading saved model...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
        X_train_vals, X_test_vals, y_train, y_test = splits
        X_train_df = pd.DataFrame(X_train_vals, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_vals, columns=feature_names)
    else:
        print("Training fresh RF model...")
        model, X_train_vals, X_test_vals, y_train, y_test, _ = train_honest_model(
            X_df, y, task
        )
        X_train_df = pd.DataFrame(X_train_vals, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_vals, columns=feature_names)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(split_path, "wb") as f:
            pickle.dump((X_train_vals, X_test_vals, y_train, y_test), f)
        print(f"Saved model to {model_path}")

    # Quick performance check
    if task == "classification":
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, model.predict(X_test_vals.astype(float)))
        print(f"Test accuracy: {acc:.3f}")
    else:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, model.predict(X_test_vals.astype(float)))
        print(f"Test R²: {r2:.3f}")

    # 2. Setup predict_fn for filter
    predict_fn = make_predict_fn(model, task)
    delta = DELTA_CLASS if task == "classification" else DELTA_REG

    # 3. Select test points
    n_points = min(N_EXPLAIN, len(X_test_df))
    X_explain = X_test_df.iloc[:n_points]

    # rho_matrix: shape (n_points, 2, 3) — 2 levels × 3 explainers
    rho_matrix = np.full((n_points, 2, 3), np.nan)

    for i in range(n_points):
        if i % 5 == 0:
            print(f"  Point {i}/{n_points}...")

        X_ref = X_explain.iloc[i]
        ref_pred = float(predict_fn(X_ref.to_frame().T)[0])

        for l_idx, level in enumerate(LEVELS):
            # Generate neighbors
            if level == "random":
                neighbors = generate_neighbors_random(
                    X_ref, X_train_df, n=N_NEIGHBORS, random_state=i
                )
            else:  # medoid
                neighbors = generate_neighbors_medoid(
                    X_ref, X_train_df, n=N_NEIGHBORS, random_state=i
                )

            # Filter
            neighbors = filter_same_prediction(neighbors, predict_fn, ref_pred, delta=delta)
            if len(neighbors) == 0:
                continue

            X_ref_df = X_ref.to_frame().T
            X_combined = pd.concat([X_ref_df, neighbors], ignore_index=True)

            # TreeSHAP
            try:
                tree_all = explain_treeshap_adapted(model, X_combined, task)
                ref_tree = tree_all[0]
                nbr_tree = tree_all[1:]
                rho_matrix[i, l_idx, 0] = spearman_robustness(ref_tree, nbr_tree)
            except Exception as e:
                print(f"    TreeSHAP error at i={i}, level={level}: {e}")

            # KernelSHAP
            try:
                kernel_all = explain_kernelshap_adapted(
                    model, X_train_vals, X_combined, task
                )
                ref_kernel = kernel_all[0]
                nbr_kernel = kernel_all[1:]
                rho_matrix[i, l_idx, 1] = spearman_robustness(ref_kernel, nbr_kernel)
            except Exception as e:
                print(f"    KernelSHAP error at i={i}, level={level}: {e}")

            # LIME
            try:
                lime_all = explain_lime_adapted(
                    model, X_train_vals, feature_names, X_combined, task
                )
                lime_ref = lime_all[0]
                lime_nbrs = lime_all[1:]
                rho_matrix[i, l_idx, 2] = spearman_robustness(lime_ref, lime_nbrs)
            except Exception as e:
                print(f"    LIME error at i={i}, level={level}: {e}")

    # 4. Aggregate results
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

    # Print table
    print(f"\n  2×3 Spearman ρ matrix — {name} (n_points={n_points}):")
    print(f"  {'':>10}" + "".join(f"{e:>14}" for e in EXPLAINERS))
    print("  " + "-" * (10 + 14 * len(EXPLAINERS)))
    for level in LEVELS:
        row = f"  {level:>10}"
        for exp in EXPLAINERS:
            v = results[level][exp]["mean"]
            row += f"{v:14.4f}"
        print(row)

    # 5. Save
    np.save(out_dir / "rho_matrix.npy", rho_matrix)
    with open(out_dir / "results_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_dir}/")

    return results, n_points


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Load Slack datasets
    print("Loading Slack datasets from fooling-lime-shap repo...")
    (X_compas, y_compas), (X_german, y_german), (X_cc, y_cc) = load_slack_data()

    # COMPAS: classification (y is 0/1)
    compas_results, n_compas = run_dataset(
        "COMPAS", X_compas, y_compas, "classification",
        RESULTS_DIR / "compas"
    )

    # German: classification (y is 0/1)
    german_results, n_german = run_dataset(
        "German", X_german, y_german, "classification",
        RESULTS_DIR / "german"
    )

    # CC: Slack uses classification (binary label: > 50th percentile).
    # Per task spec: treat as classification. Use predict_proba delta=0.15
    cc_results, n_cc = run_dataset(
        "CC_psi_two", X_cc, y_cc, "classification",
        RESULTS_DIR / "cc_psi_two"
    )

    # Housing reference values (from Exp 5)
    housing_ref = {
        "random": {"TreeSHAP": 0.845, "KernelSHAP": 0.847, "LIME": 0.523},
        "medoid": {"TreeSHAP": 0.515, "KernelSHAP": 0.531, "LIME": 0.473},
    }

    # ── Combined table ──────────────────────────────────────────────────────
    all_datasets = [
        ("Housing", housing_ref, "Exp 5 reference"),
        ("COMPAS", compas_results, f"n={n_compas}"),
        ("German", german_results, f"n={n_german}"),
        ("CC psi_two", cc_results, f"n={n_cc}"),
    ]

    # Build comparison doc
    lines = ["# Experiment 5b — Slack Robustness Comparison\n"]
    lines.append("## Overview\n")
    lines.append(
        "This document reports Experiment 5b: replication of the Exp 5 housing "
        "2×3 Spearman ρ robustness matrix (Random / Medoid × TreeSHAP / KernelSHAP / LIME) "
        "across three Slack datasets — COMPAS, German Credit, and Communities-and-Crime "
        "(CC psi_two). Each honest model is a Random Forest (100 trees, random_state=42) "
        "trained on an 80/20 split of Slack's standard preprocessing. "
        f"Due to KernelSHAP budget constraints, n={N_EXPLAIN} test points per dataset "
        "(vs 50 in housing). Neighborhood size: 50 per level. "
        "Filter threshold δ=0.15 on class-1 probability (all three datasets are binary classification).\n"
    )

    lines.append("## Combined Table\n")
    lines.append("| Dataset | Level | TreeSHAP | KernelSHAP | LIME | Δ TreeSHAP | Δ KernelSHAP | Δ LIME |")
    lines.append("|---|---|---|---|---|---|---|---|")

    dataset_deltas = {}
    for ds_name, ds_results, note in all_datasets:
        for level in ["random", "medoid"]:
            if level == "random":
                ts = ds_results["random"]["TreeSHAP"] if isinstance(ds_results["random"]["TreeSHAP"], float) else ds_results["random"]["TreeSHAP"]["mean"]
                ks = ds_results["random"]["KernelSHAP"] if isinstance(ds_results["random"]["KernelSHAP"], float) else ds_results["random"]["KernelSHAP"]["mean"]
                lm = ds_results["random"]["LIME"] if isinstance(ds_results["random"]["LIME"], float) else ds_results["random"]["LIME"]["mean"]
                ts_med = ds_results["medoid"]["TreeSHAP"] if isinstance(ds_results["medoid"]["TreeSHAP"], float) else ds_results["medoid"]["TreeSHAP"]["mean"]
                ks_med = ds_results["medoid"]["KernelSHAP"] if isinstance(ds_results["medoid"]["KernelSHAP"], float) else ds_results["medoid"]["KernelSHAP"]["mean"]
                lm_med = ds_results["medoid"]["LIME"] if isinstance(ds_results["medoid"]["LIME"], float) else ds_results["medoid"]["LIME"]["mean"]
                dt = ts - ts_med
                dk = ks - ks_med
                dl = lm - lm_med
                dataset_deltas[ds_name] = (dt, dk, dl)
                lines.append(f"| {ds_name} ({note}) | Random | {ts:.3f} | {ks:.3f} | {lm:.3f} | {dt:+.3f} | {dk:+.3f} | {dl:+.3f} |")
            else:
                ts_med = ds_results["medoid"]["TreeSHAP"] if isinstance(ds_results["medoid"]["TreeSHAP"], float) else ds_results["medoid"]["TreeSHAP"]["mean"]
                ks_med = ds_results["medoid"]["KernelSHAP"] if isinstance(ds_results["medoid"]["KernelSHAP"], float) else ds_results["medoid"]["KernelSHAP"]["mean"]
                lm_med = ds_results["medoid"]["LIME"] if isinstance(ds_results["medoid"]["LIME"], float) else ds_results["medoid"]["LIME"]["mean"]
                lines.append(f"| {ds_name} | Medoid | {ts_med:.3f} | {ks_med:.3f} | {lm_med:.3f} | — | — | — |")

    lines.append("")
    lines.append("_Δ = Random − Medoid. Positive Δ = Random neighborhood yields higher explanation stability._\n")

    # ── Cross-dataset interpretation ────────────────────────────────────────
    lines.append("## Cross-Dataset Pattern\n")

    # determine if random > medoid for each dataset × explainer
    pattern_holds = {}
    for ds_name, (dt, dk, dl) in dataset_deltas.items():
        holds_ts = dt > 0
        holds_ks = dk > 0
        holds_lm = dl > 0
        all_hold = holds_ts and holds_ks and holds_lm
        pattern_holds[ds_name] = {
            "TreeSHAP": holds_ts,
            "KernelSHAP": holds_ks,
            "LIME": holds_lm,
            "all": all_hold,
        }

    replicates_all = all(v["all"] for v in pattern_holds.values())
    lines.append(
        f"**Does Random > Medoid replicate across all datasets?** "
        f"{'Yes — the pattern holds uniformly.' if replicates_all else 'Partially — see per-dataset breakdown below.'}\n"
    )

    for ds_name, ph in pattern_holds.items():
        dt, dk, dl = dataset_deltas[ds_name]
        lines.append(f"- **{ds_name}**: TreeSHAP Δ={dt:+.3f} ({'✓' if ph['TreeSHAP'] else '✗'}), "
                     f"KernelSHAP Δ={dk:+.3f} ({'✓' if ph['KernelSHAP'] else '✗'}), "
                     f"LIME Δ={dl:+.3f} ({'✓' if ph['LIME'] else '✗'})")

    lines.append("")
    # Avg deltas
    all_dt = [v[0] for v in dataset_deltas.values()]
    all_dk = [v[1] for v in dataset_deltas.values()]
    all_dl = [v[2] for v in dataset_deltas.values()]
    lines.append(
        f"**Mean Δ across all 4 datasets**: "
        f"TreeSHAP={np.mean(all_dt):+.3f}, "
        f"KernelSHAP={np.mean(all_dk):+.3f}, "
        f"LIME={np.mean(all_dl):+.3f}\n"
    )

    lines.append(
        "**Interpretation:** The systematic advantage of random (Gaussian) neighborhoods over "
        "on-manifold (medoid) neighborhoods for explanation stability is hypothesized to arise "
        "from tree-model split boundaries. Medoid neighbors, by construction, lie on the data "
        "manifold near the reference point — but this places them in regions where the tree's "
        "learned split structure can sharply redirect feature attributions. Random (Gaussian) "
        "neighbors perturb each feature independently by a small σ, which tends to stay in "
        "the same high-density region without crossing sharp tree boundaries. "
        "If this pattern replicates to the Slack suite (lower-dimensional tabular classifiers "
        "with fewer spatial co-variation structures), it strengthens the hypothesis that the "
        "finding is a general property of tree models rather than a Seattle housing artifact.\n"
    )

    lines.append("## Implications for Exp 5 Headline\n")
    lines.append(
        "If the pattern replicates, the Exp 5 finding — "
        '"random Gaussian neighborhoods yield more stable explanations than on-manifold medoid neighbors" '
        "— should be re-framed as a general tree-model finding rather than a geospatial one. "
        "The spatial Exp 5 level adds value as an existence proof that spatial coherence "
        "does not rescue explanation stability, but the core result is dataset-agnostic.\n"
    )

    doc_path = DOCS_DIR / "slack_robustness_comparison.md"
    with open(doc_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nComparison doc saved: {doc_path}")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ds_labels = ["Housing", "COMPAS", "German", "CC psi_two"]
    colors = {"random": "#4C72B0", "medoid": "#DD8452"}
    x = np.arange(len(ds_labels))
    width = 0.35

    all_ds_results_ordered = [
        ("Housing", housing_ref),
        ("COMPAS", compas_results),
        ("German", german_results),
        ("CC psi_two", cc_results),
    ]

    for e_idx, exp in enumerate(EXPLAINERS):
        ax = axes[e_idx]
        rand_vals = []
        med_vals = []
        for ds_name, ds_results in all_ds_results_ordered:
            rv = ds_results["random"][exp] if isinstance(ds_results["random"][exp], float) else ds_results["random"][exp]["mean"]
            mv = ds_results["medoid"][exp] if isinstance(ds_results["medoid"][exp], float) else ds_results["medoid"][exp]["mean"]
            rand_vals.append(rv)
            med_vals.append(mv)

        bars_r = ax.bar(x - width/2, rand_vals, width, label="Random", color=colors["random"], alpha=0.85)
        bars_m = ax.bar(x + width/2, med_vals, width, label="Medoid", color=colors["medoid"], alpha=0.85)

        ax.set_title(exp, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean Spearman ρ" if e_idx == 0 else "", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # annotate housing reference values
        ax.annotate(
            f"Housing\nRef",
            xy=(x[0], max(rand_vals[0], med_vals[0]) + 0.03),
            ha="center", fontsize=7, color="gray"
        )

    fig.suptitle(
        "Experiment 5b: Random vs Medoid Neighborhood Robustness\n"
        "Housing (reference) + COMPAS + German + CC psi_two",
        fontsize=13,
    )
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_robustness_slack_vs_housing.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path}")

    # ── Final console summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 5b SUMMARY")
    print("="*60)
    print(f"{'Dataset':<14} {'Level':<8} {'TreeSHAP':>10} {'KernelSHAP':>12} {'LIME':>8}")
    print("-"*54)
    for ds_name, ds_results in all_ds_results_ordered:
        for level in ["random", "medoid"]:
            ts = ds_results[level]["TreeSHAP"] if isinstance(ds_results[level]["TreeSHAP"], float) else ds_results[level]["TreeSHAP"]["mean"]
            ks = ds_results[level]["KernelSHAP"] if isinstance(ds_results[level]["KernelSHAP"], float) else ds_results[level]["KernelSHAP"]["mean"]
            lm = ds_results[level]["LIME"] if isinstance(ds_results[level]["LIME"], float) else ds_results[level]["LIME"]["mean"]
            print(f"{ds_name:<14} {level:<8} {ts:10.3f} {ks:12.3f} {lm:8.3f}")
    print()


if __name__ == "__main__":
    main()
