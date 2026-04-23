"""Generate honest + adversarial SHAP + LIME explanations for a single
biased feature, and save the matrices for downstream SBM analysis.

Produces, under results/scaffold_<feature>/:
    honest/
        shap_values.npy        (n_explain, d)
        lime_values.npy        (n_explain, d)
        test_indices.npy       (n_explain,)
        feature_names.npy      (d,)
    adv/
        adv_shap_values.npy    (n_explain, d)
        adv_lime_values.npy    (n_explain, d)

where d = len(baseline_features) + 1 if the bias feature is off-baseline, else
d = len(baseline_features). Column order is fixed and returned via
feature_names.npy.

Run:
    python3 scripts/generate_scaffold_explanations.py --bias-feature sqft_lot
    python3 scripts/generate_scaffold_explanations.py --bias-feature view
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bias-feature", type=str, required=True)
    p.add_argument("--n-explain", type=int, default=200)
    p.add_argument("--out-dir", type=str, default=None,
                   help="override output dir; default is results/scaffold_<feature>/")
    p.add_argument("--penalty", type=float, default=None,
                   help="override configs/scaffold.yaml biased_model.penalty")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)

    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        base_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_features = yaml.safe_load(f)["features"]

    X_full, y = load_housing(dataset="demographics", feature_columns=None, target_column="log_price")
    X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])

    bias_feature = args.bias_feature
    if bias_feature not in X_full.columns:
        raise ValueError(f"{bias_feature} not in {list(X_full.columns)}")

    # Focused column set (matches run_feature_sweep.py logic)
    if bias_feature in baseline_features:
        sweep_cols = list(baseline_features)
    else:
        sweep_cols = list(baseline_features) + [bias_feature]
    df_sweep = X_full[sweep_cols]
    feature_names = list(df_sweep.columns)
    bias_col_idx = feature_names.index(bias_feature)
    base_feature_indices = [feature_names.index(f) for f in baseline_features]

    threshold = float(np.percentile(df_sweep[bias_feature].values, 75))
    penalty = args.penalty if args.penalty is not None else base_config["biased_model"]["penalty"]
    print(f"bias={bias_feature}  threshold(P75)={threshold:.4f}  penalty={penalty}")
    print(f"feature columns ({len(feature_names)}): {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=base_config["data"]["test_size"],
        random_state=base_config["data"]["random_state"],
    )
    X_train_np = X_train.values
    X_test_np = X_test.values

    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")

    # Scaffold
    f_biased = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=bias_col_idx,
        threshold=threshold,
        penalty=penalty,
    )
    safe_indices = [i for i, f in enumerate(feature_names)
                    if f not in {bias_feature, "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, base_config["innocuous_model"])
    print(f"Innocuous model fitted.")

    print("Training SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, base_config["shap_ood"])
    print(f"  SHAP OOD accuracy: {shap_scaffold.ood_diagnostics['accuracy']:.4f}")

    print("Training LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(
        f_biased, psi, X_train_np, base_config["lime_ood"])
    print(f"  LIME OOD accuracy: {lime_scaffold.ood_diagnostics['accuracy']:.4f}")

    # Explain first n_explain test instances
    n = min(args.n_explain, len(X_test))
    X_explain = X_test.iloc[:n]
    X_train_df = pd.DataFrame(X_train_np, columns=feature_names)
    X_explain_df = pd.DataFrame(X_explain.values, columns=feature_names)
    test_indices = X_test.index.values[:n]  # original df indices — BUT see note below

    # ── NOTE: the existing results/explanations/test_indices.npy uses the
    # ORIGINAL demographics-df row indices, selected from X_test. We match that
    # by taking the pandas `index` array directly (not reset_index), since
    # load_housing uses the demographics-df rows verbatim.

    # Honest explanations on the 10 baseline features, padded to d columns
    import shap as shap_lib
    explainer_shap = shap_lib.TreeExplainer(honest_model)
    print(f"Honest SHAP on {n} pts...")
    honest_shap_vals = explainer_shap.shap_values(X_explain_df[baseline_features])
    honest_shap_full = np.zeros((n, len(feature_names)))
    baseline_mask = [feature_names.index(f) for f in baseline_features]
    honest_shap_full[:, baseline_mask] = honest_shap_vals

    print(f"Honest LIME on {n} pts (using honest model)...")
    # Wrap the honest model in a scaffold where both branches = honest, to reuse explain_lime
    from src.attacks.scaffold import AdversarialScaffold as _AS
    from src.attacks.biased_model import InnocuousModel as _IM
    # Simpler: call explain_lime directly with honest_model as the "predictor".
    # But explain_lime expects an object with a `predict` method. The XGBoost
    # booster has it, so we pass it straight.
    honest_lime_vals = explain_lime(
        honest_model, X_train_df[baseline_features],
        X_explain_df[baseline_features],
        num_samples=base_config["explanation"]["lime_num_samples"],
        random_state=base_config["explanation"]["lime_random_state"],
    )
    honest_lime_full = np.zeros((n, len(feature_names)))
    honest_lime_full[:, baseline_mask] = honest_lime_vals

    print(f"Adversarial SHAP on {n} pts...")
    adv_shap = explain_shap(
        shap_scaffold, X_train_df, X_explain_df,
        n_background=base_config["explanation"]["shap_n_background"],
    )

    print(f"Adversarial LIME on {n} pts...")
    adv_lime = explain_lime(
        lime_scaffold, X_train_df, X_explain_df,
        num_samples=base_config["explanation"]["lime_num_samples"],
        random_state=base_config["explanation"]["lime_random_state"],
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_root = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"scaffold_{bias_feature}"
    (out_root / "honest").mkdir(parents=True, exist_ok=True)
    (out_root / "adv").mkdir(parents=True, exist_ok=True)

    np.save(out_root / "honest" / "shap_values.npy", honest_shap_full)
    np.save(out_root / "honest" / "lime_values.npy", honest_lime_full)
    np.save(out_root / "honest" / "test_indices.npy", test_indices)
    np.save(out_root / "honest" / "feature_names.npy", np.array(feature_names, dtype=object))

    np.save(out_root / "adv" / "adv_shap_values.npy", adv_shap)
    np.save(out_root / "adv" / "adv_lime_values.npy", adv_lime)

    print(f"\nSaved honest + adversarial matrices under {out_root}")
    print(f"  honest shape: {honest_shap_full.shape}")
    print(f"  adv    shape: {adv_shap.shape}")


if __name__ == "__main__":
    main()
