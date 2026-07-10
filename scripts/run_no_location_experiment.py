"""Train XGBoost without UTM features, explain with TreeSHAP/KernelSHAP/LIME."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.models.train import train_xgboost
from src.explainers.shap_runner import explain_shap, explain_treeshap
from src.explainers.lime_runner import explain_lime
from src.utils.io import save_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    # 1. Load config
    config_path = CONFIGS_DIR / "no_location.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Load feature data (8 non-spatial features)
    X, y = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=config["features"],
        target_column=config["target"],
    )
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Train
    model = train_xgboost(X_train, y_train, config["model"])
    print("Model trained.")

    # 5. Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(f"Results: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

    # 6. Subset for explanation
    n_explain = 200
    X_explain = X_test.iloc[:n_explain]
    print(f"Explaining {len(X_explain)} test points, {X_explain.shape[1]} features")

    # 7. Run explainers (no GeoShapley — model has no spatial coordinates)
    print("Running TreeSHAP...")
    tree_shap = explain_treeshap(model, X_explain)
    print(f"  TreeSHAP values shape: {tree_shap.shape}")

    print("Running KernelSHAP...")
    kernel_shap = explain_shap(model, X_train, X_explain, n_background=100)
    print(f"  KernelSHAP values shape: {kernel_shap.shape}")

    print("Running LIME...")
    lime_vals = explain_lime(model, X_train, X_explain, num_samples=500, random_state=42)
    print(f"  LIME values shape: {lime_vals.shape}")

    # 8. Save outputs
    out_dir = RESULTS_DIR / "no_location"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, out_dir / "model.pkl")
    save_metrics(metrics, out_dir / "metrics.json")
    np.save(out_dir / "treeshap_values.npy", tree_shap)
    np.save(out_dir / "kernelshap_values.npy", kernel_shap)
    np.save(out_dir / "lime_values.npy", lime_vals)
    np.save(out_dir / "feature_names.npy", np.array(X_explain.columns.tolist()))
    np.save(out_dir / "test_indices.npy", X_explain.index.values)
    print(f"Saved to {out_dir}/")

    # 9. Ranking table
    feature_names = X_explain.columns.tolist()
    mean_tree = np.abs(tree_shap).mean(axis=0)
    mean_kernel = np.abs(kernel_shap).mean(axis=0)
    mean_lime = np.abs(lime_vals).mean(axis=0)

    sort_order = np.argsort(mean_tree)[::-1]

    header = f"{'feature':>15} {'TreeSHAP':>10} {'KernelSHAP':>12} {'LIME':>10}"
    print("\nMean absolute feature importance across explainers:")
    print(header)
    print("-" * len(header))
    for idx in sort_order:
        name = feature_names[idx]
        t = mean_tree[idx]
        k = mean_kernel[idx]
        l = mean_lime[idx]
        print(f"{name:>15} {t:>10.4f} {k:>12.4f} {l:>10.4f}")


if __name__ == "__main__":
    main()
