"""Generate KernelSHAP, LIME, and GeoSHAP explanations for baseline model."""

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.explainers.geoshap_runner import explain_geoshap
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    # 1. Load configs
    with open(CONFIGS_DIR / "explanations.yaml") as f:
        config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_config = yaml.safe_load(f)

    # 2. Load model
    model = load_model(ROOT / config["model_path"])
    print("Model loaded.")

    # 3. Load data with same split as baseline; load UTM coords separately for GeoSHAP
    X, y = load_housing(
        dataset=baseline_config["data"]["dataset"],
        feature_columns=baseline_config["features"],
        target_column=baseline_config["target"],
    )
    coords, _ = load_housing(
        dataset=baseline_config["data"]["dataset"],
        feature_columns=["UTM_X", "UTM_Y"],
        target_column=baseline_config["target"],
    )
    X_train, X_test, _, _, coords_train, coords_test = train_test_split(
        X, y, coords,
        test_size=baseline_config["data"]["test_size"],
        random_state=baseline_config["data"]["random_state"],
    )

    # 4. Subset test points
    n_explain = config["n_explain"]
    X_explain = X_test.iloc[:n_explain]
    coords_explain = coords_test.iloc[:n_explain]
    print(f"Explaining {len(X_explain)} test points, {X_explain.shape[1]} features")

    # 5. KernelSHAP
    print("Running KernelSHAP...")
    shap_values = explain_shap(
        model, X_train, X_explain,
        n_background=config["shap"]["n_background"],
    )
    print(f"  SHAP values shape: {shap_values.shape}")

    # 6. LIME
    print("Running LIME...")
    lime_values = explain_lime(
        model, X_train, X_explain,
        num_samples=config["lime"]["num_samples"],
        random_state=config["lime"]["random_state"],
    )
    print(f"  LIME values shape: {lime_values.shape}")

    # 7. GeoSHAP — honest baseline using k=100 geographic nearest neighbors as background
    print("Running GeoSHAP (k=100 geographic background)...")
    geoshap_values = explain_geoshap(
        model, X_train, coords_train.values,
        X_explain, coords_explain.values,
        k=100,
    )
    print(f"  GeoSHAP values shape: {geoshap_values.shape}")

    # 8. Save
    out_dir = RESULTS_DIR / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "shap_values.npy", shap_values)
    np.save(out_dir / "lime_values.npy", lime_values)
    np.save(out_dir / "geoshap_values.npy", geoshap_values)
    np.save(out_dir / "test_indices.npy", X_explain.index.values)
    np.save(out_dir / "feature_names.npy", np.array(X_explain.columns.tolist()))
    print(f"Saved to {out_dir}/")

    # 9. Summary
    feature_names = X_explain.columns.tolist()
    print("\nTop features by mean |SHAP|:")
    mean_shap = np.abs(shap_values).mean(axis=0)
    for idx in np.argsort(mean_shap)[::-1]:
        print(f"  {feature_names[idx]:>15}: {mean_shap[idx]:.4f}")

    print("\nTop features by mean |LIME|:")
    mean_lime = np.abs(lime_values).mean(axis=0)
    for idx in np.argsort(mean_lime)[::-1]:
        print(f"  {feature_names[idx]:>15}: {mean_lime[idx]:.4f}")

    print("\nTop features by mean |GeoSHAP|:")
    mean_geo = np.abs(geoshap_values).mean(axis=0)
    for idx in np.argsort(mean_geo)[::-1]:
        print(f"  {feature_names[idx]:>15}: {mean_geo[idx]:.4f}")


if __name__ == "__main__":
    main()
