"""One-off: compute honest GeoShapley for the 200 explanation test points.

Standalone helper — does NOT touch existing shap_values.npy / lime_values.npy.
Saves GeoShapley components next to them.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.explainers.geoshap_runner import explain_geoshapley, geoshapley_to_flat, save_geoshapley
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    with open(CONFIGS_DIR / "explanations.yaml") as f:
        config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_config = yaml.safe_load(f)

    model = load_model(ROOT / config["model_path"])

    X, y = load_housing(
        dataset=baseline_config["data"]["dataset"],
        feature_columns=baseline_config["features"],
        target_column=baseline_config["target"],
    )
    X_train, X_test, _, _ = train_test_split(
        X, y,
        test_size=baseline_config["data"]["test_size"],
        random_state=baseline_config["data"]["random_state"],
    )

    n_explain = config["n_explain"]
    X_explain = X_test.iloc[:n_explain]
    print(f"Explaining {len(X_explain)} points with GeoShapley...")

    results = explain_geoshapley(model, X_train, X_explain)
    geo_flat = geoshapley_to_flat(results)
    print(f"GeoShapley primary shape: {results.primary.shape}")

    out_dir = RESULTS_DIR / "explanations"
    save_geoshapley(results, out_dir)
    print(f"Saved to {out_dir}/")

    non_geo_names = [c for c in X_explain.columns if c not in ("UTM_X", "UTM_Y")]
    geo_names = non_geo_names + ["geo_effect"]
    mean_geo = np.abs(geo_flat).mean(axis=0)
    print("\nmean |GeoShapley| per feature:")
    for name, v in zip(geo_names, mean_geo):
        print(f"  {name:>15}: {v:.4f}")


if __name__ == "__main__":
    main()
