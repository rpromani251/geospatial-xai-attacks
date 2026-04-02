"""Run the baseline XGBoost experiment end-to-end."""

import sys
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.models.train import train_xgboost
from src.utils.io import save_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    # 1. Load config
    config_path = CONFIGS_DIR / "baseline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Load data
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

    # 6. Save outputs
    out_dir = RESULTS_DIR / "baseline"
    save_model(model, out_dir / "model.pkl")
    save_metrics(metrics, out_dir / "metrics.json")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
