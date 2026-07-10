"""Compute honest 11-feature baseline attributions for off-baseline trigger experiments.

Covers: Exp 3 (pct_black_latino), Exp 10 (in_floodplain), Exp 10b (in_flood_500yr),
        Exp 10c (in_floodplain / parcel 59k), Exp 10d (in_geologic_hazard),
        Exp 11 (in_low_score_zone / school-zone subset).

Outputs per experiment under results/honest_11feat/<exp>/:
  baseline_model.pkl, baseline_metrics.json,
  honest_shap_11feat.npy, honest_lime_11feat.npy,
  honest_attribution_means.json
  (Exp 3 only: geoshapley_primary.npy, geoshapley_geo.npy, geoshapley_interaction.npy)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.models.train import train_xgboost
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.explainers.geoshap_runner import explain_geoshapley, geoshapley_to_flat, save_geoshapley
from src.utils.io import save_model, save_metrics

RESULTS = ROOT / "results" / "honest_11feat"

MODEL_CFG = dict(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
SPLIT_CFG = dict(test_size=0.2, random_state=42)
N_EXPLAIN = 200
N_BG = 100          # default SHAP background
N_BG_PARCEL = 50    # reduced for 59k parcel dataset

# ── Experiment definitions ────────────────────────────────────────────────────
# UTM_X, UTM_Y are always last (required by GeoShapley)

EXPERIMENTS = [
    {
        "name": "exp3",
        "dataset": "demographics",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "pct_black_latino", "UTM_X", "UTM_Y"],
        "trigger": "pct_black_latino",
        "geoshap": True,
        "n_background": N_BG,
        "label": "Exp 3",
    },
    {
        "name": "exp10",
        "dataset": "flood_zone",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "in_floodplain", "UTM_X", "UTM_Y"],
        "trigger": "in_floodplain",
        "geoshap": False,
        "n_background": N_BG,
        "label": "Exp 10",
    },
    {
        "name": "exp10b",
        "dataset": "flood_500yr",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "in_flood_500yr", "UTM_X", "UTM_Y"],
        "trigger": "in_flood_500yr",
        "geoshap": False,
        "n_background": N_BG,
        "label": "Exp 10b",
    },
    {
        "name": "exp10c",
        "dataset": "parcel_flood",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "in_floodplain", "UTM_X", "UTM_Y"],
        "trigger": "in_floodplain",
        "geoshap": False,
        "n_background": N_BG_PARCEL,
        "label": "Exp 10c",
    },
    {
        "name": "exp10d",
        "dataset": "geologic_hazard",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "in_geologic_hazard", "UTM_X", "UTM_Y"],
        "trigger": "in_geologic_hazard",
        "geoshap": False,
        "n_background": N_BG,
        "label": "Exp 10d",
    },
    {
        "name": "exp11",
        "dataset": "school_zone",
        "features": ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
                     "waterfront", "view", "age", "in_low_score_zone", "UTM_X", "UTM_Y"],
        "trigger": "in_low_score_zone",
        "geoshap": False,
        "n_background": N_BG,
        "label": "Exp 11",
    },
]


def run_experiment(exp: dict) -> dict:
    t0 = time.time()
    name = exp["name"]
    label = exp["label"]
    trigger = exp["trigger"]
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label} — {trigger}  ({exp['dataset']})")
    print(f"{'='*60}")

    # 1. Load data
    X, y = load_housing(dataset=exp["dataset"], feature_columns=exp["features"])
    feature_names = list(X.columns)
    print(f"  Data loaded: {X.shape}")

    # 2. Train/test split (same seed as all scaffolds)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **SPLIT_CFG
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Retrain 11-feature honest XGBoost
    model = train_xgboost(X_train, y_train, MODEL_CFG)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    metrics = {
        "r2": r2,
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
    }
    print(f"  11-feat baseline R² = {r2:.4f}")
    save_model(model, out_dir / "baseline_model.pkl")
    save_metrics(metrics, out_dir / "baseline_metrics.json")

    # 4. Subset 200 test points (first 200, matching scaffold convention)
    X_explain = X_test.iloc[:N_EXPLAIN]
    print(f"  Explaining {len(X_explain)} test points...")

    # 5. KernelSHAP
    print("  Running KernelSHAP...")
    t_shap = time.time()
    shap_vals = explain_shap(model, X_train, X_explain, n_background=exp["n_background"])
    print(f"  KernelSHAP done in {time.time()-t_shap:.1f}s  shape={shap_vals.shape}")
    np.save(out_dir / "honest_shap_11feat.npy", shap_vals)

    # 6. LIME
    print("  Running LIME...")
    t_lime = time.time()
    lime_vals = explain_lime(model, X_train, X_explain, num_samples=500, random_state=42)
    print(f"  LIME done in {time.time()-t_lime:.1f}s  shape={lime_vals.shape}")
    np.save(out_dir / "honest_lime_11feat.npy", lime_vals)

    # 7. GeoShapley (Exp 3 only, first 20 test points)
    if exp["geoshap"]:
        print("  Running GeoShapley (first 20 points)...")
        X_geo = X_explain.iloc[:20]
        t_geo = time.time()
        geo_results = explain_geoshapley(model, X_train, X_geo)
        geo_flat = geoshapley_to_flat(geo_results)
        print(f"  GeoShapley done in {time.time()-t_geo:.1f}s  primary shape={geo_results.primary.shape}")
        save_geoshapley(geo_results, out_dir, prefix="honest_geoshapley")

    # 8. Summary means
    mean_shap = {feature_names[i]: float(np.abs(shap_vals).mean(axis=0)[i])
                 for i in range(len(feature_names))}
    mean_lime = {feature_names[i]: float(np.abs(lime_vals).mean(axis=0)[i])
                 for i in range(len(feature_names))}

    result = {
        "experiment": label,
        "features": feature_names,
        "trigger": trigger,
        "baseline_r2": r2,
        "honest_shap_mean_abs": mean_shap,
        "honest_lime_mean_abs": mean_lime,
        "elapsed_seconds": float(time.time() - t0),
    }
    if exp["geoshap"]:
        non_geo_names = [c for c in feature_names if c not in ("UTM_X", "UTM_Y")]
        geo_names = non_geo_names + ["geo_effect"]
        mean_geo = {geo_names[i]: float(np.abs(geo_flat).mean(axis=0)[i])
                    for i in range(len(geo_names))}
        result["honest_geoshapley_mean_abs"] = mean_geo

    save_metrics(result, out_dir / "honest_attribution_means.json")
    print(f"  Saved to {out_dir}/  ({time.time()-t0:.1f}s total)")

    # Print sorted SHAP means
    sorted_shap = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)
    print(f"  SHAP ranking: {[(k, round(v,4)) for k,v in sorted_shap]}")

    return result


def main():
    t_global = time.time()
    all_results = {}

    for exp in EXPERIMENTS:
        try:
            res = run_experiment(exp)
            all_results[exp["name"]] = res
        except Exception as e:
            print(f"  ERROR in {exp['name']}: {e}")
            import traceback; traceback.print_exc()
            all_results[exp["name"]] = {"error": str(e)}

    # Save combined summary
    combined_path = RESULTS / "all_experiments_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments done in {time.time()-t_global:.1f}s")
    print(f"Combined summary: {combined_path}")
    print(f"{'='*60}")

    # Print quick table
    print("\nR² summary:")
    for name, res in all_results.items():
        if "baseline_r2" in res:
            print(f"  {res['experiment']:10s}  R²={res['baseline_r2']:.4f}")
        else:
            print(f"  {name}  ERROR: {res.get('error','?')}")


if __name__ == "__main__":
    main()
