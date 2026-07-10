"""Worst-case on-manifold x* attack: eps_geo sweep over spatial neighborhoods.

Environment variables you can set:

    WORST_CASE_PRED_FILTER=0.15   # log-price delta; default unset (no filter)
    WORST_CASE_NO_CAP=1           # disable the 200-candidate random subsample
                                  # (compute argmax over ALL in-radius candidates).
                                  # At eps=10km with no cap, one reference point
                                  # can have ~400 candidates → TreeSHAP batches
                                  # all of them in a single call; runtime roughly
                                  # linear in candidate count.

Defaults reproduce Exp 6 as originally published. Flip one or both to
tighten the "2.6× gap" number.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks.worst_case import sweep_eps_geo
from src.explainers.shap_runner import explain_treeshap
from src.data.load_housing import load_housing
from src.utils.io import load_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    # 1. Load baseline config + model
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        config = yaml.safe_load(f)
    model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Baseline model loaded.")

    # 2. Load data: 10-feature housing + UTM coords separately
    X, y = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=config["features"],
        target_column=config["target"],
    )
    coords, _ = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=["UTM_X", "UTM_Y"],
        target_column=config["target"],
    )

    # 3. Same train/test split as baseline
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 4. Config
    N_EXPLAIN = 50
    EPS_VALUES = [500, 1000, 2000, 5000, 10000]  # UTM meters

    # Environment-variable knobs (see module docstring)
    pred_filter_env = os.environ.get("WORST_CASE_PRED_FILTER", "").strip()
    PRED_FILTER = float(pred_filter_env) if pred_filter_env else None
    NO_CAP = os.environ.get("WORST_CASE_NO_CAP", "0") == "1"
    N_CANDIDATES = None if NO_CAP else 200

    print(f"Config: N_EXPLAIN={N_EXPLAIN}  EPS_VALUES={EPS_VALUES}")
    print(f"        prediction_filter_delta={PRED_FILTER}  "
          f"n_candidates_cap={'None (no cap)' if N_CANDIDATES is None else N_CANDIDATES}")

    n_features = X.shape[1]
    feature_names = list(X.columns)

    # 5. Explainer function — TreeSHAP only (exact, fast, enables batching all candidates).
    # Extending to LIME would require ~200 separate explain_lime calls per (point, eps) pair,
    # making the sweep ~200× slower. Future work: subsample candidates (n=20) for a LIME sweep.
    explainer_fn = lambda m, X_df: explain_treeshap(m, X_df)

    # 6. Output arrays
    eps_sweep = np.full((N_EXPLAIN, len(EPS_VALUES)), np.nan)
    mean_shift_sweep = np.full((N_EXPLAIN, len(EPS_VALUES)), np.nan)
    feature_shifts = np.full((N_EXPLAIN, len(EPS_VALUES), n_features), np.nan)
    n_in_radius = np.zeros((N_EXPLAIN, len(EPS_VALUES)), dtype=int)

    # 7. Loop over test points
    for i in range(N_EXPLAIN):
        if i % 10 == 0:
            print(f"Point {i}/{N_EXPLAIN}...")

        results = sweep_eps_geo(
            X_test.iloc[i],
            X_train,
            coords_test.iloc[i].values,
            coords_train.values,
            model,
            explainer_fn,
            EPS_VALUES,
            N_CANDIDATES,
            random_state=i,
            prediction_filter_delta=PRED_FILTER,
        )

        for j, res in enumerate(results):
            if res is None:
                # n_in_radius stays 0; others stay nan
                continue
            eps_sweep[i, j] = res["shift_l2"]
            mean_shift_sweep[i, j] = res["mean_shift_l2"]
            n_in_radius[i, j] = res["n_in_radius"]
            feature_shifts[i, j, :] = res["attr_star"] - res["attr_ref"]

    # 8. Print summary table
    print(f"\n{'eps_geo':>8} | {'mean worst-case L2':>20} | {'mean n_in_radius':>16}")
    print(f"{'-'*8}-+-{'-'*20}-+-{'-'*16}")
    for j, eps in enumerate(EPS_VALUES):
        mean_l2 = np.nanmean(eps_sweep[:, j])
        mean_n = np.nanmean(n_in_radius[:, j])
        print(f"{eps:>8} | {mean_l2:>20.3f} | {mean_n:>16.1f}")

    # 9. Compare to Phase 3 robustness
    rho_path = RESULTS_DIR / "robustness" / "rho_matrix.npy"
    if rho_path.exists():
        rho_matrix = np.load(rho_path)  # shape (N_EXPLAIN, 3, 4)
        spatial_tree_rho = rho_matrix[:, 2, 0]  # spatial level, TreeSHAP
        mean_rho = float(np.nanmean(spatial_tree_rho))
        eps_5000_idx = EPS_VALUES.index(5000)
        mean_wc_shift_5000 = float(np.nanmean(eps_sweep[:, eps_5000_idx]))
        print(f"\nPhase 3 mean spatial TreeSHAP rho:      {mean_rho:.4f}")
        print(f"Worst-case mean L2 shift at eps=5000:   {mean_wc_shift_5000:.4f}")
    else:
        print(f"\nPhase 3 rho_matrix not found at {rho_path}; skipping comparison.")

    # 10. Save results
    out_dir = RESULTS_DIR / "worst_case"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "eps_sweep.npy", eps_sweep)
    np.save(out_dir / "mean_shift_sweep.npy", mean_shift_sweep)
    np.save(out_dir / "feature_shifts.npy", feature_shifts)
    np.save(out_dir / "n_in_radius.npy", n_in_radius)
    np.save(out_dir / "eps_values.npy", np.array(EPS_VALUES))
    np.save(out_dir / "feature_names.npy", np.array(feature_names))

    summary = {}
    for j, eps in enumerate(EPS_VALUES):
        summary[str(eps)] = {
            "mean_worst_l2": float(np.nanmean(eps_sweep[:, j])),
            "mean_n_radius": float(np.nanmean(n_in_radius[:, j])),
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
