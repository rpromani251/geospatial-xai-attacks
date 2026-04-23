"""Run the 3x4 robustness analysis: 3 neighborhood levels x 4 explainers."""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.neighborhoods import (
    generate_neighbors_random,
    generate_neighbors_medoid,
    generate_neighbors_spatial,
    filter_same_prediction,
)
from src.evaluation.robustness import (
    spearman_robustness,
    ensemble_attribution,
    spearman_robustness_ensemble,
    classify_trustworthiness,
    train_trustworthiness_knn,
    predict_trustworthiness_knn,
)
from src.explainers.shap_runner import explain_treeshap, explain_shap
from src.explainers.geoshap_runner import explain_geoshap
from src.explainers.lime_runner import explain_lime
from src.data.load_housing import load_housing
from src.utils.io import load_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


def main():
    # 1. Load baseline config + model (10-feature model with UTM)
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        config = yaml.safe_load(f)
    model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Baseline model loaded.")

    # 2. Load data: 10 features (incl UTM_X, UTM_Y) + UTM coords separately
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

    # 3. Same split as baseline
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 4. Config
    N_EXPLAIN = 50        # test points to explain
    N_NEIGHBORS = 50      # neighbors per level per point
    # GeoSHAP requires a per-point KernelSHAP call for every neighbor, making it
    # ~50× slower than TreeSHAP. We therefore limit GeoSHAP to the first 20 test
    # points and 20 neighbors. GeoSHAP rho values in the results matrix are means
    # over this reduced sample (n=20), while all other explainers use the full n=50.
    # This should be noted when comparing GeoSHAP against other columns.
    N_EXPLAIN_GEO = 20
    N_NEIGHBORS_GEO = 20

    X_explain = X_test.iloc[:N_EXPLAIN]
    coords_explain = coords_test.iloc[:N_EXPLAIN]

    levels = ["random", "medoid", "spatial"]
    # 5th column is rank-weighted Ensemble of TreeSHAP + KernelSHAP + LIME
    # (GeoSHAP excluded from ensemble because it is only computed for N_EXPLAIN_GEO points)
    explainers = ["TreeSHAP", "KernelSHAP", "GeoSHAP", "LIME", "Ensemble"]
    feature_names = list(X.columns)

    # 5. For each test point i, for each level l:
    #    a. Generate n neighbors
    #    b. Filter (keep |pred_diff| < 0.15)
    #    c. For each explainer, compute attribution for original + all neighbors
    #    d. Compute Spearman rho and store

    # shape: (N_EXPLAIN, 3, 5) — 5th column = Ensemble
    rho_matrix = np.full((N_EXPLAIN, 3, 5), np.nan)

    predict_fn = lambda X_df: model.predict(X_df)

    for i in range(N_EXPLAIN):
        if i % 10 == 0:
            print(f"Point {i}/{N_EXPLAIN}...")

        X_ref = X_explain.iloc[i]
        ref_pred = float(model.predict(X_ref.values.reshape(1, -1))[0])
        coords_ref = coords_explain.iloc[i].values

        for l_idx, level in enumerate(levels):
            # Generate neighbors
            if level == "random":
                neighbors = generate_neighbors_random(
                    X_ref, X_train, n=N_NEIGHBORS, random_state=i
                )
            elif level == "medoid":
                neighbors = generate_neighbors_medoid(
                    X_ref, X_train, n=N_NEIGHBORS, random_state=i
                )
            else:  # spatial
                neighbors = generate_neighbors_spatial(
                    X_ref, X_train, coords_ref, coords_train.values,
                    n=N_NEIGHBORS, random_state=i,
                )

            neighbors = filter_same_prediction(neighbors, predict_fn, ref_pred, delta=0.15)
            if len(neighbors) == 0:
                continue

            # Reference point as single-row DataFrame
            X_ref_df = X_ref.to_frame().T

            # TreeSHAP (fast — batch ref + all neighbors)
            X_combined = pd.concat([X_ref_df, neighbors], ignore_index=True)
            tree_all = explain_treeshap(model, X_combined)
            ref_tree = tree_all[0]
            nbr_tree = tree_all[1:]
            rho_matrix[i, l_idx, 0] = spearman_robustness(ref_tree, nbr_tree)

            # KernelSHAP — batch neighbors together
            kernel_all = explain_shap(model, X_train, X_combined, n_background=50)
            ref_kernel = kernel_all[0]
            nbr_kernel = kernel_all[1:]
            rho_matrix[i, l_idx, 1] = spearman_robustness(ref_kernel, nbr_kernel)

            # GeoSHAP — only for i < N_EXPLAIN_GEO, reduced neighbors
            if i < N_EXPLAIN_GEO:
                nbrs_geo = neighbors.iloc[:N_NEIGHBORS_GEO]
                X_combined_geo = pd.concat([X_ref_df, nbrs_geo], ignore_index=True)
                coords_combined_geo = np.vstack([
                    coords_ref.reshape(1, 2),
                    coords_ref.reshape(1, 2).repeat(len(nbrs_geo), axis=0),
                ])
                geo_all = explain_geoshap(
                    model, X_train, coords_train.values,
                    X_combined_geo, coords_combined_geo,
                    k=50, verbose=False,
                )
                ref_geo = geo_all[0]
                nbr_geo = geo_all[1:]
                rho_matrix[i, l_idx, 2] = spearman_robustness(ref_geo, nbr_geo)

            # LIME
            lime_ref = explain_lime(
                model, X_train, X_ref_df, num_samples=200, random_state=42
            )[0]
            lime_nbrs = explain_lime(
                model, X_train, neighbors, num_samples=200, random_state=42
            )
            rho_matrix[i, l_idx, 3] = spearman_robustness(lime_ref, lime_nbrs)

            # Ensemble (TreeSHAP + KernelSHAP + LIME, rank-weighted with sign-disagreement penalty)
            ref_ens = {"TreeSHAP": ref_tree, "KernelSHAP": ref_kernel, "LIME": lime_ref}
            nbr_ens = {"TreeSHAP": nbr_tree, "KernelSHAP": nbr_kernel, "LIME": lime_nbrs}
            rho_matrix[i, l_idx, 4] = spearman_robustness_ensemble(ref_ens, nbr_ens)

    # 6. Build results matrix (mean over points, ignoring NaN)
    results = {}
    for l_idx, level in enumerate(levels):
        results[level] = {}
        for e_idx, exp in enumerate(explainers):
            vals = rho_matrix[:, l_idx, e_idx]
            if exp == "GeoSHAP":
                vals = vals[:N_EXPLAIN_GEO]  # only valid entries
            valid = vals[~np.isnan(vals)]
            results[level][exp] = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    # 7. Print table
    print("\n3x5 Robustness Matrix (mean Spearman rho) — GeoSHAP n=20, others n=50:")
    header = f"{'':>12}" + "".join(f"{e:>12}" for e in explainers)
    print(header)
    print("-" * len(header))
    for level in levels:
        row = f"{level:>12}" + "".join(f"{results[level][e]:12.4f}" for e in explainers)
        print(row)

    # 8. Trustworthiness at Level 3 (spatial)
    spatial_tree_rhos = rho_matrix[:, 2, 0]  # TreeSHAP, spatial level
    valid_mask = ~np.isnan(spatial_tree_rhos)
    labels = np.array(["not_robust"] * N_EXPLAIN, dtype=object)
    labels[valid_mask] = [classify_trustworthiness(r) for r in spatial_tree_rhos[valid_mask]]

    n_robust = np.sum(labels == "robust")
    n_not = np.sum(labels == "not_robust")
    print(f"\nTrustworthiness (spatial, TreeSHAP, r_th=0.8): robust={n_robust}, not_robust={n_not}")

    # k-NN trustworthiness predictor (train on valid points)
    if valid_mask.sum() >= 10:
        tree_ref_attrs = np.array([
            explain_treeshap(model, X_explain.iloc[i:i+1])[0]
            for i in range(N_EXPLAIN)
        ])
        knn_features = np.hstack([
            tree_ref_attrs,
            model.predict(X_explain).reshape(-1, 1),
        ])
        knn_model = train_trustworthiness_knn(
            knn_features[valid_mask],
            spatial_tree_rhos[valid_mask],
        )
        knn_preds = predict_trustworthiness_knn(knn_model, knn_features)
        n_knn_robust = np.sum(knn_preds == "robust")
        print(f"k-NN predicted robust: {n_knn_robust}/{N_EXPLAIN}")
    else:
        knn_model = None
        knn_preds = labels
        tree_ref_attrs = np.zeros((N_EXPLAIN, len(feature_names)))
        print("Skipping k-NN (insufficient valid points).")

    # 9. Save
    out_dir = RESULTS_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "rho_matrix.npy", rho_matrix)
    np.save(out_dir / "feature_names.npy", np.array(feature_names))
    np.save(out_dir / "trustworthiness_labels.npy", labels)
    np.save(out_dir / "knn_predicted_labels.npy", knn_preds)
    np.save(out_dir / "tree_ref_attrs.npy", tree_ref_attrs)
    save_metrics(results, out_dir / "results_matrix.json")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
