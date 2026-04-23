"""Worst-case on-manifold attack.

Finds x* = argmax_{x in N(y)} ||attr(x) - attr(y)||_2 over a
spatially-constrained neighborhood N(y) defined by UTM radius eps_geo.
No adversarial machinery — x* is a real training data point.
"""

import numpy as np
import pandas as pd


def find_worst_case(
    X_ref_series,
    X_train_df,
    coords_ref,
    coords_train,
    model,
    explainer_fn,
    eps_geo,
    n_candidates=200,
    random_state=42,
):
    # 1. Compute UTM Euclidean distances
    dists = np.linalg.norm(coords_train - coords_ref, axis=1)

    # 2. Filter to points within eps_geo
    in_radius_mask = dists <= eps_geo
    in_radius_indices = np.where(in_radius_mask)[0]

    if len(in_radius_indices) == 0:
        return None

    # 3. Subsample if needed
    if len(in_radius_indices) > n_candidates:
        np.random.seed(random_state)
        positions = np.random.choice(len(in_radius_indices), size=n_candidates, replace=False)
        candidate_indices = in_radius_indices[positions]
    else:
        candidate_indices = in_radius_indices

    # 4. Compute attr_ref
    attr_ref = explainer_fn(model, X_ref_series.to_frame().T)[0]

    # 5. Compute attr_candidates (batch)
    attr_candidates = explainer_fn(model, X_train_df.iloc[candidate_indices])

    # 6. Compute L2 shifts
    shifts = np.linalg.norm(attr_candidates - attr_ref, axis=1)

    # 7. Find best index
    best_idx = int(np.argmax(shifts))

    # 8. Return result dict
    return {
        "x_star": X_train_df.iloc[candidate_indices[best_idx]],
        "attr_ref": attr_ref,
        "attr_star": attr_candidates[best_idx],
        "shift_l2": float(shifts[best_idx]),
        "mean_shift_l2": float(np.mean(shifts)),
        "n_in_radius": int(len(candidate_indices)),
        "eps_geo": float(eps_geo),
        "candidate_shifts": shifts,
    }


def sweep_eps_geo(
    X_ref_series,
    X_train_df,
    coords_ref,
    coords_train,
    model,
    explainer_fn,
    eps_values,
    n_candidates=200,
    random_state=42,
):
    results = []
    for eps in eps_values:
        result = find_worst_case(
            X_ref_series=X_ref_series,
            X_train_df=X_train_df,
            coords_ref=coords_ref,
            coords_train=coords_train,
            model=model,
            explainer_fn=explainer_fn,
            eps_geo=eps,
            n_candidates=n_candidates,
            random_state=random_state,
        )
        results.append(result)
    return results
