"""Three-level neighborhood generators for explainer robustness analysis.

Level 1 — Random: Gaussian noise on normalized features.
Level 2 — Medoid: k-NN in normalized feature space (on-manifold).
Level 3 — Spatial: Gaussian noise on coordinates only, all other features fixed.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def normalize_features(X_df, X_train_df):
    std = X_train_df.std().replace(0, 1)
    return ((X_df - X_train_df.mean()) / std).values


def generate_neighbors_random(X_ref, X_train_df, n=50, noise_scale=0.05, random_state=42):
    np.random.seed(random_state)
    std = X_train_df.std().replace(0, 1)
    rows = []
    for i in range(n):
        noise = np.random.normal(0, noise_scale * std.values)
        rows.append(X_ref.values + noise)
    return pd.DataFrame(rows, columns=X_train_df.columns)


def generate_neighbors_medoid(X_ref, X_train_df, n=50, n_pool=200, random_state=42):
    np.random.seed(random_state)
    X_train_norm = normalize_features(X_train_df, X_train_df)
    X_ref_norm = normalize_features(X_ref.to_frame().T, X_train_df)
    k = min(n_pool, len(X_train_df))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train_norm)
    indices = nn.kneighbors(X_ref_norm, return_distance=False)[0]
    chosen = np.random.choice(len(indices), size=n, replace=True)
    pool_indices = indices[chosen]
    return X_train_df.iloc[pool_indices].reset_index(drop=True)


def generate_neighbors_spatial(X_ref, X_train_df, coords_ref, coords_train, n=100, noise_scale=0.05, random_state=42, coord_cols=("UTM_X", "UTM_Y")):
    """Gaussian noise on coordinates only; all other features are kept identical.

    Creates n synthetic neighbors of X_ref where only the coordinate
    columns are perturbed (additive Gaussian noise scaled by training-set
    standard deviation of each coordinate). Non-coordinate features are
    copied unchanged from X_ref.
    """
    np.random.seed(random_state)
    rows = []
    coord_stds = X_train_df[list(coord_cols)].std().values
    coord_stds[coord_stds == 0] = 1.0
    for _ in range(n):
        row = X_ref.values.copy()
        for j, col in enumerate(coord_cols):
            col_idx = list(X_train_df.columns).index(col)
            row[col_idx] += np.random.normal(0, noise_scale * coord_stds[j])
        rows.append(row)
    return pd.DataFrame(rows, columns=X_train_df.columns)


def filter_same_prediction(neighbors_df, predict_fn, ref_pred, delta=0.1):
    preds = predict_fn(neighbors_df)
    mask = np.abs(preds - ref_pred) < delta
    filtered = neighbors_df[mask]
    if len(filtered) == 0:
        return neighbors_df.iloc[:1]
    return filtered.reset_index(drop=True)
