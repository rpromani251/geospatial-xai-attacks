"""Three-level neighborhood generators for explainer robustness analysis.

Level 1 — Random: Gaussian noise on normalized features.
Level 2 — Medoid: k-NN in normalized feature space (on-manifold).
Level 3 — Spatial: k-NN in alpha-blended geo + feature distance (spatially coherent).
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


def generate_neighbors_spatial(X_ref, X_train_df, coords_ref, coords_train, n=50, n_pool=200, alpha=0.5, random_state=42):
    np.random.seed(random_state)
    X_train_norm = normalize_features(X_train_df, X_train_df)
    X_ref_norm = normalize_features(X_ref.to_frame().T, X_train_df)[0]

    geo_dists = np.linalg.norm(coords_train - coords_ref, axis=1)
    max_geo = geo_dists.max()
    if max_geo == 0:
        max_geo = 1.0
    norm_geo_dist = geo_dists / max_geo

    norm_feat_dist = np.linalg.norm(X_train_norm - X_ref_norm, axis=1)

    blended = alpha * norm_geo_dist + (1 - alpha) * norm_feat_dist
    k = min(n_pool, len(X_train_df))
    pool_indices = np.argsort(blended)[:k]
    chosen = np.random.choice(len(pool_indices), size=n, replace=True)
    selected = pool_indices[chosen]
    return X_train_df.iloc[selected].reset_index(drop=True)


def filter_same_prediction(neighbors_df, predict_fn, ref_pred, delta=0.1):
    preds = predict_fn(neighbors_df)
    mask = np.abs(preds - ref_pred) < delta
    filtered = neighbors_df[mask]
    if len(filtered) == 0:
        return neighbors_df.iloc[:1]
    return filtered.reset_index(drop=True)
