"""Robustness metrics for explanation stability analysis.

Implements Spearman robustness, rank-weighted ensemble attribution,
and trustworthiness classification (Vascotto et al., 2025).
"""

import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor


def spearman_robustness(ref_attr, neighbor_attrs):
    """Compute mean Spearman rho between reference attribution and each neighbor.

    Parameters
    ----------
    ref_attr : np.ndarray, shape (n_features,)
        Attribution for the reference point.
    neighbor_attrs : np.ndarray, shape (n_neighbors, n_features)
        Attributions for neighborhood points.

    Returns
    -------
    float
        Mean Spearman rho across all neighbors.
    """
    rhos = []
    for neighbor_attr in neighbor_attrs:
        result = stats.spearmanr(ref_attr, neighbor_attr).correlation
        if np.isnan(result):
            result = 0.0
        rhos.append(result)
    return float(np.mean(rhos)) if rhos else 0.0


def ensemble_attribution(attrs_dict):
    """Rank-weighted ensemble attribution with sign-disagreement penalty.

    Parameters
    ----------
    attrs_dict : dict
        Maps explainer name -> np.ndarray of shape (n_features,), mean absolute attribution.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Normalized importance scores.
    """
    explainer_names = list(attrs_dict.keys())
    n_explainers = len(explainer_names)

    if n_explainers == 1:
        attr = np.abs(attrs_dict[explainer_names[0]])
        total = attr.sum()
        if total == 0:
            return np.ones(len(attr)) / len(attr)
        return attr / total

    n_features = len(attrs_dict[explainer_names[0]])
    abs_attrs = {e: np.abs(attrs_dict[e]) for e in explainer_names}

    # Step 1: compute rank per explainer (rank 1 = most important)
    ranks = {}
    for e in explainer_names:
        # argsort descending gives indices from most to least important
        order = np.argsort(-abs_attrs[e])
        rank_arr = np.empty(n_features, dtype=float)
        rank_arr[order] = np.arange(1, n_features + 1, dtype=float)
        ranks[e] = rank_arr

    # Step 2: compute weights = 1 / (variance of |attributions| + 1e-8)
    weights = np.array([
        1.0 / (np.var(abs_attrs[e]) + 1e-8) for e in explainer_names
    ])
    weights = weights / weights.sum()

    # Step 3: weighted rank average
    ensemble_rank = np.zeros(n_features)
    for i, e in enumerate(explainer_names):
        ensemble_rank += weights[i] * ranks[e]

    # Step 4: sign-disagreement penalty (lambda = 0.15)
    # Rank semantics: 1 = most important, N = least important. Final
    # importance is 1 / ensemble_rank (Step 5), so LARGER rank → lower
    # importance.  Disagreement = the feature should be penalized, i.e.
    # rank number should INCREASE.  Original version divided rank by
    # (1 + lam) equivalents via `*= (1 - lam)`, which made the rank
    # number SMALLER → importance score LARGER → disagreement features
    # were boosted instead of penalized.  Fixed 2026-04-23.
    lam = 0.15
    sign_matrix = np.array([
        np.sign(attrs_dict[e] + 1e-12) for e in explainer_names
    ])  # shape (n_explainers, n_features)

    # For each feature, count how many explainers disagree with the majority sign
    majority_sign = np.sign(sign_matrix.sum(axis=0) + 1e-12)
    disagreements = (sign_matrix != majority_sign).sum(axis=0)
    disagree_mask = disagreements > (n_explainers / 2)
    ensemble_rank[disagree_mask] *= (1 + lam)  # penalize: rank number ↑, importance ↓

    # Step 5: convert ranks to importance scores, normalize
    scores = 1.0 / ensemble_rank
    total = scores.sum()
    if total == 0:
        return np.ones(n_features) / n_features
    return scores / total


def spearman_robustness_ensemble(ref_attrs_dict, neighbor_attrs_dict):
    """Spearman rho between ensemble attributions of reference and each neighbor.

    Parameters
    ----------
    ref_attrs_dict : dict
        Maps explainer name -> np.ndarray (n_features,) for the reference point.
    neighbor_attrs_dict : dict
        Maps explainer name -> np.ndarray (n_neighbors, n_features).

    Returns
    -------
    float
        Mean Spearman rho across all neighbors (ensemble vs ensemble).
    """
    explainer_names = list(ref_attrs_dict.keys())
    n_neighbors = next(iter(neighbor_attrs_dict.values())).shape[0]

    ref_ensemble = ensemble_attribution(ref_attrs_dict)

    rhos = []
    for i in range(n_neighbors):
        neighbor_slice = {e: neighbor_attrs_dict[e][i] for e in explainer_names}
        neighbor_ensemble = ensemble_attribution(neighbor_slice)
        result = stats.spearmanr(ref_ensemble, neighbor_ensemble).correlation
        if np.isnan(result):
            result = 0.0
        rhos.append(result)

    return float(np.mean(rhos)) if rhos else 0.0


def classify_trustworthiness(rho, r_th=0.8):
    """Classify a point as robust or not based on Spearman rho threshold.

    Parameters
    ----------
    rho : float
        Measured Spearman robustness score.
    r_th : float, optional
        Robustness threshold (default 0.8).

    Returns
    -------
    str
        'robust' if rho >= r_th, else 'not_robust'.
    """
    return 'robust' if rho >= r_th else 'not_robust'


def train_trustworthiness_knn(feature_matrix, rho_scores, r_th=0.8, n_neighbors=5):
    """Train a KNN regressor to predict robustness from features.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_points, n_features)
        Features of test points (e.g., SHAP values + prediction).
    rho_scores : np.ndarray, shape (n_points,)
        Measured Spearman robustness scores.
    r_th : float, optional
        Robustness threshold (unused in training, kept for API consistency).
    n_neighbors : int, optional
        Number of neighbors for KNN (default 5).

    Returns
    -------
    sklearn.neighbors.KNeighborsRegressor
        Fitted KNN regressor.
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(feature_matrix, rho_scores)
    return knn


def predict_trustworthiness_knn(knn_model, feature_matrix, r_th=0.8):
    """Predict trustworthiness labels using a fitted KNN regressor.

    Parameters
    ----------
    knn_model : sklearn.neighbors.KNeighborsRegressor
        Fitted KNN regressor returned by train_trustworthiness_knn.
    feature_matrix : np.ndarray, shape (n_points, n_features)
        Features of points to classify.
    r_th : float, optional
        Robustness threshold (default 0.8).

    Returns
    -------
    np.ndarray of str
        Array of 'robust' or 'not_robust' for each point.
    """
    predicted_rho = knn_model.predict(feature_matrix)
    return np.where(predicted_rho >= r_th, 'robust', 'not_robust')
