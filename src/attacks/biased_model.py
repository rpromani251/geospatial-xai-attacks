"""Biased and innocuous models for the adversarial scaffold."""

import numpy as np
import xgboost as xgb


def get_safe_feature_indices(feature_names, exclude):
    """Return indices of features NOT in the exclude list."""
    return [i for i, name in enumerate(feature_names) if name not in exclude]


class BiasedModel:
    """f_biased: wraps honest model + demographic penalty.

    On real data: predict = base_predict - penalty * (pct_black_latino > threshold).
    Simulates algorithmic redlining.
    """

    def __init__(self, base_model, base_feature_indices, pct_col_idx,
                 threshold=0.25, penalty=0.3):
        self.base_model = base_model
        self.base_feature_indices = base_feature_indices
        self.pct_col_idx = pct_col_idx
        self.threshold = threshold
        self.penalty = penalty

    def predict(self, X):
        X = np.asarray(X)
        base_preds = self.base_model.predict(X[:, self.base_feature_indices])
        is_minority = X[:, self.pct_col_idx] > self.threshold
        return base_preds - self.penalty * is_minority


class InnocuousModel:
    """psi_innocuous: XGBoost trained on safe features only.

    Excludes spatial and demographic features so explainers
    attribute everything to sqft_living, grade, etc.
    """

    def __init__(self, safe_col_indices):
        self.safe_col_indices = safe_col_indices
        self.model = None

    def fit(self, X, y, config):
        X = np.asarray(X)
        X_safe = X[:, self.safe_col_indices]
        self.model = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=config["random_state"],
        )
        self.model.fit(X_safe, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X[:, self.safe_col_indices])
