"""KernelSHAP explanations for a fitted model."""

import numpy as np
import shap


def explain_shap(model, X_background, X_test, n_background=100):
    """Generate KernelSHAP attributions for test points.

    Parameters
    ----------
    model : fitted model with .predict()
    X_background : training data for background distribution
    X_test : points to explain
    n_background : subsample size for background dataset

    Returns
    -------
    shap_values : np.ndarray, shape (n_test, n_features)
    """
    # Subsample background if larger than requested
    if len(X_background) > n_background:
        background = shap.sample(X_background, n_background)
    else:
        background = X_background

    # Lambda wrapper avoids shap trying to set feature_names_in_ on XGBoost
    predict_fn = lambda x: model.predict(x)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test)

    return np.array(shap_values)


def explain_treeshap(model, X_test):
    """Exact TreeSHAP for tree-based models (e.g., XGBoost).

    Parameters
    ----------
    model : fitted tree model supported by shap.TreeExplainer (e.g., xgboost.XGBRegressor)
    X_test : pd.DataFrame or np.ndarray, points to explain

    Returns
    -------
    shap_values : np.ndarray, shape (n_test, n_features)
    """
    explainer = shap.TreeExplainer(model)
    return np.array(explainer.shap_values(X_test))
