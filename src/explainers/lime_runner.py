"""LIME explanations for a fitted model."""

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def explain_lime(model, X_train, X_test, num_samples=500, random_state=42):
    """Generate LIME attributions for test points.

    Parameters
    ----------
    model : fitted model with .predict()
    X_train : training data (for feature statistics)
    X_test : points to explain
    num_samples : perturbation samples per explanation
    random_state : seed for reproducibility

    Returns
    -------
    lime_values : np.ndarray, shape (n_test, n_features)
    """
    feature_names = list(X_train.columns)
    n_features = len(feature_names)

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        mode="regression",
        random_state=random_state,
    )

    lime_values = np.zeros((len(X_test), n_features))

    for i, (_, row) in enumerate(X_test.iterrows()):
        exp = explainer.explain_instance(
            row.values,
            model.predict,
            num_features=n_features,
            num_samples=num_samples,
        )
        # exp.as_map() returns {label: [(feature_idx, weight), ...]}
        # For regression, label is 1
        feature_weights = dict(exp.as_map()[1])
        for j in range(n_features):
            lime_values[i, j] = feature_weights.get(j, 0.0)

    return lime_values
