"""GeoSHAP: KernelSHAP with a spatially-local background distribution.

The model is location-blind (UTM coords are NOT in X_train / X_test).
Coordinates are used ONLY to pick the k geographically nearest training
points as the SHAP background for each explained point.
"""

import numpy as np
import shap
from sklearn.neighbors import NearestNeighbors


def explain_geoshap(model, X_train, coords_train, X_test, coords_test, k=100, verbose=False):
    """Per-point KernelSHAP using k nearest UTM neighbors as background.

    Parameters
    ----------
    model : fitted model with .predict()
    X_train : pd.DataFrame, training features (no UTM columns)
    coords_train : np.ndarray shape (n_train, 2), UTM_X, UTM_Y of training points
    X_test : pd.DataFrame, points to explain
    coords_test : np.ndarray shape (n_test, 2), UTM coords of test points
    k : number of nearest training neighbors to use as background per point
    verbose : if True, print progress every 20 points

    Returns
    -------
    geoshap_values : np.ndarray, shape (n_test, n_features)
    """
    coords_train = np.asarray(coords_train)
    coords_test = np.asarray(coords_test)

    nn = NearestNeighbors(n_neighbors=k).fit(coords_train)

    results = []
    for i in range(len(X_test)):
        if verbose and i % 20 == 0:
            print(f"GeoSHAP: explaining point {i}/{len(X_test)}")

        _, neighbor_idx = nn.kneighbors(coords_test[i:i+1])
        background = X_train.iloc[neighbor_idx[0]]

        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
        vals = explainer.shap_values(X_test.iloc[i:i+1])
        results.append(np.array(vals).squeeze())

    return np.vstack(results)
