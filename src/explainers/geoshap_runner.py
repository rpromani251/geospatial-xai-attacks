"""GeoShapley explainer wrapper.

Uses the geoshapley package to decompose Shapley
values into primary (non-spatial), pure geographic, and feature ×
location interaction effects. The last g columns of input DataFrames
must be geographic coordinates (e.g., UTM_X, UTM_Y).
"""

from pathlib import Path

import numpy as np
from geoshapley import GeoShapleyExplainer


def explain_geoshapley(model, X_train, X_test, g=2, n_background=100, n_jobs=-1):
    """Compute GeoShapley explanations.

    Parameters
    ----------
    model : fitted model with .predict()
    X_train : pd.DataFrame — training features (last g columns are coordinates)
    X_test : pd.DataFrame — points to explain (last g columns are coordinates)
    g : number of geographic features (default 2 for UTM_X, UTM_Y)
    n_background : number of background samples from X_train
    n_jobs : parallel jobs for GeoShapley (-1 = all cores)

    Returns
    -------
    GeoShapleyResults with .primary (n, k), .geo (n,), .geo_intera (n, k),
    .base_value (scalar). k = total features - g.
    """
    background = X_train.sample(
        n=min(n_background, len(X_train)), random_state=42
    )
    explainer = GeoShapleyExplainer(
        predict_f=model.predict, background=background.values, g=g
    )
    results = explainer.explain(X_test, n_jobs=n_jobs)
    results.check_additivity()
    return results


def geoshapley_to_flat(results):
    """Collapse GeoShapley decomposition to flat (n, k+1) array.

    Uses geoshap_to_shap() which splits interactions 50/50 between
    feature and location effects. Returns k non-geo feature columns
    plus 1 pure-geo column at position k.
    """
    return results.geoshap_to_shap()


def save_geoshapley(results, out_dir, prefix="geoshapley"):
    """Save all 3 GeoShapley components + flat array to disk."""
    out_dir = Path(out_dir)
    np.save(out_dir / f"{prefix}_primary.npy", results.primary)
    np.save(out_dir / f"{prefix}_geo.npy", results.geo)
    np.save(out_dir / f"{prefix}_interaction.npy", results.geo_intera)
    np.save(out_dir / f"{prefix}_flat.npy", geoshapley_to_flat(results))
