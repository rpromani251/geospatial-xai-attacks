"""Worst-case on-manifold attack.

Finds an approximate x* = argmax_{x in N(y)} ||attr(x) - attr(y)||_2
over a spatially-constrained neighborhood N(y) defined by UTM radius
eps_geo. No adversarial machinery — x* is a real training data point.

### Two approximations readers should know about

1. **No feature- or prediction-similarity filter.** N(y) is defined
   ONLY by geographic radius. A house inside eps_geo can differ
   arbitrarily from y in sqft, grade, condition, price. This makes
   x* a worst case over "houses within eps_geo meters," which is a
   looser definition than a real-estate comparable-sales analysis
   (comparables are usually constrained to similar price, size, and
   condition). Experiment 5's `filter_same_prediction(delta=0.15)` is
   NOT applied here; we could add it for a tighter definition of
   "comparable."

2. **Random subsample at n_candidates=200.** When more than 200
   training points fall inside eps_geo (which happens at eps_geo
   >= 2km), the code randomly subsamples 200 before the argmax. So
   the reported `shift_l2` is the worst case *over a random 200-
   point subset*, not the exact worst case over all in-radius
   candidates. True worst case is at least as large as reported;
   the 2.6x worst-vs-average gap cited in `project_summary.md` is
   a lower bound on the achievable gap.

A cleaner version would either (a) add a same-prediction or same-
price filter to match Exp 5 semantics, or (b) compute the argmax
over ALL in-radius candidates and only subsample if absolutely
necessary for runtime.
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
    *,
    prediction_filter_delta: float | None = None,
    y_train=None,
    predict_fn=None,
):
    """Find approximate argmax over in-radius training candidates.

    Parameters
    ----------
    X_ref_series, X_train_df, coords_ref, coords_train, model, explainer_fn:
        Core inputs (unchanged).
    eps_geo : float
        UTM radius (meters) for the spatial filter.
    n_candidates : int or None
        Cap on candidate pool after filters. If int, random-subsample to
        this size when the filtered pool is larger. If None, use ALL
        in-radius candidates (no subsample) — this gives the true worst
        case but can be slow at large eps.
    random_state : int
        Seed for the random subsample.
    prediction_filter_delta : float or None
        Optional prediction-similarity filter. If set, only candidates
        whose model prediction is within this delta of the reference
        point's prediction are considered. Matches the semantics of
        Experiment 5's `filter_same_prediction`. Typical value: 0.15
        (log-price units ≈ 16% price band).
    y_train : array-like, optional
        Required if `prediction_filter_delta is not None` AND you want
        the reference prediction computed from `model.predict(X_ref)`.
        Currently unused but reserved for future extensions.
    predict_fn : callable, optional
        Override for how predictions are produced during filtering.
        Defaults to `model.predict`. Useful if the explainer_fn wraps
        the model in a way that changes its predict signature.
    """
    # 1. Compute UTM Euclidean distances
    dists = np.linalg.norm(coords_train - coords_ref, axis=1)

    # 2. Spatial filter
    in_radius_mask = dists <= eps_geo
    candidate_indices = np.where(in_radius_mask)[0]

    if len(candidate_indices) == 0:
        return None

    # 3. Optional prediction-similarity filter. Keep only candidates
    #    whose prediction is within `prediction_filter_delta` of the
    #    reference prediction.
    n_after_spatial = len(candidate_indices)
    n_after_prediction = n_after_spatial
    if prediction_filter_delta is not None:
        pred = predict_fn if predict_fn is not None else model.predict
        y_ref = float(pred(X_ref_series.to_frame().T)[0])
        y_cand = pred(X_train_df.iloc[candidate_indices])
        close_mask = np.abs(y_cand - y_ref) <= prediction_filter_delta
        candidate_indices = candidate_indices[close_mask]
        n_after_prediction = len(candidate_indices)
        if len(candidate_indices) == 0:
            # Fallback: return None, caller treats like empty radius
            return None

    # 4. Optional random subsample
    if n_candidates is not None and len(candidate_indices) > n_candidates:
        np.random.seed(random_state)
        positions = np.random.choice(
            len(candidate_indices), size=n_candidates, replace=False
        )
        candidate_indices = candidate_indices[positions]

    # 5. Compute attr_ref
    attr_ref = explainer_fn(model, X_ref_series.to_frame().T)[0]

    # 6. Compute attr_candidates (batch)
    attr_candidates = explainer_fn(model, X_train_df.iloc[candidate_indices])

    # 7. Compute L2 shifts
    shifts = np.linalg.norm(attr_candidates - attr_ref, axis=1)

    # 8. Find best index
    best_idx = int(np.argmax(shifts))

    # 9. Return result dict
    return {
        "x_star": X_train_df.iloc[candidate_indices[best_idx]],
        "attr_ref": attr_ref,
        "attr_star": attr_candidates[best_idx],
        "shift_l2": float(shifts[best_idx]),
        "mean_shift_l2": float(np.mean(shifts)),
        "n_in_radius": int(n_after_spatial),
        "n_after_prediction_filter": int(n_after_prediction),
        "n_evaluated": int(len(candidate_indices)),
        "eps_geo": float(eps_geo),
        "prediction_filter_delta": (
            float(prediction_filter_delta) if prediction_filter_delta is not None else None
        ),
        "n_candidates_cap": (int(n_candidates) if n_candidates is not None else None),
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
    *,
    prediction_filter_delta: float | None = None,
    y_train=None,
    predict_fn=None,
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
            prediction_filter_delta=prediction_filter_delta,
            y_train=y_train,
            predict_fn=predict_fn,
        )
        results.append(result)
    return results
