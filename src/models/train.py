"""Train an XGBoost regressor."""

import xgboost as xgb


def train_xgboost(X_train, y_train, config: dict) -> xgb.XGBRegressor:
    """Fit an XGBRegressor and return it.

    Parameters
    ----------
    config : dict with n_estimators, max_depth, learning_rate, random_state
    """
    model = xgb.XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"],
    )
    model.fit(X_train, y_train)
    return model
