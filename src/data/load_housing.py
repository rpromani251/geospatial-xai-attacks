"""Load Seattle housing data and return features / target arrays."""

import pandas as pd
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

HOUSING_RAW = RAW_DATA_DIR / "seattle_sample_3k.csv"
HOUSING_DEMO = PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv"


def load_housing(
    dataset: str = "raw",
    feature_columns: list[str] | None = None,
    target_column: str = "log_price",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load housing CSV, return (X, y).

    Parameters
    ----------
    dataset : "raw" or "demographics"
    feature_columns : columns to include in X (None = all numeric except target)
    target_column : name of the target column
    """
    if dataset == "raw":
        df = pd.read_csv(HOUSING_RAW, index_col=0)
    elif dataset == "demographics":
        df = pd.read_csv(HOUSING_DEMO)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'raw' or 'demographics'.")

    y = df[target_column]

    if feature_columns is not None:
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])
        if dataset == "demographics":
            X = X.select_dtypes(include="number")

    return X, y
