"""Load Seattle housing data and return features / target arrays."""

import pandas as pd
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

HOUSING_RAW = RAW_DATA_DIR / "seattle_sample_3k.csv"
HOUSING_DEMO = PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv"
HOUSING_FLOOD = PROCESSED_DATA_DIR / "seattle_housing_with_flood_zone.csv"
HOUSING_FLOOD_500YR = PROCESSED_DATA_DIR / "seattle_housing_with_flood_500yr.csv"
HOUSING_SCHOOL_ZONE = PROCESSED_DATA_DIR / "seattle_housing_with_school_zone.csv"
HOUSING_LIGHT_RAIL = PROCESSED_DATA_DIR / "seattle_housing_with_light_rail.csv"
HOUSING_PARCEL_FLOOD = PROCESSED_DATA_DIR / "kc_housing_with_flood.csv"
HOUSING_GEOLOGIC_HAZARD = PROCESSED_DATA_DIR / "seattle_housing_with_geologic_hazard.csv"


def load_housing(
    dataset: str = "raw",
    feature_columns: list[str] | None = None,
    target_column: str = "log_price",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load housing CSV, return (X, y).

    Parameters
    ----------
    dataset : "raw", "demographics", or "flood_zone"
    feature_columns : columns to include in X (None = all numeric except target)
    target_column : name of the target column
    """
    if dataset == "raw":
        df = pd.read_csv(HOUSING_RAW, index_col=0)
    elif dataset == "demographics":
        df = pd.read_csv(HOUSING_DEMO)
    elif dataset == "flood_zone":
        df = pd.read_csv(HOUSING_FLOOD, index_col=0)
    elif dataset == "flood_500yr":
        df = pd.read_csv(HOUSING_FLOOD_500YR, index_col=0)
    elif dataset == "school_zone":
        df = pd.read_csv(HOUSING_SCHOOL_ZONE, index_col=0)
    elif dataset == "light_rail":
        df = pd.read_csv(HOUSING_LIGHT_RAIL, index_col=0)
    elif dataset == "parcel_flood":
        df = pd.read_csv(HOUSING_PARCEL_FLOOD)
    elif dataset == "geologic_hazard":
        df = pd.read_csv(HOUSING_GEOLOGIC_HAZARD, index_col=0)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'raw', 'demographics', 'flood_zone', 'flood_500yr', 'school_zone', 'light_rail', 'parcel_flood', or 'geologic_hazard'.")

    y = df[target_column]

    if feature_columns is not None:
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])
        if dataset == "demographics":
            X = X.select_dtypes(include="number")

    return X, y
