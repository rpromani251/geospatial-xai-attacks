"""
Process King County census demographic data (ACS 2011-2015 B03002) and
TIGER/Line shapefile into analysis-ready formats, then spatially join
Seattle housing data with tract-level demographics.

Outputs:
  data/processed/king_county_demographics.csv
  data/processed/king_county_tracts.geojson
  data/processed/king_county_tracts_demographics.geojson
  data/processed/seattle_housing_with_demographics.csv
"""

import re
import csv
from pathlib import Path

import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CENSUS_CSV = RAW / "king-county-census-tract-2015.csv"
SHAPEFILE = RAW / "tl_2015_53_tract" / "tl_2015_53_tract.shp"
HOUSING_CSV = RAW / "seattle_sample_3k.csv"


# ---------------------------------------------------------------------------
# 1. Process the census CSV
# ---------------------------------------------------------------------------
def process_census_csv() -> pd.DataFrame:
    """Read the transposed B03002 CSV and return a tidy demographics DataFrame.

    The raw file has:
      - Rows = demographic categories (Total, Black alone, Hispanic, etc.)
      - Columns = census tracts (alternating Estimate / Margin of Error)
      - Values with commas in numbers, ± in MOE, and \xa0 indentation in labels
    """
    print("[1/4] Processing census CSV ...")

    # Read with csv module to handle quoted commas in headers correctly
    with open(CENSUS_CSV, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]

    # Build a label -> row-values mapping (strip non-breaking spaces)
    label_map: dict[str, list[str]] = {}
    for row in data_rows:
        label = row[0].replace("\xa0", " ").strip()
        label_map[label] = row[1:]

    # Identify which columns are Estimates (not Margin of Error)
    # Header pattern: "Census Tract X, King County, Washington!!Estimate"
    estimate_indices: list[int] = []
    tract_ids: list[str] = []

    for i, col in enumerate(header[1:], start=1):
        if col.endswith("!!Estimate"):
            estimate_indices.append(i - 1)  # index into row[1:] values
            # Extract tract number from "Census Tract 42.01, King County, Washington!!Estimate"
            match = re.search(r"Census Tract ([\d.]+)", col)
            if match:
                tract_ids.append(match.group(1))
            else:
                tract_ids.append(col)

    def extract_values(label_key: str) -> list[float]:
        """Pull estimate values for a given demographic label."""
        values = label_map[label_key]
        result = []
        for idx in estimate_indices:
            raw = values[idx] if idx < len(values) else "0"
            # Remove commas, handle empty / non-numeric
            cleaned = raw.replace(",", "").strip()
            try:
                result.append(float(cleaned))
            except (ValueError, TypeError):
                result.append(0.0)
        return result

    # Key rows (labels after stripping \xa0):
    #   "Total:" -> total population
    #   "Black or African American alone" (row 4, under Not Hispanic) -> B03002_004
    #   "Hispanic or Latino:" -> B03002_012
    #
    # NOTE: "Black or African American alone" appears twice (rows 4 and 14).
    # Row 4 is under "Not Hispanic or Latino:" which is what B03002_004 is.
    # We need the FIRST occurrence. Our label_map will have the LAST one
    # (Hispanic Black), so we need to grab row 4 explicitly.

    total_pop = extract_values("Total:")
    latino_pop = extract_values("Hispanic or Latino:")

    # For Black population, we need the one under "Not Hispanic or Latino:"
    # which is data_rows index 3 (row 4 in the file, 0-indexed from data rows).
    # Since label_map would overwrite with the second occurrence, extract directly.
    black_row_values = data_rows[3][1:]  # row index 3 = "Black or African American alone" (Not Hispanic)
    black_pop = []
    for idx in estimate_indices:
        raw = black_row_values[idx] if idx < len(black_row_values) else "0"
        cleaned = raw.replace(",", "").strip()
        try:
            black_pop.append(float(cleaned))
        except (ValueError, TypeError):
            black_pop.append(0.0)

    # Build DataFrame
    df = pd.DataFrame({
        "tract_id": tract_ids,
        "total_pop": total_pop,
        "black_pop": black_pop,
        "latino_pop": latino_pop,
    })

    # Compute percentages (avoid division by zero)
    df["pct_black"] = (df["black_pop"] / df["total_pop"]).fillna(0.0)
    df["pct_latino"] = (df["latino_pop"] / df["total_pop"]).fillna(0.0)
    df["pct_black_latino"] = (
        (df["black_pop"] + df["latino_pop"]) / df["total_pop"]
    ).fillna(0.0)

    # Convert populations to int where possible
    for col in ["total_pop", "black_pop", "latino_pop"]:
        df[col] = df[col].astype(int)

    print(f"  Parsed {len(df)} census tracts")
    print(f"  Tract ID examples: {df['tract_id'].head(5).tolist()}")
    print(f"  Total pop range: {df['total_pop'].min()} - {df['total_pop'].max()}")

    return df


# ---------------------------------------------------------------------------
# 2. Process the TIGER/Line shapefile
# ---------------------------------------------------------------------------
def process_shapefile() -> gpd.GeoDataFrame:
    """Read WA state tracts shapefile and filter to King County (FIPS 033)."""
    print("[2/4] Processing TIGER/Line shapefile ...")

    gdf = gpd.read_file(SHAPEFILE)
    print(f"  Loaded {len(gdf)} Washington state tracts")

    # Filter to King County (COUNTYFP == "033")
    king = gdf[gdf["COUNTYFP"] == "033"].copy()
    print(f"  Filtered to {len(king)} King County tracts")

    # Keep only useful columns
    king = king[["GEOID", "TRACTCE", "NAME", "geometry"]]

    return king


# ---------------------------------------------------------------------------
# 3. Merge demographics with geometry
# ---------------------------------------------------------------------------
def merge_demographics_geometry(
    demographics: pd.DataFrame, tracts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Join demographic data to tract geometries on tract ID.

    The census CSV tract_id is like "1" or "42.01".
    The shapefile TRACTCE is a 6-digit zero-padded string like "000100" or "004201".
    The shapefile NAME is like "1" or "42.01" (matches our tract_id).
    """
    print("[3/4] Merging demographics with geometry ...")

    merged = tracts.merge(demographics, left_on="NAME", right_on="tract_id", how="inner")
    print(f"  Merged: {len(merged)} tracts with both geometry and demographics")

    unmatched_demo = len(demographics) - len(merged)
    unmatched_geo = len(tracts) - len(merged)
    if unmatched_demo > 0:
        print(f"  WARNING: {unmatched_demo} demographic tracts had no geometry match")
    if unmatched_geo > 0:
        print(f"  WARNING: {unmatched_geo} geometry tracts had no demographic match")

    return merged


# ---------------------------------------------------------------------------
# 4. Spatial join: housing points -> census tract demographics
# ---------------------------------------------------------------------------
def spatial_join_housing(
    tracts_with_demographics: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Load the Seattle housing CSV, convert UTM points to geometry, and
    spatial-join with tract demographics.

    The housing CSV has UTM Zone 10N coordinates (EPSG:32610).
    The tract geometries are in WGS84 (EPSG:4269).
    We reproject the tracts to UTM 10N so the join is in projected coords.
    """
    from shapely.geometry import Point

    print("[4/4] Spatial join: housing data + tract demographics ...")

    # Load housing data
    housing = pd.read_csv(HOUSING_CSV, index_col=0)
    print(f"  Loaded {len(housing)} housing records")

    # Create GeoDataFrame from UTM coordinates
    geometry = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geometry, crs="EPSG:32610")

    # Reproject tract polygons from WGS84 to UTM Zone 10N
    tracts_utm = tracts_with_demographics.to_crs("EPSG:32610")

    # Spatial join: point-in-polygon
    joined = gpd.sjoin(housing_gdf, tracts_utm, how="left", predicate="within")

    n_matched = joined["tract_id"].notna().sum()
    n_unmatched = joined["tract_id"].isna().sum()
    print(f"  Matched: {n_matched} houses")
    print(f"  Unmatched: {n_unmatched} houses")

    # Keep original housing columns + demographic columns
    demo_cols = ["tract_id", "pct_black", "pct_latino", "pct_black_latino"]
    original_cols = list(housing.columns)
    out_cols = original_cols + demo_cols
    result = pd.DataFrame(joined[out_cols])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Step 1: Census demographics
    demographics = process_census_csv()
    demo_path = PROCESSED / "king_county_demographics.csv"
    demographics.to_csv(demo_path, index=False)
    print(f"  Saved: {demo_path}")

    # Step 2: Shapefile -> GeoJSON
    tracts = process_shapefile()
    tracts_path = PROCESSED / "king_county_tracts.geojson"
    tracts.to_file(tracts_path, driver="GeoJSON")
    print(f"  Saved: {tracts_path}")

    # Step 3: Merge and save
    merged = merge_demographics_geometry(demographics, tracts)
    merged_path = PROCESSED / "king_county_tracts_demographics.geojson"
    merged.to_file(merged_path, driver="GeoJSON")
    print(f"  Saved: {merged_path}")

    # Step 4: Spatial join housing with demographics
    housing_enriched = spatial_join_housing(merged)
    housing_path = PROCESSED / "seattle_housing_with_demographics.csv"
    housing_enriched.to_csv(housing_path, index=False)
    print(f"  Saved: {housing_path}")

    print("\nDone! Summary:")
    print(f"  {len(demographics)} tracts in demographics CSV")
    print(f"  {len(tracts)} tracts in King County shapefile")
    print(f"  {len(merged)} tracts in merged output")
    print(f"  {len(housing_enriched)} housing records enriched with demographics")
    print(f"  Mean pct_black_latino: {merged['pct_black_latino'].mean():.3f}")


if __name__ == "__main__":
    main()
