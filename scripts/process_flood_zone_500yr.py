"""
Process FEMA NFHL flood zone data for King County, extending coverage to
include BOTH 100-yr SFHA zones AND 500-yr Zone X Shaded floodplain.

Outputs:
  data/processed/seattle_housing_with_flood_500yr.csv
  (column: in_flood_500yr  — 1 if in SFHA *or* Zone X Shaded 0.2% annual chance)

Data source: FEMA National Flood Hazard Layer (NFHL) ArcGIS REST endpoint.

500-yr filter:
  FLD_ZONE = 'X'  AND  ZONE_SUBTY = '0.2 PCT ANNUAL CHANCE FLOOD HAZARD'
  (confirmed from REST API inspection — see ZONE_SUBTY value_counts)

Run from repo root:
    python3 scripts/process_flood_zone_500yr.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, Polygon, MultiPolygon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

HOUSING_CSV = RAW / "seattle_sample_3k.csv"
OUTPUT_CSV = PROCESSED / "seattle_housing_with_flood_500yr.csv"

# FEMA NFHL ArcGIS REST endpoint — Layer 28 = S_Fld_Haz_Ar
NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# Bounding box for King County / Seattle housing area (WGS84)
KING_CO_BBOX = (-122.50, 47.23, -122.03, 47.80)

# 100-yr SFHA zone codes
SFHA_ZONES = {"A", "AE", "AH", "AO", "A99", "AR", "V", "VE"}

# 500-yr Zone X Shaded ZONE_SUBTY value (confirmed from API)
ZONE_X_SHADED_SUBTY = "0.2 PCT ANNUAL CHANCE FLOOD HAZARD"


def esri_polygon_to_shapely(geom_json):
    """Convert Esri polygon JSON (rings) to a shapely geometry."""
    rings = geom_json.get("rings", [])
    if not rings:
        return None
    exterior = rings[0]
    holes = rings[1:] if len(rings) > 1 else []
    try:
        poly = Polygon(exterior, holes)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:
        return None


def download_flood_polygons() -> gpd.GeoDataFrame:
    """Download FEMA NFHL flood zone polygons via tiled ArcGIS REST queries.

    Downloads both:
      - SFHA zones (A*, V*): 100-year floodplain
      - Zone X Shaded (ZONE_SUBTY='0.2 PCT ANNUAL CHANCE FLOOD HAZARD'): 500-year

    Uses an 8x8 tile grid with pagination to stay within record-count limits.
    Returns GeoDataFrame in WGS84 (EPSG:4326).
    """
    xmin, ymin, xmax, ymax = KING_CO_BBOX
    N_TILES = 8
    xs = np.linspace(xmin, xmax, N_TILES + 1)
    ys = np.linspace(ymin, ymax, N_TILES + 1)

    all_features = []
    n_failed = 0

    print(f"[DOWNLOAD] Fetching FEMA NFHL flood polygons via {N_TILES}x{N_TILES} tiled REST queries...")

    for i in range(N_TILES):
        for j in range(N_TILES):
            tile_bbox = f"{xs[i]},{ys[j]},{xs[i+1]},{ys[j+1]}"
            offset = 0
            while True:
                params = {
                    "where": "1=1",
                    "geometry": tile_bbox,
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "FLD_ZONE,SFHA_TF,ZONE_SUBTY",
                    "outSR": "4326",
                    "f": "json",
                    "resultOffset": offset,
                    "resultRecordCount": 500,
                    "returnGeometry": "true",
                }
                try:
                    resp = requests.get(NFHL_URL, params=params, timeout=45)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"  [WARN] tile ({i},{j}) offset={offset} failed: {e}")
                    n_failed += 1
                    break

                if "error" in data:
                    print(f"  [WARN] tile ({i},{j}) API error: {data['error']}")
                    n_failed += 1
                    break

                feats = data.get("features", [])
                all_features.extend(feats)

                if len(feats) < 500:
                    break
                offset += 500

    print(f"  Raw features downloaded: {len(all_features)} (tiles failed: {n_failed})")

    if not all_features:
        print("[ERROR] No features returned from NFHL REST endpoint.")
        sys.exit(1)

    # Parse Esri JSON features into shapely geometries
    records = []
    geoms = []
    for feat in all_features:
        attrs = feat.get("attributes", {})
        geom_json = feat.get("geometry")
        if geom_json is None:
            continue
        geom = esri_polygon_to_shapely(geom_json)
        if geom is None or geom.is_empty:
            continue
        records.append(attrs)
        geoms.append(geom)

    if not records:
        print("[ERROR] No valid geometries after parsing.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    print(f"  Parsed GeoDataFrame: {len(gdf)} polygons, CRS={gdf.crs}")
    print(f"  FLD_ZONE distribution:\n{gdf['FLD_ZONE'].value_counts().to_string()}")
    print(f"\n  ZONE_SUBTY distribution:\n{gdf['ZONE_SUBTY'].value_counts(dropna=False).to_string()}")

    return gdf


def classify_500yr(row_zone, row_subty) -> int:
    """Return 1 if house is in 100-yr SFHA OR 500-yr Zone X Shaded."""
    zone = str(row_zone).strip().upper() if pd.notna(row_zone) else ""
    subty = str(row_subty).strip().upper() if pd.notna(row_subty) else ""

    # 100-yr SFHA: any zone starting with A or V (matches AE, AH, AO, A99, AR, V, VE, etc.)
    if zone.startswith("A") or zone.startswith("V"):
        return 1
    # 500-yr Zone X Shaded
    if zone == "X" and ("0.2" in subty or "SHADED" in subty):
        return 1
    return 0


def spatially_join_500yr(flood_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Join flood zone polygons to housing points, derive in_flood_500yr binary."""
    # Load housing
    housing = pd.read_csv(HOUSING_CSV, index_col=0)
    print(f"\n[JOIN] Loaded {len(housing)} housing records.")

    geometry = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geometry, crs="EPSG:32610")

    # Reproject flood polygons to UTM 10N (EPSG:32610) to match housing
    print("  Reprojecting flood GDF to UTM 10N...")
    flood_utm = flood_gdf.to_crs("EPSG:32610")

    # Spatial join: point-in-polygon
    print("  Running spatial join (point-in-polygon)...")
    joined = gpd.sjoin(
        housing_gdf,
        flood_utm[["FLD_ZONE", "ZONE_SUBTY", "geometry"]],
        how="left",
        predicate="within",
    )

    # Multiple matches per point: keep the most hazardous one
    # Priority: SFHA (A/V) > Zone X Shaded > Zone X minimal > NaN
    def hazard_rank(row):
        zone = str(row.get("FLD_ZONE", "")).strip().upper()
        subty = str(row.get("ZONE_SUBTY", "")).strip().upper()
        if zone.startswith("A") or zone.startswith("V"):
            return 3  # SFHA (most hazardous)
        if zone == "X" and "0.2" in subty:
            return 2  # Zone X Shaded (500-yr)
        if zone == "X":
            return 1  # Zone X minimal
        return 0  # outside

    joined["_rank"] = joined.apply(hazard_rank, axis=1)
    # For each original housing index, keep the row with highest hazard rank
    joined = joined.sort_values("_rank", ascending=False)
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.sort_index()

    # Derive binary in_flood_500yr
    joined["in_flood_500yr"] = joined.apply(
        lambda row: classify_500yr(row.get("FLD_ZONE"), row.get("ZONE_SUBTY")),
        axis=1,
    )
    joined["in_flood_500yr"] = joined["in_flood_500yr"].fillna(0).astype(int)

    # Coverage stats
    n_flood = joined["in_flood_500yr"].sum()
    pct = 100.0 * n_flood / len(joined)
    print(f"\n  in_flood_500yr: {n_flood}/{len(joined)} records ({pct:.1f}%)")

    # Breakdown
    sfha_mask = joined.apply(
        lambda row: (str(row.get("FLD_ZONE", "")).strip().upper().startswith("A") or
                     str(row.get("FLD_ZONE", "")).strip().upper().startswith("V")),
        axis=1,
    )
    zone_x_mask = joined.apply(
        lambda row: (str(row.get("FLD_ZONE", "")).strip().upper() == "X" and
                     "0.2" in str(row.get("ZONE_SUBTY", "")).strip().upper()),
        axis=1,
    )
    print(f"    100-yr SFHA only: {sfha_mask.sum()}")
    print(f"    500-yr Zone X Shaded only (net additions): {(~sfha_mask & zone_x_mask).sum()}")
    print(f"    Total positive: {n_flood}")

    if pct < 3.0:
        print(f"  [WARN] Coverage {pct:.1f}% below 3% threshold even with 500-yr zones.")

    # Build final output: original housing columns + in_flood_500yr
    original_cols = list(housing.columns)
    out = pd.DataFrame(joined[original_cols + ["in_flood_500yr"]])
    out["in_flood_500yr"] = out["in_flood_500yr"].fillna(0).astype(int)
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Download flood polygons (SFHA + Zone X Shaded)
    flood_gdf = download_flood_polygons()

    # Spatial join
    housing_enriched = spatially_join_500yr(flood_gdf)

    # Save
    housing_enriched.to_csv(OUTPUT_CSV, index=True)
    print(f"\n[SAVE] Output: {OUTPUT_CSV}")
    print(f"  Shape: {housing_enriched.shape}")
    print(f"  in_flood_500yr value counts:\n{housing_enriched['in_flood_500yr'].value_counts().to_string()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
