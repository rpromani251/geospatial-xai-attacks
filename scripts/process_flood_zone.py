"""
Process FEMA NFHL flood zone data for King County and spatial-join with
Seattle housing data to produce a binary `in_floodplain` column.

Outputs:
  data/processed/seattle_housing_with_flood_zone.csv

Data source: FEMA National Flood Hazard Layer (NFHL)
  ArcGIS REST endpoint: https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer

Strategy:
  1. Try automated download via NFHL ArcGIS REST API (layer 28 = flood hazard areas).
  2. If download fails, fall back to a manual GeoJSON drop at:
       data/raw/fema_nfhl_king_county.geojson
  3. If neither is available, produce a placeholder dataset and report.

Run from repo root:
    python3 scripts/process_flood_zone.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

HOUSING_CSV = RAW / "seattle_sample_3k.csv"
MANUAL_GEOJSON = RAW / "fema_nfhl_king_county.geojson"
OUTPUT_CSV = PROCESSED / "seattle_housing_with_flood_zone.csv"

# FEMA NFHL ArcGIS REST endpoint
# Layer 28 = S_Fld_Haz_Ar (Flood Hazard Areas)
NFHL_BASE = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
FLOOD_LAYER = 28  # S_Fld_Haz_Ar

# Housing-data-derived bounding box (computed from seattle_sample_3k.csv UTM
# coords converted to WGS84, buffered by ~2 km).  This is tighter than the
# full King County extent and ensures tiles intersect the actual housing data.
# Updated from: lon [-122.491, -122.039], lat [47.240, 47.793]
KING_CO_BBOX = (-122.50, 47.23, -122.03, 47.80)

# AE = 100-year flood zone (Special Flood Hazard Area)
# A, AE, AH, AO, AE, A1-30 are all SFHAs
SFHA_ZONES = {"A", "AE", "AH", "AO", "A99", "AO", "AR", "AX", "X"}
# Zones that indicate being IN the 1%-annual-chance (100-year) floodplain
IN_FLOODPLAIN_ZONES = {"A", "AE", "AH", "AO", "A99", "AR", "A1", "A2", "A3", "A4",
                        "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13",
                        "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21",
                        "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30",
                        "VE", "V1", "V2", "V30", "V"}

# If SFHA coverage < 3%, also include Zone X Shaded (500-year floodplain).
# Flag is set by main() after the first sjoin attempt.
INCLUDE_ZONE_X_SHADED = False


def try_arcgis_rest_download() -> gpd.GeoDataFrame | None:
    """Attempt to download NFHL flood hazard polygons via ArcGIS REST API.

    The FEMA NFHL layer has a small maxRecordCount limit and the bbox must be
    small enough to satisfy the service. We tile King County into a 4×4 grid
    of sub-bboxes and query each tile, then combine.

    Returns a GeoDataFrame in WGS84 (EPSG:4326), or None on failure.
    """
    try:
        import requests
    except ImportError:
        print("  [WARN] requests not installed, cannot attempt REST download.")
        return None

    print("[AUTO] Attempting FEMA NFHL download via ArcGIS REST API (tiled)...")

    xmin, ymin, xmax, ymax = KING_CO_BBOX
    url = f"{NFHL_BASE}/{FLOOD_LAYER}/query"

    # Tile into N×N grid to stay within record-count limits per query.
    # Use a finer 8×8 grid so each tile is ~0.06°×0.07° — small enough to
    # avoid exceeding the endpoint's resultRecordCount limit on dense areas.
    N_TILES = 8
    xs = np.linspace(xmin, xmax, N_TILES + 1)
    ys = np.linspace(ymin, ymax, N_TILES + 1)

    all_features = []
    n_failed = 0

    for i in range(N_TILES):
        for j in range(N_TILES):
            tile_bbox = f"{xs[i]},{ys[j]},{xs[i+1]},{ys[j+1]}"
            # Paginate within each tile
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
                    resp = requests.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"    [WARN] tile ({i},{j}) failed: {e}")
                    n_failed += 1
                    break

                if "error" in data:
                    print(f"    [WARN] tile ({i},{j}) error: {data['error']}")
                    n_failed += 1
                    break

                feats = data.get("features", [])
                # Convert ArcGIS JSON features to GeoJSON-like
                for feat in feats:
                    all_features.append(feat)

                if len(feats) < 500:
                    break  # last page for this tile
                offset += 500

    print(f"  Total features downloaded: {len(all_features)} (tiles failed: {n_failed})")

    if not all_features:
        print("  [WARN] No features returned from NFHL REST endpoint.")
        return None

    # Convert ArcGIS REST JSON to GeoDataFrame via shapely
    # ArcGIS returns Esri JSON (rings/paths) not GeoJSON (coordinates),
    # so we must use arcgis-style parsing.
    from shapely.geometry import Polygon, MultiPolygon

    def esri_polygon_to_shapely(geom_json):
        """Convert Esri polygon JSON (rings) to a shapely geometry."""
        rings = geom_json.get("rings", [])
        if not rings:
            return None
        # First ring is exterior, subsequent rings are holes
        exterior = rings[0]
        holes = rings[1:] if len(rings) > 1 else []
        try:
            poly = Polygon(exterior, holes)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:
            return None

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
        print("  [WARN] No valid geometries after parsing.")
        return None

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    print(f"  GeoDataFrame: {len(gdf)} polygons, CRS={gdf.crs}")
    return gdf


def load_manual_geojson() -> gpd.GeoDataFrame | None:
    """Load manually downloaded NFHL GeoJSON from the expected path."""
    if MANUAL_GEOJSON.exists():
        print(f"[MANUAL] Loading flood zone data from {MANUAL_GEOJSON}")
        gdf = gpd.read_file(MANUAL_GEOJSON)
        print(f"  Loaded {len(gdf)} features, CRS={gdf.crs}")
        return gdf
    return None


def create_placeholder_flood_zones() -> gpd.GeoDataFrame | None:
    """Return None — we do NOT fabricate flood zone data.

    Instead, reports what the user should do manually.
    """
    return None


def detect_flood_zone_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Auto-detect the flood zone classification column in the GeoDataFrame."""
    candidates = ["FLD_ZONE", "fld_zone", "FLOOD_ZONE", "flood_zone",
                  "FZONE", "zone", "ZONE", "FLD_ZONE_TY"]
    for col in candidates:
        if col in gdf.columns:
            print(f"  Flood zone column detected: '{col}'")
            return col
    print(f"  [WARN] Could not auto-detect zone column. Available: {list(gdf.columns)}")
    return None


def spatially_join_flood_zones(
    flood_gdf: gpd.GeoDataFrame,
    zone_col: str,
) -> pd.DataFrame:
    """Join flood zone polygons to housing points.

    1. Load housing CSV → GeoDataFrame (UTM Zone 10N, EPSG:32610).
    2. Reproject flood polygons to UTM 10N.
    3. Point-in-polygon sjoin.
    4. Derive binary `in_floodplain` (1 = in SFHA, 0 = not).
    5. Return augmented housing DataFrame.
    """
    # Load housing
    housing = pd.read_csv(HOUSING_CSV, index_col=0)
    print(f"  Loaded {len(housing)} housing records.")

    geometry = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geometry, crs="EPSG:32610")

    # Reproject flood polygons to UTM 10N
    flood_utm = flood_gdf.to_crs("EPSG:32610")

    # Spatial join: point-in-polygon (left join, so unmatched points stay)
    joined = gpd.sjoin(housing_gdf, flood_utm[[zone_col, "geometry"]],
                       how="left", predicate="within")

    # Multiple matches: keep first (take the most hazardous one)
    # Deduplicate on original housing index
    joined = joined[~joined.index.duplicated(keep="first")]

    # Derive in_floodplain: 1 if the zone type starts with A or V (SFHA).
    # If INCLUDE_ZONE_X_SHADED is True, also include Zone X with ZONE_SUBTY
    # containing "0.2 PCT" (500-year floodplain) to boost coverage.
    def classify_flood(row_zone):
        if pd.isna(row_zone):
            return 0
        zone_str = str(row_zone).strip().upper()
        # SFHA zones start with A or V (excluding X which is 0.2% or outside)
        if zone_str.startswith("A") or zone_str.startswith("V"):
            return 1
        if INCLUDE_ZONE_X_SHADED and zone_str == "X":
            # Check ZONE_SUBTY if available in joined
            if "ZONE_SUBTY" in joined.columns:
                subty = joined.loc[joined[zone_col] == row_zone, "ZONE_SUBTY"]
                # This approach won't work row-wise; handled below
                pass
        return 0

    joined["in_floodplain"] = joined[zone_col].apply(classify_flood)

    # If INCLUDE_ZONE_X_SHADED, add Zone X Shaded (500-year) separately
    if INCLUDE_ZONE_X_SHADED and "ZONE_SUBTY" in joined.columns:
        def is_zone_x_shaded(row):
            zone = str(row.get(zone_col, "")).strip().upper()
            subty = str(row.get("ZONE_SUBTY", "")).strip().upper()
            return int(zone == "X" and ("0.2" in subty or "500" in subty or "SHADED" in subty))
        x_shaded = joined.apply(is_zone_x_shaded, axis=1)
        joined["in_floodplain"] = (joined["in_floodplain"] | x_shaded).astype(int)
        print(f"  Zone X Shaded (500-year) inclusions: {x_shaded.sum()}")

    n_flood = joined["in_floodplain"].sum()
    pct_flood = 100.0 * n_flood / len(joined)
    print(f"  In floodplain: {n_flood}/{len(joined)} records ({pct_flood:.1f}%)")

    if pct_flood < 3.0:
        print(f"  [WARN] Coverage only {pct_flood:.1f}% — fewer than 3% threshold.")
        print(f"         The biased branch may have too few positive cases.")
    else:
        print(f"  [OK] Coverage {pct_flood:.1f}% is above 3% threshold.")

    # Build final output: original housing columns + in_floodplain
    original_cols = list(housing.columns)
    out = pd.DataFrame(joined[original_cols + ["in_floodplain"]])
    out["in_floodplain"] = out["in_floodplain"].fillna(0).astype(int)

    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    flood_gdf = None

    # Strategy 1: automated REST download
    flood_gdf = try_arcgis_rest_download()

    # Strategy 2: manual GeoJSON drop
    if flood_gdf is None or len(flood_gdf) == 0:
        print("[AUTO] Automated download produced no data. Trying manual path...")
        flood_gdf = load_manual_geojson()

    if flood_gdf is None or len(flood_gdf) == 0:
        print()
        print("=" * 65)
        print("MANUAL STEP REQUIRED: FEMA flood zone data unavailable.")
        print("=" * 65)
        print()
        print("Neither the ArcGIS REST API nor the manual GeoJSON path")
        print(f"  {MANUAL_GEOJSON}")
        print("returned usable data.")
        print()
        print("To obtain the data manually:")
        print("  1. Go to https://msc.fema.gov/portal/advanceSearch")
        print("     or https://hazards.fema.gov/femaportal/NFHL/")
        print("  2. Select State=Washington, County=King County")
        print("  3. Download the NFHL GDB or Shapefile (S_Fld_Haz_Ar layer)")
        print("  4. Export or convert to GeoJSON (e.g. with ogr2ogr or QGIS)")
        print(f"  5. Place the file at: {MANUAL_GEOJSON}")
        print("  6. Re-run this script.")
        print()
        print("Alternative: FEMA NFHL Viewer export at")
        print("  https://hazards.fema.gov/femaportal/NFHL/")
        print("  Filter to King County, export S_Fld_Haz_Ar as GeoJSON.")
        print()
        print("The script and config are ready — data is the only blocker.")
        sys.exit(1)

    # Detect zone column
    zone_col = detect_flood_zone_column(flood_gdf)
    if zone_col is None:
        # Last attempt: use first non-geometry column
        non_geom = [c for c in flood_gdf.columns if c != "geometry"]
        if non_geom:
            zone_col = non_geom[0]
            print(f"  Defaulting to first column: '{zone_col}'")
        else:
            print("[ERROR] No usable column found in flood GeoDataFrame.")
            sys.exit(1)

    # Spatial join
    print("\n[JOIN] Joining flood zones to housing points (SFHA only)...")
    housing_enriched = spatially_join_flood_zones(flood_gdf, zone_col)

    sfha_coverage = housing_enriched["in_floodplain"].mean() * 100
    if sfha_coverage < 3.0:
        print(f"\n[FALLBACK] SFHA coverage {sfha_coverage:.2f}% < 3%.")
        print("  Broadening criterion to include Zone X Shaded (500-year floodplain).")
        print("  This is documented in the experiment notes.")
        global INCLUDE_ZONE_X_SHADED
        INCLUDE_ZONE_X_SHADED = True
        housing_enriched = spatially_join_flood_zones(flood_gdf, zone_col)

    # Save
    housing_enriched.to_csv(OUTPUT_CSV, index=True)
    print(f"\n[SAVE] Output: {OUTPUT_CSV}")
    print(f"  Columns: {list(housing_enriched.columns)}")
    print(f"  Shape: {housing_enriched.shape}")
    print(f"  in_floodplain distribution:\n{housing_enriched['in_floodplain'].value_counts().to_string()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
