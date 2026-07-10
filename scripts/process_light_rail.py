"""
Process Sound Transit Link light rail station data and spatial-join with
Seattle housing data to produce binary `near_light_rail` column.

Feature: within 1000m of a Sound Transit Link light rail station (any status).

Data source: ArcGIS Open Data Hub — Sound Transit Link Light Station Point
  Dataset ID: 1a047b18afc842e1ac6a6075d525e539_0
  CRS: EPSG:2926 (WA State Plane North, feet)

Strategy:
  1. Download station GeoJSON from ArcGIS Open Data Hub.
  2. If download fails, fall back to hardcoded known station coordinates.
  3. Reproject to EPSG:32610 (UTM Zone 10N).
  4. Apply 1000m point buffer, spatial join with housing points.

Run from repo root:
    python3 scripts/process_light_rail.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

HOUSING_CSV = RAW / "seattle_sample_3k.csv"
OUTPUT_CSV = PROCESSED / "seattle_housing_with_light_rail.csv"
STATIONS_GEOJSON = RAW / "sound_transit_link_stations.geojson"

# ArcGIS Open Data URL for Sound Transit Link stations
STATIONS_URL = "https://opendata.arcgis.com/datasets/1a047b18afc842e1ac6a6075d525e539_0.geojson"

# Buffer distance in meters
BUFFER_M = 1000

# Fallback: known Sound Transit Link stations in WA State Plane feet (EPSG:2926)
# Source: Sound Transit official station list + ArcGIS dataset
# Includes COMPLETE and under-construction (CONST) stations as of 2024
FALLBACK_STATIONS_SP_FEET = [
    ("Westlake Station", 1269613.625, 226704.922),
    ("Symphony Station", 1269759.5, 225329.594),
    ("Pioneer Square Station", 1270907.0, 223391.797),
    ("Int'l Dist / Chinatown Station", 1271672.375, 221842.75),
    ("Stadium Station", 1271823.5, 219195.984),
    ("SODO Station", 1271698.625, 215537.641),
    ("Beacon Hill Station", 1275599.08, 214825.093),
    ("Mount Baker Station", 1278998.625, 213746.75),
    ("Columbia City Station", 1280113.124, 207590.346),
    ("Othello Station", 1282711.125, 199607.438),
    ("Rainier Beach Station", 1283140.75, 193910.484),
    ("Tukwila International Blvd Station", 1280612.5, 172665.578),
    ("Airport / SeaTac Station", 1278281.43, 165905.888),
    ("Angle Lake Station", 1277922.219, 157638.893),
    ("Northgate Station", 1272340.586, 260018.027),
    ("Roosevelt Station", 1275186.28, 250319.199),
    ("U District Station", 1275526.198, 244368.223),
    ("Capitol Hill Station", 1273741.777, 229356.583),
    ("University of Washington Station", 1278007.826, 240495.012),
    ("Shoreline South/148th Station", 1273330.525, 272063.093),
    ("Shoreline North/185th Station", 1274114.264, 282255.97),
    ("Mountlake Terrace Station", 1276242.955, 289847.647),
    ("Lynnwood City Center Station", 1281380.38, 300926.775),
    ("Mercer Island Station", 1294980.511, 217698.853),
    ("Judkins Park Station", 1277409.111, 218793.141),
    ("South Bellevue Station", 1305538.93, 216910.587),
    ("East Main Station", 1305492.754, 224803.111),
    ("Bellevue Downtown Station", 1305306.924, 227373.086),
    ("Wilburton Station", 1307376.864, 228337.789),
    ("Spring District/120th Station", 1308693.38, 230432.824),
    ("Bel-Red/130th Station", 1311886.418, 230623.587),
    ("Overlake Village Station", 1318548.172, 234836.617),
    ("Redmond Technology Station", 1319905.126, 237926.248),
    ("South 272nd Street Station", 1277563.564, 134540.853),
    ("Federal Way Transit Center Station", 1275658.51, 118701.949),
    ("Kent/Des Moines Station", 1278509.46, 145511.471),
]


def download_stations() -> gpd.GeoDataFrame | None:
    """Download Sound Transit Link stations from ArcGIS Open Data Hub."""
    try:
        import requests
    except ImportError:
        print("  [WARN] requests not installed.")
        return None

    print(f"[AUTO] Downloading Sound Transit Link stations...")
    try:
        r = requests.get(STATIONS_URL, timeout=30)
        r.raise_for_status()
        import json
        d = r.json()
        features = d.get("features", [])
        print(f"  Downloaded {len(features)} features.")
        if not features:
            return None

        # Parse: CRS is EPSG:2926 (WA State Plane feet)
        transformer = Transformer.from_crs("EPSG:2926", "EPSG:32610", always_xy=True)
        records = []
        geoms = []
        for f in features:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue
            x_sp, y_sp = coords[0], coords[1]
            x_utm, y_utm = transformer.transform(x_sp, y_sp)
            records.append({"name": props.get("NAME", ""), "status": props.get("STATUS", "")})
            geoms.append(Point(x_utm, y_utm))

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:32610")
        print(f"  Converted {len(gdf)} stations to EPSG:32610.")
        return gdf
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return None


def load_fallback_stations() -> gpd.GeoDataFrame:
    """Use hardcoded station coordinates (WA State Plane → UTM 10N)."""
    print("[FALLBACK] Using hardcoded Sound Transit Link station coordinates.")
    transformer = Transformer.from_crs("EPSG:2926", "EPSG:32610", always_xy=True)
    records = []
    geoms = []
    for name, x_sp, y_sp in FALLBACK_STATIONS_SP_FEET:
        x_utm, y_utm = transformer.transform(x_sp, y_sp)
        records.append({"name": name, "status": "KNOWN"})
        geoms.append(Point(x_utm, y_utm))
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:32610")
    print(f"  Loaded {len(gdf)} stations.")
    return gdf


def compute_near_light_rail(
    stations_gdf: gpd.GeoDataFrame,
    buffer_m: float = BUFFER_M,
) -> pd.DataFrame:
    """Spatial join: near_light_rail = 1 if within buffer_m of any station."""
    # Load housing
    housing = pd.read_csv(HOUSING_CSV, index_col=0)
    print(f"  Housing records: {len(housing)}")

    geoms = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geoms, crs="EPSG:32610")

    # Buffer the stations
    station_buffer = stations_gdf.copy()
    station_buffer["geometry"] = stations_gdf.geometry.buffer(buffer_m)
    station_buffer = station_buffer.dissolve()  # merge all buffers
    print(f"  Buffered all stations by {buffer_m}m.")

    # Point-in-polygon join
    joined = gpd.sjoin(housing_gdf, station_buffer[["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    housing["near_light_rail"] = (~joined["index_right"].isna()).astype(int)

    n_near = housing["near_light_rail"].sum()
    pct_near = 100.0 * n_near / len(housing)
    print(f"  near_light_rail: {n_near}/{len(housing)} ({pct_near:.1f}%)")

    if pct_near < 4.0:
        print(f"  [WARN] Coverage {pct_near:.1f}% < 4% hard gate.")
    elif pct_near > 40.0:
        print(f"  [WARN] Coverage {pct_near:.1f}% > 40% hard gate.")
    else:
        print(f"  [OK] Coverage {pct_near:.1f}% is within [4%, 40%] gate.")

    return housing


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Try automated download
    stations_gdf = download_stations()

    # Fall back to hardcoded if download failed
    if stations_gdf is None or len(stations_gdf) == 0:
        stations_gdf = load_fallback_stations()

    # Save stations GeoJSON
    stations_gdf.to_file(STATIONS_GEOJSON, driver="GeoJSON")
    print(f"  Stations saved to {STATIONS_GEOJSON}")

    # Spatial join
    print(f"\n[JOIN] Joining {BUFFER_M}m buffer to housing points...")
    housing_enriched = compute_near_light_rail(stations_gdf, BUFFER_M)

    # Save
    housing_enriched.to_csv(OUTPUT_CSV, index=True)
    print(f"\n[SAVE] Output: {OUTPUT_CSV}")
    print(f"  Shape: {housing_enriched.shape}")
    print(f"  near_light_rail distribution:\n{housing_enriched['near_light_rail'].value_counts().to_string()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
