"""
Build expanded King County housing dataset from KC Assessor parcel data.

Data sources:
  - KC PropertyInfo MapServer Layer 3: Property sales (last 3 yrs) with geometry
    https://gismaps.kingcounty.gov/arcgis/rest/services/Property/KingCo_PropertyInfo/MapServer/3
  - KC Assessor Residential Building: EXTR_ResBldg.csv (sqft, grade, bedrooms, age)
    https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip
  - KC Assessor Parcel: EXTR_Parcel.csv (SqFtLot, waterfront, view flags)
    https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip
  - FEMA NFHL: data/raw/fema_nfhl_king_county.geojson (cached)
  - Demographics: data/processed/king_county_tracts_demographics.geojson

Output:
  data/raw/kc_parcels/kc_sales_raw.csv
  data/raw/kc_parcels/EXTR_ResBldg.csv   (symlink if exists)
  data/processed/kc_housing_with_flood.csv

Feature imputations documented below.
"""

import sys
import io
import json
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPolygon
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KC_PARCELS = RAW / "kc_parcels"
PROCESSED = ROOT / "data" / "processed"

FEMA_GEOJSON = RAW / "fema_nfhl_king_county.geojson"
DEMOGRAPHICS_GEOJSON = PROCESSED / "king_county_tracts_demographics.geojson"
OUTPUT_CSV = PROCESSED / "kc_housing_with_flood.csv"

CURRENT_YEAR = 2026

SALES_URL = ("https://gismaps.kingcounty.gov/arcgis/rest/services/"
             "Property/KingCo_PropertyInfo/MapServer/3/query")


# ── Step 1: Fetch sales from ArcGIS REST ─────────────────────────────────────

def fetch_sales_from_rest() -> pd.DataFrame:
    """Paginate through all KC property sales (SalePrice 50K-10M) with geometry."""
    print("[1] Fetching KC property sales from ArcGIS REST...")
    all_records = []
    offset = 0
    PAGE = 1000

    while True:
        params = {
            "where": "SalePrice > 50000 AND SalePrice < 10000000",
            "outFields": "PIN,SalePrice,SaleDate",
            "outSR": "4326",
            "f": "json",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": PAGE,
        }
        try:
            r = requests.get(SALES_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [WARN] page at offset {offset} failed: {e}")
            break

        if "error" in data:
            print(f"  [ERROR] {data['error']}")
            break

        feats = data.get("features", [])
        for feat in feats:
            attrs = feat.get("attributes", {})
            geom = feat.get("geometry")
            if geom is None:
                continue
            # Polygon — take centroid of first ring
            rings = geom.get("rings", [])
            if not rings:
                continue
            ring = rings[0]
            if len(ring) < 3:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            attrs["lon"] = sum(xs) / len(xs)
            attrs["lat"] = sum(ys) / len(ys)
            all_records.append(attrs)

        print(f"  offset={offset}: got {len(feats)} records (total so far: {len(all_records)})")
        if len(feats) < PAGE:
            break
        offset += PAGE
        time.sleep(0.1)  # be polite

    df = pd.DataFrame(all_records)
    print(f"  Total sales fetched: {len(df)}")
    return df


# ── Step 2: Download and join Residential Building data ───────────────────────

def download_res_bldg() -> pd.DataFrame:
    """Download EXTR_ResBldg.csv if not already cached, return DataFrame."""
    cached = KC_PARCELS / "EXTR_ResBldg.csv"
    if cached.exists():
        print(f"[2] Loading cached ResBldg from {cached}")
        return pd.read_csv(cached, encoding="latin-1")

    print("[2] Downloading Residential Building data...")
    r = requests.get(
        "https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip",
        timeout=180, stream=True,
    )
    r.raise_for_status()
    chunks = [c for c in r.iter_content(65536)]
    data = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract("EXTR_ResBldg.csv", str(KC_PARCELS))
    return pd.read_csv(cached, encoding="latin-1")


def download_parcel_csv() -> pd.DataFrame:
    """Download EXTR_Parcel.csv if not cached (has SqFtLot, waterfront, view flags)."""
    cached = KC_PARCELS / "EXTR_Parcel.csv"
    if cached.exists():
        print(f"[2b] Loading cached Parcel from {cached}")
        return pd.read_csv(cached, encoding="latin-1")

    print("[2b] Downloading Parcel CSV...")
    r = requests.get(
        "https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip",
        timeout=180, stream=True,
    )
    r.raise_for_status()
    chunks = [c for c in r.iter_content(65536)]
    data = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract("EXTR_Parcel.csv", str(KC_PARCELS))
    return pd.read_csv(cached, encoding="latin-1")


# ── Step 3: Feature engineering ───────────────────────────────────────────────

def engineer_features(sales: pd.DataFrame, res_bldg: pd.DataFrame, parcel: pd.DataFrame) -> pd.DataFrame:
    """
    Join and engineer features to match seattle_sample_3k.csv schema:
      bathrooms, sqft_living, sqft_lot, grade, condition, waterfront,
      view, age, UTM_X, UTM_Y, log_price, in_floodplain (added later)

    Feature imputations:
      - bathrooms:  BathFullCount + 0.5*BathHalfCount + 0.75*Bath3qtrCount
      - sqft_living: SqFtTotLiving
      - sqft_lot:   SqFtLot from EXTR_Parcel (join on Major+Minor=PIN)
      - grade:      BldgGrade (1-13 scale, same as KC assessor)
      - condition:  Condition (1-5 scale)
      - waterfront: WfntLocation != 0 → 1, else 0
      - view:       max of all view flags (Olympics, SeattleSkyline, PugetSound,
                    LakeWashington, LakeSammamish, SmallLakeRiverCreek, OtherView)
                    mapped to 0-4 scale by max(sum, 4)
      - age:        CURRENT_YEAR - YrBuilt
      - UTM_X/Y:    convert lat/lon (WGS84) → UTM Zone 10N (EPSG:32610)
      - log_price:  log(SalePrice)
    """
    print("[3] Engineering features...")

    # Normalize PIN in sales: PIN is 10-digit string (Major=7, Minor=4 but padded)
    # ResBldg has Major (int) and Minor (int). PIN = Major (zero-padded to 6) + Minor (zero-padded to 4)
    # KC PIN format: first 6 chars = Major, last 4 = Minor
    res_bldg = res_bldg.copy()
    res_bldg["PIN"] = (
        res_bldg["Major"].astype(str).str.zfill(6) +
        res_bldg["Minor"].astype(str).str.zfill(4)
    )

    parcel = parcel.copy()
    parcel["PIN"] = (
        parcel["Major"].astype(str).str.zfill(6) +
        parcel["Minor"].astype(str).str.zfill(4)
    )

    # Keep only primary buildings (BldgNbr==1)
    res_primary = res_bldg[res_bldg["BldgNbr"] == 1].copy()

    # Deduplicate parcel (keep first)
    parcel_dedup = parcel.drop_duplicates("PIN")
    res_dedup = res_primary.drop_duplicates("PIN")

    print(f"  Sales: {len(sales)}, ResBldg primary: {len(res_dedup)}, Parcel: {len(parcel_dedup)}")

    # Join sales → ResBldg on PIN
    merged = sales.merge(res_dedup, on="PIN", how="inner")
    print(f"  After join sales+ResBldg: {len(merged)}")

    # Join → Parcel for sqft_lot, waterfront, view
    parcel_cols = ["PIN", "SqFtLot", "WfntLocation", "WfntFootage",
                   "Olympics", "SeattleSkyline", "PugetSound",
                   "LakeWashington", "LakeSammamish", "SmallLakeRiverCreek", "OtherView"]
    parcel_sub = parcel_dedup[parcel_cols].copy()
    merged = merged.merge(parcel_sub, on="PIN", how="left")
    print(f"  After join +Parcel: {len(merged)}")

    # ── Compute target ──────────────────────────────────────────────────────
    merged["log_price"] = np.log(merged["SalePrice"].clip(lower=1))

    # ── bathrooms ───────────────────────────────────────────────────────────
    merged["bathrooms"] = (
        merged["BathFullCount"].fillna(0) +
        0.75 * merged["Bath3qtrCount"].fillna(0) +
        0.5 * merged["BathHalfCount"].fillna(0)
    )

    # ── sqft_living ─────────────────────────────────────────────────────────
    merged["sqft_living"] = merged["SqFtTotLiving"].fillna(0)

    # ── sqft_lot ────────────────────────────────────────────────────────────
    merged["sqft_lot"] = merged["SqFtLot"].fillna(merged["sqft_living"] * 3)
    # Imputation note: ~5% of parcels lack SqFtLot; fill with 3× living area as proxy.

    # ── grade ────────────────────────────────────────────────────────────────
    # BldgGrade is on KC 1-13 scale (same scale as seattle_sample_3k)
    merged["grade"] = merged["BldgGrade"].fillna(7).clip(1, 13)

    # ── condition ───────────────────────────────────────────────────────────
    # Condition: 1=Poor, 2=Fair, 3=Average, 4=Good, 5=Very Good (same scale)
    merged["condition"] = merged["Condition"].fillna(3).clip(1, 5)

    # ── waterfront ──────────────────────────────────────────────────────────
    # WfntLocation: 0=none, else waterfront code
    merged["waterfront"] = (merged["WfntLocation"].fillna(0) != 0).astype(int)

    # ── view ────────────────────────────────────────────────────────────────
    # Sum of binary view indicators (0 or 1 each); cap at 4 to match
    # original scale (0=None, 1=Fair, 2=Average, 3=Good, 4=Excellent)
    view_flags = ["Olympics", "SeattleSkyline", "PugetSound",
                  "LakeWashington", "LakeSammamish", "SmallLakeRiverCreek", "OtherView"]
    for col in view_flags:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    merged["view"] = merged[view_flags].sum(axis=1).clip(0, 4).astype(int)

    # ── age ─────────────────────────────────────────────────────────────────
    yr_built = pd.to_numeric(merged["YrBuilt"], errors="coerce").fillna(1970)
    yr_built = yr_built.clip(lower=1800, upper=CURRENT_YEAR)
    merged["age"] = (CURRENT_YEAR - yr_built).clip(lower=0)

    # ── UTM coordinates ──────────────────────────────────────────────────────
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    utm_x, utm_y = transformer.transform(merged["lon"].values, merged["lat"].values)
    merged["UTM_X"] = utm_x
    merged["UTM_Y"] = utm_y

    # ── Filter unreasonable values ───────────────────────────────────────────
    before = len(merged)
    merged = merged[merged["sqft_living"] > 100]
    merged = merged[merged["age"] >= 0]
    merged = merged[merged["log_price"] > 0]
    merged = merged[merged["UTM_X"].between(300000, 700000)]
    merged = merged[merged["UTM_Y"].between(5100000, 5350000)]
    print(f"  After quality filters: {len(merged)} (removed {before - len(merged)})")

    # ── Keep key columns ─────────────────────────────────────────────────────
    keep = ["PIN", "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
            "waterfront", "view", "age", "UTM_X", "UTM_Y", "log_price", "lat", "lon"]
    result = merged[keep].copy()
    result = result.dropna(subset=["bathrooms", "sqft_living", "grade", "UTM_X", "UTM_Y"])
    print(f"  Final clean dataset: {len(result)} parcels")
    return result


# ── Step 4: Spatial join demographics ────────────────────────────────────────

def join_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial join with tract demographics to attach pct_black_latino."""
    print("[4] Joining demographics...")
    tracts = gpd.read_file(DEMOGRAPHICS_GEOJSON)
    print(f"  Loaded {len(tracts)} demographic tracts")

    geometry = [Point(x, y) for x, y in zip(df["UTM_X"], df["UTM_Y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32610")
    tracts_utm = tracts.to_crs("EPSG:32610")

    joined = gpd.sjoin(gdf, tracts_utm[["pct_black_latino", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    n_matched = joined["pct_black_latino"].notna().sum()
    print(f"  Demographic join: {n_matched}/{len(df)} matched ({100*n_matched/len(df):.1f}%)")

    df = df.copy()
    df["pct_black_latino"] = joined["pct_black_latino"].values
    df["pct_black_latino"] = df["pct_black_latino"].fillna(df["pct_black_latino"].median())
    return df


# ── Step 5: Spatial join FEMA flood zones ────────────────────────────────────

def join_flood_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial join with FEMA NFHL to derive in_floodplain binary flag."""
    print("[5] Joining FEMA flood zones...")
    if not FEMA_GEOJSON.exists():
        raise FileNotFoundError(f"FEMA GeoJSON not found at {FEMA_GEOJSON}. Run process_flood_zone.py first.")

    flood_gdf = gpd.read_file(FEMA_GEOJSON)
    print(f"  Loaded {len(flood_gdf)} FEMA polygons, CRS={flood_gdf.crs}")

    # Detect zone column
    zone_col = None
    for col in ["FLD_ZONE", "fld_zone", "FLOOD_ZONE"]:
        if col in flood_gdf.columns:
            zone_col = col
            break
    if zone_col is None:
        raise ValueError(f"No zone column found. Columns: {list(flood_gdf.columns)}")

    geometry = [Point(x, y) for x, y in zip(df["UTM_X"], df["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32610")
    flood_utm = flood_gdf.to_crs("EPSG:32610")

    joined = gpd.sjoin(housing_gdf, flood_utm[[zone_col, "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    def classify_flood(zone_str):
        if pd.isna(zone_str):
            return 0
        z = str(zone_str).strip().upper()
        if z.startswith("A") or z.startswith("V"):
            return 1
        return 0

    joined["in_floodplain"] = joined[zone_col].apply(classify_flood)

    n_flood = joined["in_floodplain"].sum()
    n_total = len(joined)
    pct = 100.0 * n_flood / n_total

    # Also compute 500-yr coverage (Zone X)
    n_x = (joined[zone_col].fillna("").str.upper() == "X").sum()
    pct_x = 100.0 * n_x / n_total

    print(f"  100-yr SFHA:  {n_flood}/{n_total} ({pct:.2f}%)")
    print(f"  500-yr Zone X: {n_x}/{n_total} ({pct_x:.2f}%)")

    df = df.copy()
    df["in_floodplain"] = joined["in_floodplain"].values
    df["in_floodplain"] = df["in_floodplain"].fillna(0).astype(int)

    # Store coverage stats as a side-effect output
    coverage = {
        "total_parcels": int(n_total),
        "n_100yr_sfha": int(n_flood),
        "pct_100yr_sfha": float(pct),
        "n_500yr_zone_x": int(n_x),
        "pct_500yr_zone_x": float(pct_x),
    }
    cov_path = ROOT / "results" / "scaffold_parcel_flood" / "flood_coverage.json"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cov_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"  Coverage saved to {cov_path}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    KC_PARCELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch sales (or use cache)
    sales_cache = KC_PARCELS / "kc_sales_raw.csv"
    if sales_cache.exists():
        print(f"[1] Loading cached sales from {sales_cache}")
        sales = pd.read_csv(sales_cache)
    else:
        sales = fetch_sales_from_rest()
        sales.to_csv(sales_cache, index=False)
        print(f"  Saved raw sales to {sales_cache}")

    # Step 2: Residential Building data
    res_bldg = download_res_bldg()
    parcel_csv = download_parcel_csv()

    # Save copies to kc_parcels if not there
    rb_dest = KC_PARCELS / "EXTR_ResBldg.csv"
    if not rb_dest.exists():
        import shutil
        shutil.copy("/tmp/kc_assessor/EXTR_ResBldg.csv", rb_dest)
    pc_dest = KC_PARCELS / "EXTR_Parcel.csv"
    if not pc_dest.exists():
        import shutil
        shutil.copy("/tmp/kc_assessor/EXTR_Parcel.csv", pc_dest)

    # Step 3: Feature engineering
    df = engineer_features(sales, res_bldg, parcel_csv)

    # Step 4: Demographics join
    df = join_demographics(df)

    # Step 5: Flood zone join
    df = join_flood_zones(df)

    # Step 6: Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(df)} parcels to {OUTPUT_CSV}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  in_floodplain distribution:\n{df['in_floodplain'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
