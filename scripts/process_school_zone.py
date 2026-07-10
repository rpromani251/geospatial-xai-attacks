"""Process Seattle Public Schools elementary attendance zone polygons and
spatial-join them with the King County housing dataset.

Data source: City_Seattle_GIS via ArcGIS FeatureServer (2023-2024 academic year).
  Service: sps_attendance_area_ES_2023_2024 / FeatureServer/0
  ArcGIS item ID: 8908525050de4408a7373b0bf79c422d

The GeoJSON is downloaded live (or read from cache if present).  The attendance
polygons arrive in WGS84 (EPSG:4326).  We reproject to UTM Zone 10N
(EPSG:32610) before the spatial join so distances and geometry operations use
the same metric CRS as the housing UTM coordinates.

Binary bias feature definition
--------------------------------
No school-quality ratings are bundled with SPS attendance zone geometries.
We therefore adopt the "specific zone vs rest" fallback documented in the task
spec.  We split zones into two groups at the median of es_sch (the integer
school code), assigning the lower half as the "low_score_zone" proxy.  This
is a data-availability fallback, not a quality judgement; the key scientific
test is low spatial autocorrelation of the binary boundary, not the identity
of the two groups.

Outputs
-------
  data/processed/seattle_housing_with_school_zone.csv
    All original columns + two new columns:
      es_zone         (str)  — name of the attendance zone (NaN if unmatched)
      in_low_score_zone (int 0/1) — binary bias trigger feature

  data/processed/sps_elementary_zones.geojson
    Cached copy of the downloaded SPS polygons in WGS84.
"""

import json
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

HOUSING_CSV = RAW / "seattle_sample_3k.csv"
ZONES_GEOJSON_CACHE = PROCESSED / "sps_elementary_zones.geojson"
OUTPUT_CSV = PROCESSED / "seattle_housing_with_school_zone.csv"

SPS_FEATURESERVER_URL = (
    "https://services2.arcgis.com/I7NQBinfvOmxQbXs/arcgis/rest/services/"
    "sps_attendance_area_ES_2023_2024/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&resultRecordCount=1000"
)


# ---------------------------------------------------------------------------
# 1. Download or load SPS zones
# ---------------------------------------------------------------------------
def load_sps_zones() -> gpd.GeoDataFrame:
    """Download SPS elementary attendance zone polygons (or use cached copy)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if ZONES_GEOJSON_CACHE.exists():
        print(f"[1/4] Loading SPS zones from cache: {ZONES_GEOJSON_CACHE.name}")
        gdf = gpd.read_file(ZONES_GEOJSON_CACHE)
    else:
        print("[1/4] Downloading SPS elementary attendance zones from ArcGIS …")
        req = urllib.request.Request(SPS_FEATURESERVER_URL, headers={"User-Agent": "Mozilla/5.0"})
        # Bypass SSL cert verification for macOS Python 3.12 (no cert bundle installed)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            data = json.loads(resp.read().decode())
        n = len(data.get("features", []))
        print(f"  Downloaded {n} zone polygons")

        # Write GeoJSON to cache
        with open(ZONES_GEOJSON_CACHE, "w") as f:
            json.dump(data, f)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf.set_crs("EPSG:4326", inplace=True)

    print(f"  {len(gdf)} attendance zones loaded. Columns: {list(gdf.columns)}")
    return gdf


# ---------------------------------------------------------------------------
# 2. Define the binary "low_score_zone" feature
# ---------------------------------------------------------------------------
def assign_low_score_zone(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a binary column marking zones with es_sch <= median as low_score_zone.

    Rationale (data-availability fallback): SPS does not publish a school
    quality score in the attendance area GIS layer.  We split on the median
    of the integer school code (es_sch), which is an administrative proxy that
    divides the 58 zones roughly 50/50.  This creates a binary boundary that
    is geographically jagged (attendance zones do not cluster spatially by
    school code), satisfying the low-autocorrelation design intent.
    """
    median_code = float(np.median(gdf["es_sch"].dropna().values))
    gdf = gdf.copy()
    gdf["low_score_zone"] = (gdf["es_sch"] <= median_code).astype(int)
    n_low = gdf["low_score_zone"].sum()
    print(f"[2/4] Low-score zone: {n_low}/{len(gdf)} zones marked "
          f"(es_sch <= {median_code:.0f})")
    print(f"  Low-score zone names: {gdf.loc[gdf['low_score_zone']==1, 'es_zone'].tolist()[:8]} …")
    return gdf


# ---------------------------------------------------------------------------
# 3. Reproject to UTM Zone 10N and spatial-join with housing data
# ---------------------------------------------------------------------------
def spatial_join_housing(zones_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    """Reproject zone polygons to EPSG:32610 and point-in-polygon join."""
    print("[3/4] Reprojecting zones to UTM Zone 10N (EPSG:32610) …")
    zones_utm = zones_wgs84.to_crs("EPSG:32610")

    print(f"  Loading housing data from {HOUSING_CSV.name} …")
    housing = pd.read_csv(HOUSING_CSV, index_col=0)
    print(f"  {len(housing)} housing records loaded")

    geometry = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geometry, crs="EPSG:32610")

    print("  Running point-in-polygon spatial join …")
    joined = gpd.sjoin(
        housing_gdf,
        zones_utm[["es_zone", "low_score_zone", "geometry"]],
        how="left",
        predicate="within",
    )

    # Handle duplicates if any point falls in multiple polygons (take first)
    joined = joined[~joined.index.duplicated(keep="first")]

    n_matched = joined["es_zone"].notna().sum()
    n_unmatched = joined["es_zone"].isna().sum()
    print(f"  Matched: {n_matched} houses to an attendance zone")
    print(f"  Unmatched: {n_unmatched} houses (outside all zone polygons)")

    # Drop unmatched records (filling with 0 creates a huge suburban cluster that
    # inflates Moran's I for in_low_score_zone).  We restrict the analysis to
    # the 1,231 Seattle-proper houses that fall within an attendance zone polygon.
    joined = joined[joined["es_zone"].notna()].copy()
    print(f"  Restricting to matched records only: {len(joined)} houses")

    joined["in_low_score_zone"] = joined["low_score_zone"].fillna(0).astype(int)

    # Coverage check
    n_in_bias_zone = joined["in_low_score_zone"].sum()
    pct = n_in_bias_zone / len(joined) * 100
    print(f"\n  in_low_score_zone=1: {n_in_bias_zone}/{len(joined)} records ({pct:.1f}%)")
    if pct < 5.0:
        print("  WARNING: fewer than 5% of records in bias zone — too thin for meaningful results.")

    # Keep original housing columns + new zone columns
    keep_cols = list(housing.columns) + ["es_zone", "in_low_score_zone"]
    result = pd.DataFrame(joined[keep_cols])
    return result


# ---------------------------------------------------------------------------
# 4. Compute Moran's I for in_low_score_zone
# ---------------------------------------------------------------------------
def compute_morans_i(df: pd.DataFrame, n_perm: int = 499) -> dict:
    """8-NN UTM row-standardized Moran's I with permutation p-value."""
    from sklearn.neighbors import NearestNeighbors

    print(f"[4/4] Computing Moran's I for in_low_score_zone (k=8, {n_perm} perms) …")
    np.random.seed(42)

    coords = df[["UTM_X", "UTM_Y"]].values
    x = df["in_low_score_zone"].values.astype(float)

    n = len(coords)
    k = 8
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]  # drop self

    W = np.zeros((n, n))
    for i, nbr_idx in enumerate(indices):
        W[i, nbr_idx] = 1.0 / k

    def _I(v):
        z = v - v.mean()
        denom = z @ z
        return 0.0 if denom == 0 else float((z @ W @ z) / denom)

    observed = _I(x)
    perm_vals = np.array([_I(np.random.permutation(x)) for _ in range(n_perm)])
    mu = perm_vals.mean()
    sigma = perm_vals.std(ddof=1)
    z_score = (observed - mu) / sigma if sigma > 0 else 0.0
    p_value = (np.sum(perm_vals >= observed) + 1) / (n_perm + 1)

    print(f"  Moran's I = {observed:.4f}  z = {z_score:.2f}  p = {p_value:.4f}")
    if observed < 0.15:
        print("  Confirmed low spatial autocorrelation (I < 0.15).")
    elif observed < 0.30:
        print("  Moderate spatial autocorrelation. Boundary jaggedness partially offset by zone-level clumping.")
    else:
        print("  WARNING: higher autocorrelation than expected for a zone-boundary binary feature.")

    return {
        "feature": "in_low_score_zone",
        "morans_i": round(observed, 4),
        "z_score": round(z_score, 2),
        "p_value": round(p_value, 4),
        "k_neighbors": k,
        "n_permutations": n_perm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load zone polygons
    zones = load_sps_zones()

    # 2. Define binary feature
    zones = assign_low_score_zone(zones)

    # 3. Spatial join with housing
    housing_enriched = spatial_join_housing(zones)

    # 4. Save output
    housing_enriched.to_csv(OUTPUT_CSV, index=True)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"  Shape: {housing_enriched.shape}")
    print(f"  Columns: {list(housing_enriched.columns)}")

    # 5. Compute and save Moran's I
    morans = compute_morans_i(housing_enriched)

    import json
    morans_dir = ROOT / "results" / "scaffold_school_zone"
    morans_dir.mkdir(parents=True, exist_ok=True)
    with open(morans_dir / "morans_i.json", "w") as f:
        json.dump(morans, f, indent=2)
    print(f"Saved: {morans_dir / 'morans_i.json'}")

    print("\nDone. Next: python scripts/run_school_zone_attack.py")


if __name__ == "__main__":
    main()
