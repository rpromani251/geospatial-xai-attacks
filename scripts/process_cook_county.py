"""Build the Cook County analogue of the King County sample.

Inputs (downloaded separately; paths are arguments):
  --ccao  Cook_County_Assessor_s_Modeling_Data.csv: the Cook County Assessor's Office residential modeling data
          (2013-2018 sales with characteristics), as published in September 2019 and mirrored at
          github.com/mateomontoya/HCE_DS100_CookCo_AssessorHousingModel_HW (zip).
  --acs   acs2015_census_tract_data.csv: ACS 2011-2015 5-year tract estimates (DP05 shares), the widely mirrored
          Kaggle file (e.g. github.com/cyz-ai/infomin, data/; git blob 74853aec). Hispanic and non-Hispanic Black
          shares; for King County this reproduces our B03002 tract shares to within 0.001 on average.

Outputs (data/processed/):
  cook_county_housing_with_demographics.csv  3,000 single-family arm's-length sales from 2015, with eight
      structural features, UTM 16N coordinates, log10 price and the tract share of Black and Latino residents
  cook_county_tract_points.npz  UTM coordinates and tract share of every geocoded sale in the CCAO file (all
      years, one row per PIN): the coordinate-to-tract lookup used inside the location-design model

Feature mapping (King County name in brackets where it differs): bathrooms = full + half/2; sqft_living = building
square feet; sqft_lot = land square feet; quality = 4 - construction quality code [grade]; condition = 4 - repair
condition code; garage = garage indicator [waterfront]; fireplaces [view]; age.

  python -I scripts/process_cook_county.py --ccao PATH --acs PATH
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_COOK = ["bathrooms", "sqft_living", "sqft_lot", "quality", "condition", "garage", "fireplaces", "age"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccao", required=True)
    ap.add_argument("--acs", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    import pyproj
    cols = ["PIN", "Sale Year", "Modeling Group", "Pure Market Filter", "Sale Price", "Longitude", "Latitude",
            "Census Tract", "Full Baths", "Half Baths", "Building Square Feet", "Land Square Feet",
            "Construction Quality", "Repair Condition", "Garage Indicator", "Fireplaces", "Age",
            "Multi Property Indicator"]
    d = pd.read_csv(a.ccao, usecols=cols, low_memory=False)
    acs = pd.read_csv(a.acs)
    acs = acs[(acs.State == "Illinois") & (acs.County == "Cook")].copy()
    acs["tract6"] = acs.CensusTract.astype("int64") % 1_000_000
    acs["pct_black_latino"] = (acs.Hispanic + acs.Black) / 100.0
    share = acs.set_index("tract6").pct_black_latino
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)

    # coordinate-to-tract lookup: every geocoded sale, one row per PIN
    pts = d.dropna(subset=["Longitude", "Latitude", "Census Tract"]).drop_duplicates("PIN")
    pts = pts[pts["Census Tract"].astype("int64").isin(share.index)]
    ux, uy = tr.transform(pts.Longitude.values, pts.Latitude.values)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cook_county_tract_points.npz", UTM_X=ux, UTM_Y=uy,
                        tract=pts["Census Tract"].astype("int64").values,
                        share=share.loc[pts["Census Tract"].astype("int64")].values)

    # the sample
    s = d[(d["Sale Year"] == a.year) & (d["Modeling Group"] == "SF") & (d["Pure Market Filter"] == 1)
          & (d["Multi Property Indicator"] == 0) & (d["Sale Price"] >= 25_000) & (d["Sale Price"] <= 3_000_000)
          & (d["Building Square Feet"] >= 300)].dropna(subset=cols)
    s = s[s["Census Tract"].astype("int64").isin(share.index)].drop_duplicates("PIN")
    s = s.sample(n=a.n, random_state=a.seed)
    sx, sy = tr.transform(s.Longitude.values, s.Latitude.values)
    o = pd.DataFrame({
        "pin": s.PIN.astype(str).values,
        "bathrooms": s["Full Baths"].values + 0.5 * s["Half Baths"].values,
        "sqft_living": s["Building Square Feet"].values,
        "sqft_lot": s["Land Square Feet"].values,
        "quality": 4 - s["Construction Quality"].values,
        "condition": 4 - s["Repair Condition"].values,
        "garage": s["Garage Indicator"].values,
        "fireplaces": s["Fireplaces"].values,
        "age": s["Age"].values,
        "UTM_X": sx, "UTM_Y": sy,
        "log_price": np.log10(s["Sale Price"].values),
        "tract": s["Census Tract"].astype("int64").values,
        "pct_black_latino": share.loc[s["Census Tract"].astype("int64")].values,
    })
    o.to_csv(out / "cook_county_housing_with_demographics.csv", index=False)
    print(f"{len(pts)} lookup points over {pts['Census Tract'].nunique()} tracts; sample {len(o)} sales in "
          f"{o.tract.nunique()} tracts; share above 0.25: {(o.pct_black_latino > 0.25).mean():.3f}; "
          f"median price ${10 ** o.log_price.median():,.0f}")


if __name__ == "__main__":
    main()
