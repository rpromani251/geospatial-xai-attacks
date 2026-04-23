"""Generate demographic maps of King County census tracts.

Produces:
  1. Choropleth of pct_black by tract with housing points
  2. Choropleth of pct_latino by tract with housing points
  3. Combined pct_black_latino choropleth with housing points
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

TRACTS_GEO = PROCESSED_DATA_DIR / "king_county_tracts_demographics.geojson"
HOUSING_CSV = PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv"
OUT_DIR = RESULTS_DIR / "figures"


def load_data():
    tracts = gpd.read_file(TRACTS_GEO)
    # Reproject to UTM 10N for consistent coords with housing data
    tracts = tracts.to_crs("EPSG:32610")

    housing = pd.read_csv(HOUSING_CSV)
    geometry = [Point(x, y) for x, y in zip(housing["UTM_X"], housing["UTM_Y"])]
    housing_gdf = gpd.GeoDataFrame(housing, geometry=geometry, crs="EPSG:32610")

    return tracts, housing_gdf


def plot_demographic_map(tracts, housing, column, title, cmap, filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Plot all tracts with demographic choropleth
    tracts.plot(
        column=column,
        cmap=cmap,
        linewidth=0.3,
        edgecolor="0.5",
        ax=ax,
        legend=True,
        legend_kwds={
            "label": title,
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
        vmin=0,
        vmax=tracts[column].quantile(0.95),  # cap at 95th pctl for contrast
    )

    # Overlay housing points
    housing.plot(
        ax=ax,
        color="black",
        markersize=1.5,
        alpha=0.3,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_axis_off()

    # Legend for housing points
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.5, label="Housing records (n=3,000)")
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / filename}")
    plt.close(fig)


def plot_combined_map(tracts, housing):
    """Side-by-side: pct_black and pct_latino."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 11))

    for ax, col, cmap, label in [
        (axes[0], "pct_black", "Purples", "% Black Population"),
        (axes[1], "pct_latino", "Oranges", "% Latino Population"),
    ]:
        tracts.plot(
            column=col,
            cmap=cmap,
            linewidth=0.3,
            edgecolor="0.5",
            ax=ax,
            legend=True,
            legend_kwds={
                "label": label,
                "orientation": "horizontal",
                "shrink": 0.7,
                "pad": 0.02,
            },
            vmin=0,
            vmax=tracts[col].quantile(0.95),
        )
        housing.plot(ax=ax, color="black", markersize=1, alpha=0.25)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_axis_off()

    # Shared legend for housing points
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.5, label="Housing records (n=3,000)")
    ]
    axes[1].legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "King County Census Tract Demographics\nACS 2011–2015, Table B03002",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "demographics_combined.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'demographics_combined.png'}")
    plt.close(fig)


def main():
    print("Loading data...")
    tracts, housing = load_data()
    print(f"  {len(tracts)} tracts, {len(housing)} housing records")

    # Individual maps
    plot_demographic_map(
        tracts, housing,
        column="pct_black",
        title="% Black Population by Census Tract — King County",
        cmap="Purples",
        filename="demographics_pct_black.png",
    )
    plot_demographic_map(
        tracts, housing,
        column="pct_latino",
        title="% Latino Population by Census Tract — King County",
        cmap="Oranges",
        filename="demographics_pct_latino.png",
    )
    plot_demographic_map(
        tracts, housing,
        column="pct_black_latino",
        title="% Black + Latino Population by Census Tract — King County",
        cmap="Reds",
        filename="demographics_pct_black_latino.png",
    )

    # Combined side-by-side
    plot_combined_map(tracts, housing)

    print("\nDone! 4 figures saved to results/figures/")


if __name__ == "__main__":
    main()
