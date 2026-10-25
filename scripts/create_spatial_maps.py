from pathlib import Path
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CRS_EPSG, FIGURES_DIR, PROCESSED_DIR


GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"
DATA_FILE = PROCESSED_DIR / "london_pm25_final.parquet"
BOUNDARY_FILE = PROJECT_ROOT / "data" / "raw" / "London_Boroughs.gpkg"
GRID_LAYER = "london_500m_grid"
FIGURE_DPI = 300
FIGURE_SIZE = (10, 10)


def add_borough_boundaries(ax, boroughs):
    boroughs.boundary.plot(
        ax=ax,
        linewidth=0.35,
        edgecolor="black",
        alpha=0.65,
    )


def save_map(filename, title, source_text):
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_axis_off()
    fig.text(0.5, 0.025, source_text, ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=[0, 0.045, 1, 1])
    output = FIGURES_DIR / filename
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def quantile_limits(series, lower=0.02, upper=0.98):
    return series.quantile(lower), series.quantile(upper)


def main():
    print("=" * 90)
    print("CREATING LONDON SPATIAL MAPS")
    print("=" * 90)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading modelling dataset...")
    df = pd.read_parquet(DATA_FILE)
    print(f"Modelling rows: {len(df):,}")

    print("\nLoading 500 m grid...")
    grid = gpd.read_file(GRID_FILE, layer=GRID_LAYER)
    if grid.crs is None:
        raise ValueError("Grid has no CRS.")
    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(epsg=CRS_EPSG)
    print(f"Grid cells: {len(grid):,}")

    map_df = grid.merge(df, on="grid_id", how="inner", validate="one_to_one")
    print(f"Mapped cells: {len(map_df):,}")

    print("\nLoading London borough boundaries...")
    boroughs = gpd.read_file(BOUNDARY_FILE)
    if boroughs.crs is None:
        raise ValueError("Borough boundary layer has no CRS.")
    if boroughs.crs.to_epsg() != CRS_EPSG:
        boroughs = boroughs.to_crs(epsg=CRS_EPSG)
    print(f"Boroughs: {len(boroughs):,}")

    print("\nCreating PM2.5 map...")
    vmin, vmax = quantile_limits(map_df["pm25_mean"])
    ax = map_df.plot(column="pm25_mean", cmap="YlOrRd", figsize=FIGURE_SIZE,
                     linewidth=0, edgecolor="none", vmin=vmin, vmax=vmax,
                     legend=True, legend_kwds={"label": "Annual mean PM2.5 (µg/m³)", "shrink": 0.65})
    add_borough_boundaries(ax, boroughs)
    save_map("01_pm25_spatial_distribution.png", "Spatial Distribution of Modelled PM2.5 Across London",
             "Source: Greater London Authority LAEI 2022; 500 m analytical grid.")

    print("\nCreating green-cover map...")
    vmin, vmax = quantile_limits(map_df["green_cover_pct"])
    vmin = min(0.0, vmin)
    vmax = max(100.0, vmax)
    ax = map_df.plot(column="green_cover_pct", cmap="YlGn", figsize=FIGURE_SIZE,
                     linewidth=0, edgecolor="none", vmin=vmin, vmax=vmax,
                     legend=True, legend_kwds={"label": "Green cover (%)", "shrink": 0.65})
    add_borough_boundaries(ax, boroughs)
    save_map("02_green_cover_spatial_distribution.png", "Spatial Distribution of Green Cover Across London",
             "Source: Greater London Authority Green Cover 2024; underlying assessment based primarily on 2022 data.")

    print("\nCreating major-road traffic map...")
    map_df["log_major_aadf"] = np.log1p(map_df["major_mean_aadf"])
    vmin, vmax = quantile_limits(map_df["log_major_aadf"])
    ax = map_df.plot(column="log_major_aadf", cmap="Blues", figsize=FIGURE_SIZE,
                     linewidth=0, edgecolor="none", vmin=vmin, vmax=vmax,
                     legend=True, legend_kwds={"label": "log(1 + mean major-road AADF)", "shrink": 0.65})
    add_borough_boundaries(ax, boroughs)
    save_map("03_major_road_traffic.png", "Spatial Distribution of Major-Road Traffic Across London",
             "Source: LAEI 2022 traffic flows; log scale used for visualisation only.")

    print("\nCreating minor-road traffic map...")
    map_df["log_minor_vkm"] = np.log1p(map_df["minor_vkm_total"])
    vmin, vmax = quantile_limits(map_df["log_minor_vkm"])
    ax = map_df.plot(column="log_minor_vkm", cmap="PuBu", figsize=FIGURE_SIZE,
                     linewidth=0, edgecolor="none", vmin=vmin, vmax=vmax,
                     legend=True, legend_kwds={"label": "log(1 + minor-road vehicle-km)", "shrink": 0.65})
    add_borough_boundaries(ax, boroughs)
    save_map("04_minor_road_traffic.png", "Spatial Distribution of Minor-Road Vehicle Activity Across London",
             "Source: LAEI 2022 minor-road traffic data; log scale used for visualisation only.")

    print("\nCreating PM2.5 hotspot map...")
    threshold = map_df["pm25_mean"].quantile(0.90)
    map_df["pm25_hotspot"] = map_df["pm25_mean"] >= threshold
    print(f"90th-percentile PM2.5 threshold: {threshold:.4f} µg/m³")
    print(f"Hotspot cells: {map_df['pm25_hotspot'].sum():,}")
    ax = map_df.plot(figsize=FIGURE_SIZE, color="white", linewidth=0, edgecolor="none")
    map_df[map_df["pm25_hotspot"]].plot(ax=ax, color="red", linewidth=0, edgecolor="none", alpha=0.85)
    add_borough_boundaries(ax, boroughs)
    save_map("05_pm25_hotspots_top10pct.png", "PM2.5 Hotspots: Top 10% of London 500 m Cells",
             f"Hotspot threshold = {threshold:.2f} µg/m³; source: LAEI 2022.")

    summary = pd.DataFrame({
        "metric": [
            "mapped_cells", "pm25_mean", "pm25_median", "pm25_90th_percentile",
            "green_cover_mean_pct", "green_cover_median_pct", "major_aadf_mean",
            "minor_vkm_mean", "pm25_hotspot_threshold", "pm25_hotspot_cells",
        ],
        "value": [
            len(map_df), map_df["pm25_mean"].mean(), map_df["pm25_mean"].median(), threshold,
            map_df["green_cover_pct"].mean(), map_df["green_cover_pct"].median(),
            map_df["major_mean_aadf"].mean(), map_df["minor_vkm_total"].mean(),
            threshold, map_df["pm25_hotspot"].sum(),
        ],
    })
    summary_file = PROCESSED_DIR / "spatial_map_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")
    print("\nSpatial mapping complete.")


if __name__ == "__main__":
    main()