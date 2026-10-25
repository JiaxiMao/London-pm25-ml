from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DIR,
    CRS_EPSG,
)


PM25_FILE = PROCESSED_DIR / "pm25_500m.parquet"
TRAFFIC_FILE = PROCESSED_DIR / "traffic_500m.parquet"
GREEN_FILE = PROCESSED_DIR / "green_cover_500m.parquet"
GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "london_pm25_modelling_dataset.parquet"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "london_pm25_modelling_dataset.csv"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def main() -> None:

    print("=" * 90)
    print("BUILDING MASTER LONDON PM2.5 MODELLING DATASET")
    print("=" * 90)

    for path in [
        PM25_FILE,
        TRAFFIC_FILE,
        GREEN_FILE,
        GRID_FILE,
    ]:
        require_file(path)

    # ------------------------------------------------------------------
    # Load source tables
    # ------------------------------------------------------------------

    print("\nLoading PM2.5...")
    pm25 = pd.read_parquet(PM25_FILE)

    print(f"PM2.5 rows: {len(pm25):,}")

    print("\nLoading traffic...")
    traffic = pd.read_parquet(TRAFFIC_FILE)

    print(f"Traffic rows: {len(traffic):,}")

    print("\nLoading green cover...")
    green = pd.read_parquet(GREEN_FILE)

    print(f"Green Cover rows: {len(green):,}")

    print("\nLoading modelling grid...")
    grid = gpd.read_file(
        GRID_FILE,
        layer="london_500m_grid",
    )

    if grid.crs is None:
        raise ValueError(
            "Modelling grid has no CRS."
        )

    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(
            epsg=CRS_EPSG
        )

    print(f"Grid rows: {len(grid):,}")

    # ------------------------------------------------------------------
    # Keep useful spatial fields
    # ------------------------------------------------------------------

    grid_table = grid[
        [
            "grid_id",
            "centroid_x",
            "centroid_y",
            "area_m2",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Check uniqueness
    # ------------------------------------------------------------------

    print("\n--- SOURCE KEY CHECK ---")

    for name, df in [
        ("PM2.5", pm25),
        ("Traffic", traffic),
        ("Green Cover", green),
        ("Grid", grid_table),
    ]:

        duplicates = df["grid_id"].duplicated().sum()

        print(
            f"{name}: "
            f"{duplicates:,} duplicate grid IDs"
        )

        if duplicates:
            raise ValueError(
                f"{name} contains duplicate grid_id values."
            )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    print("\nMerging PM2.5 + traffic...")

    master = pm25.merge(
        traffic,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Rows after PM2.5 + traffic merge: "
        f"{len(master):,}"
    )

    print("\nMerging Green Cover...")

    master = master.merge(
        green[
            [
                "grid_id",
                "green_area_m2",
                "green_cover_pct",
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Rows after Green Cover merge: "
        f"{len(master):,}"
    )

    print("\nMerging grid coordinates...")

    master = master.merge(
        grid_table,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Rows after grid merge: "
        f"{len(master):,}"
    )

    # ------------------------------------------------------------------
    # Reorder columns
    # ------------------------------------------------------------------

    preferred_columns = [
        "grid_id",
        "centroid_x",
        "centroid_y",
        "area_m2",
        "pm25_mean",
        "pm25_median",
        "pm25_min",
        "pm25_max",
        "pm25_std",
        "n_pm25_cells",
        "green_area_m2",
        "green_cover_pct",
        "major_road_length_m",
        "major_vehicle_m_per_day",
        "major_hgv_vehicle_m_per_day",
        "major_mean_aadf",
        "major_hgv_mean_aadf",
        "minor_vkm_total",
        "minor_hgv_vkm",
    ]

    existing_columns = [
        col
        for col in preferred_columns
        if col in master.columns
    ]

    remaining_columns = [
        col
        for col in master.columns
        if col not in existing_columns
    ]

    master = master[
        existing_columns
        + remaining_columns
    ]

    # ------------------------------------------------------------------
    # Final quality check
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("--- MASTER DATASET QUALITY CHECK ---")
    print("=" * 90)

    print(
        f"Rows: {len(master):,}"
    )

    print(
        f"Columns: {len(master.columns):,}"
    )

    print("\nMissing values:")

    missing = (
        master
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    print("\nDuplicate grid IDs:")

    print(
        master["grid_id"]
        .duplicated()
        .sum()
    )

    print("\nTarget summary:")

    print(
        master[
            "pm25_mean"
        ]
        .describe()
        .to_string()
    )

    print("\nPredictor summary:")

    predictor_columns = [
        "green_cover_pct",
        "major_road_length_m",
        "major_mean_aadf",
        "major_hgv_mean_aadf",
        "minor_vkm_total",
        "minor_hgv_vkm",
    ]

    available_predictors = [
        col
        for col in predictor_columns
        if col in master.columns
    ]

    print(
        master[
            available_predictors
        ]
        .describe()
        .transpose()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    master.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    master.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("OUTPUTS")
    print("=" * 90)

    print(
        f"Parquet: {OUTPUT_FILE}"
    )

    print(
        f"CSV:     {OUTPUT_CSV}"
    )

    print(
        "\nMaster dataset construction complete."
    )


if __name__ == "__main__":
    main()