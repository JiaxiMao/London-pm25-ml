from pathlib import Path
from zipfile import ZipFile
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PM25_CSV_ZIP,
    PROCESSED_DIR,
    GRID_SIZE_METRES,
)


PM25_MEMBER = "CSV/LAEI2022_V1_PM25.csv"

GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"
OUTPUT_FILE = PROCESSED_DIR / "pm25_500m.parquet"
OUTPUT_CSV = PROCESSED_DIR / "pm25_500m.csv"


def main() -> None:
    print("=" * 90)
    print("AGGREGATING LAEI 2022 PM2.5 TO THE 500 M LONDON GRID")
    print("=" * 90)

    if not PM25_CSV_ZIP.exists():
        raise FileNotFoundError(
            f"PM2.5 ZIP not found:\n{PM25_CSV_ZIP}"
        )

    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"500 m grid not found:\n{GRID_FILE}"
        )

    # -----------------------------------------------------------------------
    # Read the 500 m grid IDs and bounding boxes.
    # We only need the geometry bounds here; the PM2.5 aggregation itself
    # can be done directly from x/y coordinates.
    # -----------------------------------------------------------------------

    import geopandas as gpd

    grid = gpd.read_file(
        GRID_FILE,
        layer="london_500m_grid",
    )

    print(f"\nLondon grid cells: {len(grid):,}")

    # Build a lookup table using the lower-left coordinate of each
    # 500 m grid cell.
    #
    # The grid was created on multiples of 500 m, so floor(x / 500) * 500
    # and floor(y / 500) * 500 identify the corresponding cell.
    grid["grid_x"] = (
        grid.geometry.bounds["minx"]
        .astype("int64")
    )

    grid["grid_y"] = (
        grid.geometry.bounds["miny"]
        .astype("int64")
    )

    grid_lookup = grid[
        ["grid_id", "grid_x", "grid_y", "area_m2"]
    ].copy()

    print(
        f"Grid X range: "
        f"{grid_lookup['grid_x'].min():,} - "
        f"{grid_lookup['grid_x'].max():,}"
    )

    print(
        f"Grid Y range: "
        f"{grid_lookup['grid_y'].min():,} - "
        f"{grid_lookup['grid_y'].max():,}"
    )

    # -----------------------------------------------------------------------
    # Read PM2.5 from the ZIP in chunks.
    # This avoids loading the entire 5.85 million-row CSV into memory.
    # -----------------------------------------------------------------------

    chunks = []

    chunk_size = 500_000

    print("\nReading PM2.5 CSV in chunks...")
    print(f"Chunk size: {chunk_size:,}")

    with ZipFile(PM25_CSV_ZIP, "r") as archive:

        if PM25_MEMBER not in archive.namelist():
            raise FileNotFoundError(
                f"{PM25_MEMBER} not found in {PM25_CSV_ZIP}"
            )

        with archive.open(PM25_MEMBER) as source:

            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    source,
                    usecols=["x", "y", "conc"],
                    chunksize=chunk_size,
                    low_memory=False,
                ),
                start=1,
            ):

                print(
                    f"  Processing chunk {chunk_number}..."
                )

                # Determine the 500 m grid cell from the 20 m coordinates.
                chunk["grid_x"] = (
                    (chunk["x"] // GRID_SIZE_METRES)
                    * GRID_SIZE_METRES
                ).astype("int64")

                chunk["grid_y"] = (
                    (chunk["y"] // GRID_SIZE_METRES)
                    * GRID_SIZE_METRES
                ).astype("int64")

                # Keep only points whose grid coordinates occur in
                # the Greater London modelling grid.
                merged = chunk.merge(
                    grid_lookup,
                    on=["grid_x", "grid_y"],
                    how="inner",
                )

                if not merged.empty:
                    chunks.append(
                        merged[
                            ["grid_id", "grid_x", "grid_y", "conc"]
                        ]
                    )

                print(
                    f"    Source rows: {len(chunk):,}"
                )
                print(
                    f"    London rows retained: {len(merged):,}"
                )

    if not chunks:
        raise RuntimeError(
            "No PM2.5 observations matched the London 500 m grid."
        )

    # -----------------------------------------------------------------------
    # Combine chunks.
    # -----------------------------------------------------------------------

    print("\nCombining processed chunks...")

    pm25 = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(
        f"PM2.5 rows retained: {len(pm25):,}"
    )

    # -----------------------------------------------------------------------
    # Aggregate 20 m PM2.5 values to one mean per 500 m cell.
    # -----------------------------------------------------------------------

    print("\nCalculating mean PM2.5 by modelling cell...")

    pm25_500m = (
        pm25
        .groupby(
            ["grid_id", "grid_x", "grid_y"],
            as_index=False,
        )
        .agg(
            pm25_mean=("conc", "mean"),
            pm25_median=("conc", "median"),
            pm25_min=("conc", "min"),
            pm25_max=("conc", "max"),
            pm25_std=("conc", "std"),
            n_pm25_cells=("conc", "size"),
        )
    )

    # -----------------------------------------------------------------------
    # Attach area information from the grid.
    # -----------------------------------------------------------------------

    pm25_500m = pm25_500m.merge(
        grid_lookup[
            ["grid_id", "area_m2"]
        ],
        on="grid_id",
        how="left",
    )

    # -----------------------------------------------------------------------
    # Basic quality checks.
    # -----------------------------------------------------------------------

    print("\n--- QUALITY CHECK ---")
    print(
        f"Modelling cells with PM2.5: "
        f"{len(pm25_500m):,}"
    )

    print(
        f"Mean PM2.5: "
        f"{pm25_500m['pm25_mean'].mean():.4f}"
    )

    print(
        f"Minimum cell mean: "
        f"{pm25_500m['pm25_mean'].min():.4f}"
    )

    print(
        f"Maximum cell mean: "
        f"{pm25_500m['pm25_mean'].max():.4f}"
    )

    print(
        f"Median cell mean: "
        f"{pm25_500m['pm25_mean'].median():.4f}"
    )

    print(
        f"Cells with missing PM2.5: "
        f"{pm25_500m['pm25_mean'].isna().sum():,}"
    )

    print(
        f"Mean original 20 m cells per 500 m cell: "
        f"{pm25_500m['n_pm25_cells'].mean():.2f}"
    )

    print(
        f"Minimum original cells in a modelling cell: "
        f"{pm25_500m['n_pm25_cells'].min():,}"
    )

    print(
        f"Maximum original cells in a modelling cell: "
        f"{pm25_500m['n_pm25_cells'].max():,}"
    )

    # -----------------------------------------------------------------------
    # Save outputs.
    # -----------------------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pm25_500m.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    pm25_500m.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n--- OUTPUTS ---")
    print(f"Parquet: {OUTPUT_FILE}")
    print(f"CSV:     {OUTPUT_CSV}")

    print("\nPM2.5 aggregation complete.")


if __name__ == "__main__":
    main()