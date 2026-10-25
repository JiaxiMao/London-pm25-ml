from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, PROCESSED_DIR, CRS_EPSG, GRID_SIZE_METRES


BOUNDARY_FILE = RAW_DIR / "London_Boroughs.gpkg"
OUTPUT_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"


def create_grid(bounds, cell_size):
    minx, miny, maxx, maxy = bounds

    xs = range(
        int(minx // cell_size) * cell_size,
        int(maxx // cell_size + 1) * cell_size,
        cell_size,
    )

    ys = range(
        int(miny // cell_size) * cell_size,
        int(maxy // cell_size + 1) * cell_size,
        cell_size,
    )

    cells = []

    for x in xs:
        for y in ys:
            cells.append(
                box(
                    x,
                    y,
                    x + cell_size,
                    y + cell_size,
                )
            )

    return gpd.GeoDataFrame(
        {"geometry": cells},
        crs=f"EPSG:{CRS_EPSG}",
    )


def main():
    print("=" * 90)
    print("BUILDING 500 M LONDON MODELLING GRID")
    print("=" * 90)

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"London boundary file not found:\n{BOUNDARY_FILE}"
        )

    # Read the official London borough boundaries.
    boroughs = gpd.read_file(BOUNDARY_FILE)

    print(f"\nBoundary rows: {len(boroughs):,}")
    print(f"Boundary CRS: {boroughs.crs}")

    # Reproject if necessary.
    if boroughs.crs.to_epsg() != CRS_EPSG:
        boroughs = boroughs.to_crs(epsg=CRS_EPSG)

    # Dissolve all boroughs into Greater London.
    london = boroughs.dissolve()

    print(f"London geometry type: {london.geometry.iloc[0].geom_type}")
    print(f"London bounds: {london.total_bounds}")

    # Create a rectangular 500 m grid covering London's bounding box.
    grid = create_grid(
        london.total_bounds,
        GRID_SIZE_METRES,
    )

    print(f"\nInitial grid cells: {len(grid):,}")

    # Keep cells that intersect Greater London.
    london_geom = london.geometry.iloc[0]

    grid = grid[
        grid.geometry.intersects(london_geom)
    ].copy()

    grid = grid.reset_index(drop=True)

    # Give each grid cell a stable ID.
    grid["grid_id"] = [
        f"G{i:05d}"
        for i in range(1, len(grid) + 1)
    ]

    # Calculate useful centre coordinates.
    grid["centroid_x"] = grid.geometry.centroid.x
    grid["centroid_y"] = grid.geometry.centroid.y

    # Calculate cell area.
    grid["area_m2"] = grid.geometry.area

    # Reorder columns.
    grid = grid[
        [
            "grid_id",
            "centroid_x",
            "centroid_y",
            "area_m2",
            "geometry",
        ]
    ]

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    grid.to_file(
        OUTPUT_FILE,
        layer="london_500m_grid",
        driver="GPKG",
    )

    print("\n--- FINAL GRID ---")
    print(f"Grid cells: {len(grid):,}")
    print(f"Grid CRS: EPSG:{CRS_EPSG}")
    print(f"Cell size: {GRID_SIZE_METRES} m")
    print(f"Output: {OUTPUT_FILE}")

    print("\nGrid construction complete.")


if __name__ == "__main__":
    main()