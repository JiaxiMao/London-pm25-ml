from pathlib import Path
from zipfile import ZipFile
import sys
import shutil
import tempfile

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.errors import GEOSException


# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    CRS_EPSG,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GREEN_ZIP = RAW_DIR / "Green_Cover_2024.zip"
GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"

OUTPUT_FILE = PROCESSED_DIR / "green_cover_500m.parquet"
OUTPUT_CSV = PROCESSED_DIR / "green_cover_500m.csv"

GRID_LAYER = "london_500m_grid"


# ---------------------------------------------------------------------------
# Geometry repair helper
# ---------------------------------------------------------------------------

def repair_geometry(geometry):
    """
    Repair an invalid geometry.

    Strategy:
    1. Return valid geometry unchanged.
    2. Try Shapely make_valid().
    3. Fall back to buffer(0).
    4. Return None if still unusable.
    """
    if geometry is None or geometry.is_empty:
        return None

    if geometry.is_valid:
        return geometry

    try:
        repaired = geometry.make_valid()

        if repaired is not None and not repaired.is_empty:
            return repaired

    except Exception:
        pass

    try:
        repaired = geometry.buffer(0)

        if repaired is not None and not repaired.is_empty:
            return repaired

    except Exception:
        pass

    return None


def main() -> None:

    print("=" * 90)
    print("ROBUST GREEN COVER AGGREGATION")
    print("=" * 90)

    if not GREEN_ZIP.exists():
        raise FileNotFoundError(
            f"Green Cover ZIP not found:\n{GREEN_ZIP}"
        )

    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"London grid not found:\n{GRID_FILE}"
        )

    # -----------------------------------------------------------------------
    # Load modelling grid
    # -----------------------------------------------------------------------

    print("\nLoading 500 m London grid...")

    grid = gpd.read_file(
        GRID_FILE,
        layer=GRID_LAYER,
    )

    if grid.crs is None:
        raise ValueError("London grid has no CRS.")

    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(epsg=CRS_EPSG)

    grid = grid[
        ["grid_id", "area_m2", "geometry"]
    ].copy()

    print(f"Grid cells: {len(grid):,}")

    # -----------------------------------------------------------------------
    # Build spatial index once
    # -----------------------------------------------------------------------

    print("Building grid spatial index...")

    grid_sindex = grid.sindex

    # -----------------------------------------------------------------------
    # Temporary extraction directory
    # -----------------------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="green_cover_",
            dir=str(PROCESSED_DIR),
        )
    )

    green_area = pd.Series(
        0.0,
        index=grid["grid_id"],
        name="green_area_m2",
    )

    failed_tiles = []
    repaired_tiles = 0
    processed_tiles = 0
    intersecting_tiles = 0

    try:

        with ZipFile(GREEN_ZIP, "r") as archive:

            members = archive.namelist()

            shapefiles = [
                name
                for name in members
                if name.lower().endswith(".shp")
            ]

            total_tiles = len(shapefiles)

            print(
                f"Green Cover tiles: {total_tiles:,}"
            )

            # ---------------------------------------------------------------
            # Process tile by tile
            # ---------------------------------------------------------------

            for tile_number, shp_member in enumerate(
                shapefiles,
                start=1,
            ):

                processed_tiles += 1

                shp_filename = Path(shp_member).name
                base_name = Path(shp_member).stem

                related_files = [
                    name
                    for name in members
                    if Path(name).stem == base_name
                ]

                # -----------------------------------------------------------
                # Extract this tile
                # -----------------------------------------------------------

                for member in related_files:

                    target = (
                        temp_dir
                        / Path(member).name
                    )

                    with archive.open(member) as source:
                        with open(
                            target,
                            "wb",
                        ) as destination:
                            shutil.copyfileobj(
                                source,
                                destination,
                            )

                shp_path = (
                    temp_dir
                    / shp_filename
                )

                # -----------------------------------------------------------
                # Read tile
                # -----------------------------------------------------------

                try:
                    tile = gpd.read_file(shp_path)
                except Exception as exc:
                    failed_tiles.append(
                        (
                            shp_filename,
                            f"read error: {exc}",
                        )
                    )

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                if tile.empty:
                    for path in temp_dir.iterdir():
                        path.unlink()
                    continue

                # -----------------------------------------------------------
                # CRS
                # -----------------------------------------------------------

                if tile.crs is None:
                    failed_tiles.append(
                        (
                            shp_filename,
                            "missing CRS",
                        )
                    )

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                if tile.crs.to_epsg() != CRS_EPSG:
                    tile = tile.to_crs(
                        epsg=CRS_EPSG
                    )

                # -----------------------------------------------------------
                # Repair invalid geometry
                # -----------------------------------------------------------

                invalid_count = (
                    (~tile.geometry.is_valid)
                    & tile.geometry.notna()
                ).sum()

                if invalid_count > 0:
                    repaired_tiles += 1

                    repaired_geometries = (
                        tile.geometry.apply(
                            repair_geometry
                        )
                    )

                    tile["geometry"] = (
                        repaired_geometries
                    )

                    tile = tile[
                        tile.geometry.notna()
                        & ~tile.geometry.is_empty
                    ].copy()

                if tile.empty:
                    failed_tiles.append(
                        (
                            shp_filename,
                            "no usable geometry after repair",
                        )
                    )

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                # -----------------------------------------------------------
                # Combine tile geometries
                # -----------------------------------------------------------

                try:
                    tile_geometry = unary_union(
                        tile.geometry.tolist()
                    )

                    tile_geometry = repair_geometry(
                        tile_geometry
                    )

                except Exception as exc:
                    failed_tiles.append(
                        (
                            shp_filename,
                            f"union error: {exc}",
                        )
                    )

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                if tile_geometry is None:
                    failed_tiles.append(
                        (
                            shp_filename,
                            "tile union could not be repaired",
                        )
                    )

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                # -----------------------------------------------------------
                # Query only grid cells whose bounding boxes can intersect
                # the tile.
                # -----------------------------------------------------------

                candidate_indices = list(
                    grid_sindex.intersection(
                        tile_geometry.bounds
                    )
                )

                if not candidate_indices:

                    for path in temp_dir.iterdir():
                        path.unlink()

                    continue

                intersecting_tiles += 1

                candidates = grid.iloc[
                    candidate_indices
                ]

                # -----------------------------------------------------------
                # Intersections
                # -----------------------------------------------------------

                for row in candidates.itertuples():

                    try:

                        intersection = (
                            row.geometry.intersection(
                                tile_geometry
                            )
                        )

                    except GEOSException:

                        # Last-resort repair of the grid cell.
                        try:
                            repaired_grid = (
                                row.geometry.buffer(0)
                            )

                            intersection = (
                                repaired_grid.intersection(
                                    tile_geometry
                                )
                            )

                        except Exception as exc:

                            failed_tiles.append(
                                (
                                    shp_filename,
                                    f"intersection error: {exc}",
                                )
                            )

                            continue

                    except Exception as exc:

                        failed_tiles.append(
                            (
                                shp_filename,
                                f"intersection error: {exc}",
                            )
                        )

                        continue

                    if (
                        intersection is None
                        or intersection.is_empty
                    ):
                        continue

                    area = intersection.area

                    if area <= 0:
                        continue

                    green_area.loc[
                        row.grid_id
                    ] += area

                # -----------------------------------------------------------
                # Clean temporary files
                # -----------------------------------------------------------

                for path in temp_dir.iterdir():
                    path.unlink()

                if (
                    tile_number % 25 == 0
                    or tile_number == total_tiles
                ):
                    print(
                        f"Processed "
                        f"{tile_number:,}/"
                        f"{total_tiles:,} tiles | "
                        f"repaired tiles: "
                        f"{repaired_tiles:,} | "
                        f"failed tiles: "
                        f"{len(failed_tiles):,}"
                    )

        # -------------------------------------------------------------------
        # Final dataset
        # -------------------------------------------------------------------

        result = grid[
            ["grid_id", "area_m2"]
        ].copy()

        result["green_area_m2"] = (
            result["grid_id"]
            .map(green_area)
            .fillna(0.0)
        )

        result["green_cover_pct"] = (
            result["green_area_m2"]
            / result["area_m2"]
            * 100.0
        )

        result["green_cover_pct"] = (
            result["green_cover_pct"]
            .clip(
                lower=0,
                upper=100,
            )
        )

        # -------------------------------------------------------------------
        # Quality check
        # -------------------------------------------------------------------

        print("\n" + "=" * 90)
        print("--- GREEN COVER QUALITY CHECK ---")
        print("=" * 90)

        print(
            f"Modelling cells: "
            f"{len(result):,}"
        )

        print(
            f"Cells with green cover: "
            f"{(result['green_area_m2'] > 0).sum():,}"
        )

        print(
            f"Cells with zero green cover: "
            f"{(result['green_area_m2'] == 0).sum():,}"
        )

        print(
            f"Minimum green cover: "
            f"{result['green_cover_pct'].min():.4f}%"
        )

        print(
            f"Maximum green cover: "
            f"{result['green_cover_pct'].max():.4f}%"
        )

        print(
            f"Mean green cover: "
            f"{result['green_cover_pct'].mean():.4f}%"
        )

        print(
            f"Median green cover: "
            f"{result['green_cover_pct'].median():.4f}%"
        )

        print("\nSummary:")

        print(
            result[
                [
                    "green_area_m2",
                    "green_cover_pct",
                ]
            ]
            .describe()
            .transpose()
            .to_string()
        )

        print("\nGeometry processing:")
        print(
            f"Tiles processed: "
            f"{processed_tiles:,}"
        )

        print(
            f"Tiles intersecting London grid: "
            f"{intersecting_tiles:,}"
        )

        print(
            f"Tiles requiring geometry repair: "
            f"{repaired_tiles:,}"
        )

        print(
            f"Recorded processing failures: "
            f"{len(failed_tiles):,}"
        )

        # -------------------------------------------------------------------
        # Save outputs
        # -------------------------------------------------------------------

        result.to_parquet(
            OUTPUT_FILE,
            index=False,
        )

        result.to_csv(
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

        # ---------------------------------------------------------------
        # Report failures, if any
        # ---------------------------------------------------------------

        if failed_tiles:

            print("\nFirst 20 recorded tile issues:")

            for filename, reason in failed_tiles[:20]:
                print(
                    f"  {filename}: {reason}"
                )

        print("\nGreen Cover aggregation complete.")

    finally:

        if temp_dir.exists():
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()