from pathlib import Path
from zipfile import ZipFile
import sys

import geopandas as gpd

# ---------------------------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, PROCESSED_DIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GREEN_ZIP = RAW_DIR / "Green_Cover_2024.zip"
GREEN_DIR = PROCESSED_DIR / "green_cover"


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("GREEN COVER 2024 INSPECTION")
    print("=" * 100)

    if not GREEN_ZIP.exists():
        raise FileNotFoundError(
            f"Green Cover ZIP not found:\n{GREEN_ZIP}"
        )

    GREEN_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Find shapefiles in the archive
    # -----------------------------------------------------------------------

    with ZipFile(GREEN_ZIP, "r") as archive:

        shapefile_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".shp")
        ]

        print(f"Shapefiles found in archive: {len(shapefile_names):,}")

        if not shapefile_names:
            raise RuntimeError(
                "No shapefiles were found in the Green Cover ZIP."
            )

        # Use the first tile for structural inspection.
        shp_member = shapefile_names[0]
        shp_filename = Path(shp_member).name
        base_name = Path(shp_member).stem

        print(f"Sample tile: {shp_member}")

        # -------------------------------------------------------------------
        # Extract the complete shapefile set (.shp, .shx, .dbf, .prj, etc.)
        # -------------------------------------------------------------------

        related_files = [
            name
            for name in archive.namelist()
            if Path(name).stem == base_name
        ]

        print(f"Related files for sample tile: {len(related_files)}")

        for member in related_files:
            target = GREEN_DIR / Path(member).name

            with archive.open(member) as source:
                with open(target, "wb") as destination:
                    destination.write(source.read())

    # -----------------------------------------------------------------------
    # Confirm extraction
    # -----------------------------------------------------------------------

    shp = GREEN_DIR / shp_filename

    if not shp.exists():
        raise RuntimeError(
            f"Expected shapefile was not created:\n{shp}"
        )

    print("\nReading sample shapefile:")
    print(shp)

    # -----------------------------------------------------------------------
    # Read GIS layer
    # -----------------------------------------------------------------------

    gdf = gpd.read_file(shp)

    # -----------------------------------------------------------------------
    # Basic dataset information
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- DATASET ---")
    print("=" * 100)

    print(f"Rows:        {len(gdf):,}")
    print(f"Columns:     {len(gdf.columns):,}")
    print(f"CRS:         {gdf.crs}")
    print(f"CRS EPSG:    {gdf.crs.to_epsg() if gdf.crs else 'Unknown'}")

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- GEOMETRY ---")
    print("=" * 100)

    print("Geometry types:")
    print(
        gdf.geometry.geom_type
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nGeometry valid:")
    print(gdf.geometry.is_valid.value_counts(dropna=False).to_string())

    print("\nTotal bounds:")
    print(gdf.total_bounds)

    # -----------------------------------------------------------------------
    # Columns
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- COLUMNS ---")
    print("=" * 100)

    for column in gdf.columns:
        print(
            f"{column:<30} "
            f"dtype={str(gdf[column].dtype):<12}"
        )

    # -----------------------------------------------------------------------
    # Sample rows
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- SAMPLE ROWS ---")
    print("=" * 100)

    print(gdf.head(10).to_string())

    # -----------------------------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- NUMERIC COLUMNS ---")
    print("=" * 100)

    numeric = gdf.select_dtypes(include="number")

    if numeric.empty:
        print("No numeric columns found.")
    else:
        print(
            numeric.describe(
                include="all"
            ).transpose().to_string()
        )

    # -----------------------------------------------------------------------
    # Non-numeric columns
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- NON-NUMERIC VALUE COUNTS ---")
    print("=" * 100)

    for column in gdf.columns:

        if column == gdf.geometry.name:
            continue

        if gdf[column].dtype == "object":

            print(f"\n{column}:")

            values = (
                gdf[column]
                .value_counts(dropna=False)
                .head(30)
            )

            print(values.to_string())

    # -----------------------------------------------------------------------
    # Missing values
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- MISSING VALUES ---")
    print("=" * 100)

    print(
        gdf.isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    # -----------------------------------------------------------------------
    # Geometry area
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- GEOMETRY AREA ---")
    print("=" * 100)

    if gdf.crs and not gdf.crs.is_geographic:

        area = gdf.geometry.area

        print(
            area.describe().to_string()
        )

        print(f"\nTotal area: {area.sum():,.2f} square metres")

    else:

        print(
            "CRS is geographic; geometry area is not reported in square metres."
        )

    # -----------------------------------------------------------------------
    # Duplicate geometries
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("--- DUPLICATE GEOMETRIES ---")
    print("=" * 100)

    duplicate_count = gdf.geometry.duplicated().sum()

    print(f"Duplicate geometries: {duplicate_count:,}")

    # -----------------------------------------------------------------------
    # Green Cover inspection complete
    # -----------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("Green Cover inspection complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()