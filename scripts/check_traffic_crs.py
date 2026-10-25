from pathlib import Path
import sys

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR


TRAFFIC_DIR = PROCESSED_DIR / "traffic"
MINOR_ROADS = TRAFFIC_DIR / "LAEI2022-minor-roads-flows-and-speeds.gpkg"


def main() -> None:
    print("=" * 90)
    print("MINOR-ROAD CRS DIAGNOSTIC")
    print("=" * 90)

    if not MINOR_ROADS.exists():
        raise FileNotFoundError(
            f"Minor-road GeoPackage not found:\n{MINOR_ROADS}"
        )

    layers = gpd.list_layers(MINOR_ROADS)

    print("\nLayers:")
    print(layers.to_string(index=False))

    layer_name = layers["name"].iloc[0]

    gdf = gpd.read_file(
        MINOR_ROADS,
        layer=layer_name,
    )

    print("\n--- CRS METADATA ---")
    print(gdf.crs)
    print(f"EPSG: {gdf.crs.to_epsg() if gdf.crs else 'Unknown'}")

    print("\n--- GEOMETRY ---")
    print(f"Rows: {len(gdf):,}")
    print(
        gdf.geometry.geom_type
        .value_counts()
        .to_string()
    )

    print("\n--- GEOMETRY BOUNDS ---")
    print(gdf.total_bounds)

    print("\n--- STORED EASTING / NORTHING ---")
    print(
        f"Easting min:  {gdf['easting'].min():,.2f}"
    )
    print(
        f"Easting max:  {gdf['easting'].max():,.2f}"
    )
    print(
        f"Northing min: {gdf['northing'].min():,.2f}"
    )
    print(
        f"Northing max: {gdf['northing'].max():,.2f}"
    )

    print("\n--- FIRST 5 FEATURES ---")

    for i, row in gdf.head(5).iterrows():
        geom = row.geometry

        print(f"\nFeature {i}:")
        print(f"  easting:  {row['easting']}")
        print(f"  northing: {row['northing']}")
        print(f"  geometry bounds: {geom.bounds}")
        print(f"  geometry centroid: {geom.centroid.x}, {geom.centroid.y}")

    print("\n--- CRS TEST ---")

    # The easting/northing values appear to be British National Grid.
    # Transform a copy of the stored geometry to EPSG:27700 if the source
    # geometry is geographic, then compare its coordinate ranges.
    if gdf.crs and gdf.crs.is_geographic:
        gdf_bng = gdf.to_crs(epsg=27700)

        print("Source geometry is geographic.")
        print("After transformation to EPSG:27700:")

        print(
            f"  X bounds: {gdf_bng.total_bounds[0]:,.2f} "
            f"to {gdf_bng.total_bounds[2]:,.2f}"
        )

        print(
            f"  Y bounds: {gdf_bng.total_bounds[1]:,.2f} "
            f"to {gdf_bng.total_bounds[3]:,.2f}"
        )

        transformed = gdf_bng.geometry.centroid

        comparison = gdf.copy()

        comparison["transformed_x"] = transformed.x
        comparison["transformed_y"] = transformed.y

        comparison["x_difference"] = (
            comparison["transformed_x"]
            - comparison["easting"]
        )

        comparison["y_difference"] = (
            comparison["transformed_y"]
            - comparison["northing"]
        )

        print("\nCentroid vs stored easting/northing:")
        print(
            comparison[
                [
                    "easting",
                    "northing",
                    "transformed_x",
                    "transformed_y",
                    "x_difference",
                    "y_difference",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print("\nMean absolute differences:")
        print(
            f"  X: {comparison['x_difference'].abs().mean():,.2f} m"
        )
        print(
            f"  Y: {comparison['y_difference'].abs().mean():,.2f} m"
        )

    else:
        print(
            "Source geometry is not marked as geographic; "
            "no CRS transformation test was performed."
        )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()