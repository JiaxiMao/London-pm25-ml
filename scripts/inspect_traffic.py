from pathlib import Path
import sys

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR


TRAFFIC_DIR = PROCESSED_DIR / "traffic"

MAJOR_ROADS = TRAFFIC_DIR / "LAEI2022-major-roads-flows-and-speeds.gpkg"
MINOR_ROADS = TRAFFIC_DIR / "LAEI2022-minor-roads-flows-and-speeds.gpkg"


def inspect_gpkg(path: Path) -> None:
    print("=" * 100)
    print(f"FILE: {path.name}")
    print("=" * 100)

    layers = gpd.list_layers(path)

    print("\nAvailable layers:")
    print(layers)

    for layer_name in layers["name"]:
        print("\n" + "-" * 100)
        print(f"LAYER: {layer_name}")

        gdf = gpd.read_file(path, layer=layer_name)

        print(f"Rows: {len(gdf):,}")
        print(f"Columns: {len(gdf.columns)}")
        print(f"CRS: {gdf.crs}")

        print("\nGeometry types:")
        print(gdf.geometry.geom_type.value_counts().to_string())

        print("\nColumns:")
        for column in gdf.columns:
            print(f"  - {column}")

        print("\nFirst 3 rows:")
        print(gdf.head(3).to_string())

        print("\nNumeric columns:")
        numeric = gdf.select_dtypes(include="number")

        if numeric.empty:
            print("No numeric columns found.")
        else:
            print(numeric.describe().transpose().to_string())


def main() -> None:
    print("LAEI 2022 TRAFFIC DATA INSPECTION")
    print("=" * 100)

    for path in [MAJOR_ROADS, MINOR_ROADS]:

        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        inspect_gpkg(path)

    print("\n" + "=" * 100)
    print("Traffic inspection complete.")


if __name__ == "__main__":
    main()