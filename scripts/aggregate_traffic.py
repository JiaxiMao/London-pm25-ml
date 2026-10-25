from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, CRS_EPSG


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAFFIC_DIR = PROCESSED_DIR / "traffic"

MAJOR_ROADS = (
    TRAFFIC_DIR
    / "LAEI2022-major-roads-flows-and-speeds.gpkg"
)

MINOR_ROADS = (
    TRAFFIC_DIR
    / "LAEI2022-minor-roads-flows-and-speeds.gpkg"
)

GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"

OUTPUT_FILE = PROCESSED_DIR / "traffic_500m.parquet"
OUTPUT_CSV = PROCESSED_DIR / "traffic_500m.csv"


# ---------------------------------------------------------------------------
# Vehicle fields
# ---------------------------------------------------------------------------

MAJOR_VEHICLE_FIELDS = [
    "AADF.motorcycles",
    "AADF.taxis",
    "AADF.cars",
    "AADF.phvs",
    "AADF.lgvs",
    "AADF.hgvs.rigid",
    "AADF.hgvs.articulated",
    "AADF.tfl.buses",
    "AADF.non.tfl.buses",
    "AADF.coaches",
]

MAJOR_HGV_FIELDS = [
    "AADF.hgvs.rigid",
    "AADF.hgvs.articulated",
]

MINOR_VEHICLE_FIELDS = [
    "vkm.motorcycles",
    "vkm.taxis",
    "vkm.cars",
    "vkm.phvs",
    "vkm.lgvs",
    "vkm.hgvs.rigid",
    "vkm.hgvs.articulated",
    "vkm.tfl.buses",
    "vkm.non.tfl.buses",
    "vkm.coaches",
]

MINOR_HGV_FIELDS = [
    "vkm.hgvs.rigid",
    "vkm.hgvs.articulated",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")


def numeric_fill(gdf: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    """Convert selected columns to numeric and replace missing values with 0."""
    for column in columns:
        gdf[column] = pd.to_numeric(
            gdf[column],
            errors="coerce",
        ).fillna(0)

    return gdf


def load_grid() -> gpd.GeoDataFrame:
    """Load the London 500 m modelling grid."""
    require_file(GRID_FILE)

    grid = gpd.read_file(
        GRID_FILE,
        layer="london_500m_grid",
    )

    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(epsg=CRS_EPSG)

    return grid


def prepare_major_roads() -> gpd.GeoDataFrame:
    """Load and prepare the major-road layer."""
    print("\nLoading major roads...")

    major = gpd.read_file(
        MAJOR_ROADS,
        layer="major-roads-vehicle-flows-and-speeds",
    )

    print(f"Major-road segments: {len(major):,}")
    print(f"Major-road CRS: {major.crs}")

    if major.crs.to_epsg() != CRS_EPSG:
        major = major.to_crs(epsg=CRS_EPSG)

    # Restrict to actual London coverage where possible.
    major = major[
        major["laei.zone"].astype(str).str.upper().eq("GLA")
        | major["borough"].astype(str).str.upper().ne("NON-GLA")
    ].copy()

    # Convert vehicle fields to numeric.
    major = numeric_fill(
        major,
        MAJOR_VEHICLE_FIELDS,
    )

    major["major_total_aadf"] = major[
        MAJOR_VEHICLE_FIELDS
    ].sum(axis=1)

    major["major_hgv_aadf"] = major[
        MAJOR_HGV_FIELDS
    ].sum(axis=1)

    major["length_m_original"] = pd.to_numeric(
        major["length.m"],
        errors="coerce",
    ).fillna(
        major.geometry.length
    )

    # Guard against zero/invalid lengths.
    major["length_m_original"] = major[
        "length_m_original"
    ].clip(lower=0)

    major = major[
        [
            "major_total_aadf",
            "major_hgv_aadf",
            "length_m_original",
            "geometry",
        ]
    ]

    return major


def aggregate_major_roads(
    grid: gpd.GeoDataFrame,
    major: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatially allocate major-road traffic to 500 m cells.

    For a road segment crossing multiple cells:
      - road length is allocated by the segment portion in each cell;
      - AADF is converted into vehicle-metres/day using the intersected length.

    This avoids simply assigning a whole road segment to a single cell.
    """

    print("\nAggregating major-road traffic...")

    candidates = gpd.sjoin(
        major,
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    print(
        f"Candidate major-road/grid intersections: "
        f"{len(candidates):,}"
    )

    if candidates.empty:
        raise RuntimeError(
            "No major roads intersected the London 500 m grid."
        )

    # The join keeps the major-road geometry as the active geometry.
    grid_subset = grid[
        ["grid_id", "geometry"]
    ]

    # Join-created index identifies the matching grid geometry.
    candidates = candidates.rename(
        columns={
            "index_right": "grid_index",
        }
    )

    candidates["grid_id"] = (
        candidates["grid_id"]
        .astype(str)
    )

    # Attach the corresponding grid geometry.
    grid_geometry = (
        grid_subset
        .reset_index()
        .rename(columns={"index": "grid_index"})
    )

    candidates = candidates.merge(
        grid_geometry[
            ["grid_index", "geometry"]
        ].rename(columns={"geometry": "grid_geometry"}),
        on="grid_index",
        how="left",
    )

    # Compute actual road portion inside each grid cell.
    intersections = candidates.geometry.intersection(
        candidates["grid_geometry"]
    )

    intersection_length = intersections.length

    candidates["intersection_length_m"] = intersection_length

    # Remove zero-length numerical artefacts.
    candidates = candidates[
        candidates["intersection_length_m"] > 0
    ].copy()

    # Allocate road length.
    candidates["allocated_road_length_m"] = (
        candidates["intersection_length_m"]
    )

    # Traffic exposure:
    # AADF vehicles/day * road length (m)
    candidates["vehicle_m_per_day"] = (
        candidates["major_total_aadf"]
        * candidates["intersection_length_m"]
    )

    candidates["hgv_vehicle_m_per_day"] = (
        candidates["major_hgv_aadf"]
        * candidates["intersection_length_m"]
    )

    result = (
        candidates
        .groupby("grid_id", as_index=False)
        .agg(
            major_road_length_m=(
                "allocated_road_length_m",
                "sum",
            ),
            major_vehicle_m_per_day=(
                "vehicle_m_per_day",
                "sum",
            ),
            major_hgv_vehicle_m_per_day=(
                "hgv_vehicle_m_per_day",
                "sum",
            ),
        )
    )

    # Derived mean traffic intensity along major road length.
    result["major_mean_aadf"] = (
        result["major_vehicle_m_per_day"]
        / result["major_road_length_m"]
    )

    result["major_hgv_mean_aadf"] = (
        result["major_hgv_vehicle_m_per_day"]
        / result["major_road_length_m"]
    )

    return result


def prepare_minor_roads() -> gpd.GeoDataFrame:
    """Load and correctly reproject the minor-road 1 km cells."""
    print("\nLoading minor roads...")

    minor = gpd.read_file(
        MINOR_ROADS,
        layer="minor-roads-vehicle-flows-and-speeds",
    )

    print(f"Minor-road cells: {len(minor):,}")
    print(f"Source CRS: {minor.crs}")

    # The source geometry is WGS84 lon/lat. Transform to British National Grid.
    minor = minor.to_crs(epsg=CRS_EPSG)

    print(f"Working CRS: {minor.crs}")

    minor = numeric_fill(
        minor,
        MINOR_VEHICLE_FIELDS,
    )

    minor["minor_total_vkm"] = minor[
        MINOR_VEHICLE_FIELDS
    ].sum(axis=1)

    minor["minor_hgv_vkm"] = minor[
        MINOR_HGV_FIELDS
    ].sum(axis=1)

    # Preserve original polygon area before intersections.
    minor["minor_original_area_m2"] = (
        minor.geometry.area
    )

    minor = minor[
        [
            "minor_total_vkm",
            "minor_hgv_vkm",
            "minor_original_area_m2",
            "geometry",
        ]
    ]

    return minor


def aggregate_minor_roads(
    grid: gpd.GeoDataFrame,
    minor: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Allocate minor-road VKT to 500 m cells according to
    area overlap with the source 1 km minor-road cells.

    This is an explicit spatial allocation assumption because
    the source VKT values are reported at approximately 1 km-cell level.
    """

    print("\nAggregating minor-road traffic...")

    candidates = gpd.sjoin(
        minor,
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    print(
        f"Candidate minor-road/grid intersections: "
        f"{len(candidates):,}"
    )

    if candidates.empty:
        raise RuntimeError(
            "No minor roads intersected the London 500 m grid."
        )

    grid_geometry = (
        grid[["grid_id", "geometry"]]
        .reset_index()
        .rename(columns={"index": "grid_index"})
    )

    candidates = candidates.rename(
        columns={
            "index_right": "grid_index",
        }
    )

    candidates = candidates.merge(
        grid_geometry[
            ["grid_index", "geometry"]
        ].rename(columns={"geometry": "grid_geometry"}),
        on="grid_index",
        how="left",
    )

    intersections = candidates.geometry.intersection(
        candidates["grid_geometry"]
    )

    candidates["intersection_area_m2"] = (
        intersections.area
    )

    candidates = candidates[
        candidates["intersection_area_m2"] > 0
    ].copy()

    candidates["area_fraction"] = (
        candidates["intersection_area_m2"]
        / candidates["minor_original_area_m2"]
    )

    # Protect against tiny numerical overshoots.
    candidates["area_fraction"] = (
        candidates["area_fraction"]
        .clip(lower=0, upper=1)
    )

    candidates["allocated_minor_vkm"] = (
        candidates["minor_total_vkm"]
        * candidates["area_fraction"]
    )

    candidates["allocated_minor_hgv_vkm"] = (
        candidates["minor_hgv_vkm"]
        * candidates["area_fraction"]
    )

    result = (
        candidates
        .groupby("grid_id", as_index=False)
        .agg(
            minor_vkm_total=(
                "allocated_minor_vkm",
                "sum",
            ),
            minor_hgv_vkm=(
                "allocated_minor_hgv_vkm",
                "sum",
            ),
        )
    )

    return result


def main() -> None:
    print("=" * 90)
    print("AGGREGATING TRAFFIC TO THE 500 M LONDON GRID")
    print("=" * 90)

    require_file(MAJOR_ROADS)
    require_file(MINOR_ROADS)
    require_file(GRID_FILE)

    grid = load_grid()

    major = prepare_major_roads()
    major_result = aggregate_major_roads(
        grid,
        major,
    )

    minor = prepare_minor_roads()
    minor_result = aggregate_minor_roads(
        grid,
        minor,
    )

    # -----------------------------------------------------------------------
    # Merge major and minor traffic predictors.
    # -----------------------------------------------------------------------

    traffic = grid[
        ["grid_id"]
    ].copy()

    traffic = traffic.merge(
        major_result,
        on="grid_id",
        how="left",
    )

    traffic = traffic.merge(
        minor_result,
        on="grid_id",
        how="left",
    )

    numeric_columns = [
        "major_road_length_m",
        "major_vehicle_m_per_day",
        "major_hgv_vehicle_m_per_day",
        "major_mean_aadf",
        "major_hgv_mean_aadf",
        "minor_vkm_total",
        "minor_hgv_vkm",
    ]

    traffic[numeric_columns] = (
        traffic[numeric_columns]
        .fillna(0)
    )

    # -----------------------------------------------------------------------
    # Quality checks
    # -----------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("--- TRAFFIC QUALITY CHECK ---")
    print("=" * 90)

    print(
        f"Modelling cells: {len(traffic):,}"
    )

    print("\nMissing values:")
    print(
        traffic.isna()
        .sum()
        .to_string()
    )

    print("\nSummary:")
    print(
        traffic[numeric_columns]
        .describe()
        .transpose()
        .to_string()
    )

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    traffic.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    traffic.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("OUTPUTS")
    print("=" * 90)

    print(f"Parquet: {OUTPUT_FILE}")
    print(f"CSV:     {OUTPUT_CSV}")

    print("\nTraffic aggregation complete.")


if __name__ == "__main__":
    main()