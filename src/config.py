from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ---------------------------------------------------------------------------
# Raw source files
# ---------------------------------------------------------------------------

PM25_CSV_ZIP = RAW_DIR / "LAEI2022-Concentrations-Data-CSV.zip"
PM25_ESRI_ZIP = RAW_DIR / "LAEI2022-Concentrations-Data-ESRI.zip"

TRAFFIC_ZIP = (
    RAW_DIR
    / "LAEI2022-Traffic-Flows-and-Speeds-Major-and-Minor-Roads-GIS.zip"
)

GREEN_COVER_ZIP = RAW_DIR / "Green_Cover_2024.zip"


# ---------------------------------------------------------------------------
# Spatial settings
# ---------------------------------------------------------------------------

# British National Grid, used by the London spatial datasets.
CRS_EPSG = 27700

# Proposed modelling grid size.
GRID_SIZE_METRES = 500


# ---------------------------------------------------------------------------
# Study definition
# ---------------------------------------------------------------------------

STUDY_YEAR = 2022
TARGET_VARIABLE = "pm25"


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

MODELLING_DATASET = PROCESSED_DIR / "london_pm25_modelling_dataset.parquet"
MODELLING_DATASET_CSV = PROCESSED_DIR / "london_pm25_modelling_dataset.csv"

MODELLING_GRID = PROCESSED_DIR / "london_500m_grid.gpkg"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create project output directories if they do not already exist."""
    for directory in [
        PROCESSED_DIR,
        FIGURES_DIR,
        OUTPUTS_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data:     {RAW_DIR}")
    print(f"Processed:    {PROCESSED_DIR}")