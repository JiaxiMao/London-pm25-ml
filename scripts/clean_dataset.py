from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR


INPUT_FILE = (
    PROCESSED_DIR
    / "london_pm25_modelling_dataset.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "london_pm25_final.parquet"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "london_pm25_final.csv"
)


TARGET = "pm25_mean"

PREDICTORS = [
    "major_mean_aadf",
    "major_hgv_mean_aadf",
    "major_road_length_m",
    "minor_vkm_total",
    "minor_hgv_vkm",
    "green_cover_pct",
    "centroid_x",
    "centroid_y",
]


def main() -> None:
    print("=" * 90)
    print("CLEANING FINAL LONDON PM2.5 MODELLING DATASET")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nInput rows: {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    required = [
        "grid_id",
        TARGET,
        *PREDICTORS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ------------------------------------------------------------------
    # Keep only the modelling fields
    # ------------------------------------------------------------------

    df = df[required].copy()

    # ------------------------------------------------------------------
    # Remove rows with missing target/predictor values
    # ------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            TARGET,
            *PREDICTORS,
        ]
    ).copy()

    removed = before - len(df)

    print(
        f"Rows removed due to missing values: "
        f"{removed:,}"
    )

    # ------------------------------------------------------------------
    # Check duplicate IDs
    # ------------------------------------------------------------------

    duplicates = (
        df["grid_id"]
        .duplicated()
        .sum()
    )

    if duplicates:
        raise ValueError(
            f"Duplicate grid IDs detected: {duplicates:,}"
        )

    # ------------------------------------------------------------------
    # Check numeric types
    # ------------------------------------------------------------------

    numeric_columns = [
        TARGET,
        *PREDICTORS,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Final missing-value check
    # ------------------------------------------------------------------

    missing_after = (
        df[numeric_columns]
        .isna()
        .sum()
    )

    if missing_after.sum() > 0:
        raise ValueError(
            "Missing/non-numeric values remain:\n"
            f"{missing_after}"
        )

    # ------------------------------------------------------------------
    # Basic range checks
    # ------------------------------------------------------------------

    if (
        (df["green_cover_pct"] < 0)
        | (df["green_cover_pct"] > 100)
    ).any():
        raise ValueError(
            "Green cover contains values outside 0-100%."
        )

    if (df[TARGET] <= 0).any():
        print(
            "WARNING: Some PM2.5 values are <= 0."
        )

    # ------------------------------------------------------------------
    # Sort by grid ID
    # ------------------------------------------------------------------

    df = df.sort_values(
        "grid_id"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL DATASET")
    print("=" * 90)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"\nVariables:\n"
        + "\n".join(
            f"  - {column}"
            for column in df.columns
        )
    )

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .to_string()
    )

    print("\nTarget summary:")
    print(
        df[TARGET]
        .describe()
        .to_string()
    )

    print("\nPredictor summary:")
    print(
        df[
            PREDICTORS
        ]
        .describe()
        .transpose()
        .to_string()
    )

    print("\nOutputs:")
    print(
        f"Parquet: {OUTPUT_FILE}"
    )

    print(
        f"CSV:     {OUTPUT_CSV}"
    )

    print(
        "\nDataset cleaning complete."
    )


if __name__ == "__main__":
    main()