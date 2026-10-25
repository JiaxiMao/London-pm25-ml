from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PM25_CSV_ZIP


def main() -> None:
    print("=" * 90)
    print("LAEI 2022 PM2.5 FULL DATA PROFILE")
    print("=" * 90)

    with ZipFile(PM25_CSV_ZIP, "r") as archive:
        pm25_member = "CSV/LAEI2022_V1_PM25.csv"

        if pm25_member not in archive.namelist():
            raise FileNotFoundError(
                f"{pm25_member} not found in {PM25_CSV_ZIP}"
            )

        print(f"Reading: {pm25_member}")
        print("This may take a little while because the CSV is large.")

        with archive.open(pm25_member) as file:
            df = pd.read_csv(
                file,
                usecols=["x", "y", "conc"],
                low_memory=False
            )

    print("\n--- DATASET SIZE ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- COORDINATES ---")
    print(f"x min: {df['x'].min():,.0f}")
    print(f"x max: {df['x'].max():,.0f}")
    print(f"y min: {df['y'].min():,.0f}")
    print(f"y max: {df['y'].max():,.0f}")

    print("\n--- UNIQUE COORDINATES ---")
    print(f"Unique x values: {df['x'].nunique():,}")
    print(f"Unique y values: {df['y'].nunique():,}")

    print("\n--- PM2.5 ---")
    print(f"Minimum: {df['conc'].min():.4f}")
    print(f"Maximum: {df['conc'].max():.4f}")
    print(f"Mean:    {df['conc'].mean():.4f}")
    print(f"Median:  {df['conc'].median():.4f}")
    print(f"Std:     {df['conc'].std():.4f}")

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().to_string())

    print("\n--- DUPLICATES ---")
    duplicate_xy = df.duplicated(subset=["x", "y"]).sum()
    print(f"Duplicate x/y pairs: {duplicate_xy:,}")

    print("\n--- COORDINATE SPACING SAMPLE ---")

    unique_x = sorted(df["x"].unique())
    unique_y = sorted(df["y"].unique())

    x_diffs = pd.Series(unique_x).diff().dropna()
    y_diffs = pd.Series(unique_y).diff().dropna()

    print("Most common x spacing:")
    print(x_diffs.value_counts().head(10))

    print("\nMost common y spacing:")
    print(y_diffs.value_counts().head(10))

    print("\n--- QUANTILES ---")
    print(
        df["conc"]
        .quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        .to_string()
    )

    print("\nProfile complete.")


if __name__ == "__main__":
    main()