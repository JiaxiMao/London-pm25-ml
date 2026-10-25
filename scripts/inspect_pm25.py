from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PM25_CSV_ZIP


def main() -> None:
    print("=" * 90)
    print("LAEI 2022 PM2.5 CSV INSPECTION")
    print("=" * 90)

    print(f"ZIP file: {PM25_CSV_ZIP}")

    if not PM25_CSV_ZIP.exists():
        raise FileNotFoundError(
            f"PM2.5 CSV ZIP not found: {PM25_CSV_ZIP}"
        )

    with ZipFile(PM25_CSV_ZIP, "r") as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        print("\nCSV files found:")
        for name in csv_members:
            print(f"  - {name}")

        pm25_files = [
            name
            for name in csv_members
            if "PM25" in Path(name).name.upper()
        ]

        if not pm25_files:
            raise RuntimeError("Could not identify the PM2.5 CSV.")

        pm25_member = pm25_files[0]

        print(f"\nSelected file: {pm25_member}")

        with archive.open(pm25_member) as file:
            sample = pd.read_csv(
                file,
                nrows=10,
                low_memory=False
            )

    print("\nColumns:")
    for column in sample.columns:
        print(f"  - {column}")

    print("\nSample rows:")
    print(sample.to_string(index=False))

    print("\nData types:")
    print(sample.dtypes.to_string())

    print("\nShape of sample:")
    print(sample.shape)

    print("\nMissing values in sample:")
    print(sample.isna().sum().to_string())

    print("\nBasic numeric summary:")
    print(sample.describe(include="all").transpose().to_string())

    print("\nInspection complete.")


if __name__ == "__main__":
    main()