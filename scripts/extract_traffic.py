from pathlib import Path
from zipfile import ZipFile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, PROCESSED_DIR


def main() -> None:
    zip_path = (
        RAW_DIR
        / "LAEI2022-Traffic-Flows-and-Speeds-Major-and-Minor-Roads-GIS.zip"
    )

    output_dir = PROCESSED_DIR / "traffic"

    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if member.endswith(".gpkg"):
                target = Path(member).name

                print(f"Extracting: {target}")

                with archive.open(member) as source:
                    output_path = output_dir / target

                    with open(output_path, "wb") as destination:
                        destination.write(source.read())

    print("\nTraffic extraction complete.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()