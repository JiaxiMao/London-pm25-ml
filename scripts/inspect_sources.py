from pathlib import Path
from zipfile import ZipFile, BadZipFile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR


EXPECTED_FILES = [
    "Green_Cover_2024.zip",
    "LAEI2022-Concentrations-Data-CSV.zip",
    "LAEI2022-Concentrations-Data-ESRI.zip",
    "LAEI2022-Traffic-Flows-and-Speeds-Major-and-Minor-Roads-GIS.zip",
]


def format_size(size_bytes: int) -> str:
    """Convert bytes into a human-readable size."""
    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def inspect_zip(zip_path: Path) -> None:
    """Print the contents of a ZIP archive without extracting it."""
    print("=" * 100)
    print(f"FILE: {zip_path.name}")
    print(f"SIZE: {format_size(zip_path.stat().st_size)}")
    print()

    try:
        with ZipFile(zip_path, "r") as archive:
            members = archive.infolist()

            print(f"Number of archived files: {len(members)}")
            print("-" * 100)

            for member in members[:100]:
                print(
                    f"{member.filename} "
                    f"[{format_size(member.file_size)}]"
                )

            if len(members) > 100:
                print(
                    f"\n... {len(members) - 100} additional files not displayed."
                )

    except BadZipFile:
        print("ERROR: Invalid ZIP archive.")

    except Exception as exc:
        print(f"ERROR: {exc}")


def main() -> None:
    print("LONDON PM2.5 PROJECT - SOURCE DATA INSPECTION")
    print("=" * 100)
    print(f"Raw data directory: {RAW_DIR}")
    print()

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DIR}"
        )

    for filename in EXPECTED_FILES:
        zip_path = RAW_DIR / filename

        if not zip_path.exists():
            print(f"MISSING: {filename}")
            print()
            continue

        inspect_zip(zip_path)
        print()

    print("=" * 100)
    print("Inspection complete.")


if __name__ == "__main__":
    main()