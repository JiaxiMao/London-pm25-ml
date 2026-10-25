from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "green_cover_500m.parquet"


def main() -> None:
    print("=" * 90)
    print("GREEN COVER OUTPUT VALIDATION")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Green Cover output not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    required = [
        "grid_id",
        "area_m2",
        "green_area_m2",
        "green_cover_pct",
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------

    print("\n--- MISSING VALUES ---")
    print(
        df[required]
        .isna()
        .sum()
        .to_string()
    )

    print("\n--- RANGE CHECKS ---")

    print(
        f"Minimum cell area: "
        f"{df['area_m2'].min():,.2f} m2"
    )

    print(
        f"Maximum cell area: "
        f"{df['area_m2'].max():,.2f} m2"
    )

    print(
        f"Minimum green area: "
        f"{df['green_area_m2'].min():,.2f} m2"
    )

    print(
        f"Maximum green area: "
        f"{df['green_area_m2'].max():,.2f} m2"
    )

    print(
        f"Minimum green cover: "
        f"{df['green_cover_pct'].min():.4f}%"
    )

    print(
        f"Maximum green cover: "
        f"{df['green_cover_pct'].max():.4f}%"
    )

    # ------------------------------------------------------------------
    # Impossible values
    # ------------------------------------------------------------------

    negative_green_area = (
        df["green_area_m2"] < 0
    ).sum()

    negative_percentage = (
        df["green_cover_pct"] < 0
    ).sum()

    over_100_percentage = (
        df["green_cover_pct"] > 100
    ).sum()

    over_area = (
        df["green_area_m2"]
        > df["area_m2"] + 1e-6
    )

    print("\n--- IMPOSSIBLE VALUES ---")

    print(
        f"Negative green-area cells: "
        f"{negative_green_area:,}"
    )

    print(
        f"Negative percentage cells: "
        f"{negative_percentage:,}"
    )

    print(
        f"Percentage > 100%: "
        f"{over_100_percentage:,}"
    )

    print(
        f"Green area > cell area: "
        f"{over_area.sum():,}"
    )

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    print("\n--- GREEN COVER DISTRIBUTION ---")

    bins = [
        -0.0001,
        0,
        10,
        25,
        50,
        75,
        90,
        100.0001,
    ]

    labels = [
        "0%",
        "0-10%",
        "10-25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "90-100%",
    ]

    categories = pd.cut(
        df["green_cover_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    print(
        categories
        .value_counts(sort=False)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Area-weighted mean
    # ------------------------------------------------------------------

    total_area = df["area_m2"].sum()

    weighted_green_area = (
        df["green_area_m2"].sum()
    )

    weighted_percentage = (
        weighted_green_area
        / total_area
        * 100
    )

    print("\n--- AREA-WEIGHTED RESULT ---")

    print(
        f"Total modelling-cell area: "
        f"{total_area:,.2f} m2"
    )

    print(
        f"Total mapped green area: "
        f"{weighted_green_area:,.2f} m2"
    )

    print(
        f"Area-weighted green cover: "
        f"{weighted_percentage:.4f}%"
    )

    # ------------------------------------------------------------------
    # Highest and lowest coverage cells
    # ------------------------------------------------------------------

    print("\n--- HIGHEST GREEN-COVER CELLS ---")

    print(
        df[
            [
                "grid_id",
                "area_m2",
                "green_area_m2",
                "green_cover_pct",
            ]
        ]
        .sort_values(
            "green_cover_pct",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\n--- LOWEST GREEN-COVER CELLS ---")

    print(
        df[
            [
                "grid_id",
                "area_m2",
                "green_area_m2",
                "green_cover_pct",
            ]
        ]
        .sort_values(
            "green_cover_pct",
            ascending=True,
        )
        .head(10)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)

    if (
        negative_green_area == 0
        and negative_percentage == 0
        and over_100_percentage == 0
        and over_area.sum() == 0
    ):
        print("VALIDATION STATUS: PASS")

    else:
        print("VALIDATION STATUS: REVIEW REQUIRED")

    print("=" * 90)


if __name__ == "__main__":
    main()