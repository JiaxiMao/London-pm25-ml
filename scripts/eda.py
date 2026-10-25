from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "london_pm25_final.parquet"


def save_figure(filename: str) -> None:
    """Save and close the current matplotlib figure."""
    output = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def save_histogram(
    series: pd.Series,
    filename: str,
    xlabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.hist(series, bins=40)
    plt.xlabel(xlabel)
    plt.ylabel("Number of 500 m grid cells")
    plt.title(title)
    save_figure(filename)


def save_scatter(
    x: pd.Series,
    y: pd.Series,
    filename: str,
    xlabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel("Annual mean PM2.5 (µg/m³)")
    plt.title(title)
    save_figure(filename)


def main() -> None:
    print("=" * 90)
    print("LONDON PM2.5 EXPLORATORY DATA ANALYSIS")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Final dataset not found:\n{INPUT_FILE}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_FILE)
    print(f"\nRows: {len(df):,}")

    target = "pm25_mean"
    predictors = [
        "major_mean_aadf",
        "major_hgv_mean_aadf",
        "major_road_length_m",
        "minor_vkm_total",
        "minor_hgv_vkm",
        "green_cover_pct",
    ]
    analysis_columns = [target, *predictors]

    print("\n--- DESCRIPTIVE STATISTICS ---")
    summary = df[analysis_columns].describe().transpose()
    print(summary.to_string())

    summary_file = PROCESSED_DIR / "eda_summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"\nSaved statistics: {summary_file}")

    print("\n--- CORRELATION MATRIX ---")
    correlation = df[analysis_columns].corr()
    print(correlation.to_string())

    correlation_file = PROCESSED_DIR / "eda_correlation_matrix.csv"
    correlation.to_csv(correlation_file)
    print(f"\nSaved correlation matrix: {correlation_file}")

    save_histogram(
        df[target],
        "01_pm25_distribution.png",
        "Annual mean PM2.5 (µg/m³)",
        "Distribution of Modelled PM2.5 Across London",
    )
    save_histogram(
        df["green_cover_pct"],
        "02_green_cover_distribution.png",
        "Green cover (%)",
        "Distribution of Green Cover Across London",
    )
    save_histogram(
        df["major_mean_aadf"],
        "08_major_traffic_distribution.png",
        "Mean AADF on major roads (vehicles/day)",
        "Distribution of Major-Road Traffic Across London",
    )
    save_histogram(
        df["major_hgv_mean_aadf"],
        "09_hgv_traffic_distribution.png",
        "Mean HGV AADF on major roads (vehicles/day)",
        "Distribution of Heavy-Goods Vehicle Traffic Across London",
    )
    save_histogram(
        df["minor_vkm_total"],
        "10_minor_traffic_distribution.png",
        "Minor-road vehicle-kilometres",
        "Distribution of Minor-Road Vehicle Activity Across London",
    )

    save_scatter(
        df["green_cover_pct"],
        df[target],
        "03_pm25_vs_green_cover.png",
        "Green cover (%)",
        "PM2.5 and Green Cover",
    )
    save_scatter(
        df["major_mean_aadf"],
        df[target],
        "04_pm25_vs_major_traffic.png",
        "Mean AADF on major roads (vehicles/day)",
        "PM2.5 and Major-Road Traffic",
    )
    save_scatter(
        df["major_hgv_mean_aadf"],
        df[target],
        "05_pm25_vs_hgv_traffic.png",
        "Mean HGV AADF on major roads (vehicles/day)",
        "PM2.5 and Heavy-Goods Vehicle Traffic",
    )
    save_scatter(
        df["minor_vkm_total"],
        df[target],
        "06_pm25_vs_minor_traffic.png",
        "Minor-road vehicle-kilometres",
        "PM2.5 and Minor-Road Vehicle Activity",
    )

    df["log_major_traffic"] = (df["major_mean_aadf"] + 1).apply(__import__("math").log)
    save_scatter(
        df["log_major_traffic"],
        df[target],
        "07_pm25_vs_log_major_traffic.png",
        "log(1 + major-road mean AADF)",
        "PM2.5 and Major-Road Traffic (Log Scale)",
    )

    print("\nEDA complete.")


if __name__ == "__main__":
    main()