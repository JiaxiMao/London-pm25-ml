from pathlib import Path
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CRS_EPSG, FIGURES_DIR, OUTPUTS_DIR, PROCESSED_DIR


GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"
PREDICTIONS_FILE = OUTPUTS_DIR / "test_set_predictions.csv"
BOUNDARY_FILE = PROJECT_ROOT / "data" / "raw" / "London_Boroughs.gpkg"
GRID_LAYER = "london_500m_grid"
DPI = 300
FIGSIZE = (10, 8)


def add_boundaries(ax, boroughs):
    boroughs.boundary.plot(ax=ax, linewidth=0.35, edgecolor="black", alpha=0.65)


def save_map(filename, title, source_text):
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_axis_off()
    fig.text(0.5, 0.025, source_text, ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=[0, 0.045, 1, 1])
    output = FIGURES_DIR / filename
    fig.savefig(output, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def regression_plot(observed, predicted, model_name, filename):
    r2 = np.corrcoef(observed, predicted)[0, 1] ** 2
    residuals = observed - predicted
    min_value = min(observed.min(), predicted.min())
    max_value = max(observed.max(), predicted.max())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(observed, predicted, s=14, alpha=0.45)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed modelled PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"{model_name}: Observed vs Predicted", fontsize=15, fontweight="bold")
    ax.text(
        0.03,
        0.95,
        f"R² = {r2:.3f}\nMAE = {np.mean(np.abs(residuals)):.3f}\nRMSE = {np.sqrt(np.mean(residuals ** 2)):.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    output = FIGURES_DIR / filename
    fig.savefig(output, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    print("=" * 90)
    print("MODEL DIAGNOSTICS AND PREDICTION MAPS")
    print("=" * 90)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{PREDICTIONS_FILE}")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    print(f"\nTest-set rows: {len(predictions):,}")
    required_columns = [
        "grid_id", "centroid_x", "centroid_y", "observed_pm25",
        "linear_prediction", "random_forest_prediction", "linear_error", "random_forest_error",
    ]
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    regression_plot(
        predictions["observed_pm25"].to_numpy(), predictions["linear_prediction"].to_numpy(),
        "Linear Regression", "06_linear_regression_observed_vs_predicted.png",
    )
    regression_plot(
        predictions["observed_pm25"].to_numpy(), predictions["random_forest_prediction"].to_numpy(),
        "Random Forest", "07_random_forest_observed_vs_predicted.png",
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(predictions["linear_error"], bins=40, alpha=0.55, label="Linear Regression")
    ax.hist(predictions["random_forest_error"], bins=40, alpha=0.55, label="Random Forest")
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed - predicted PM2.5 (µg/m³)")
    ax.set_ylabel("Number of test cells")
    ax.set_title("Prediction Error Distribution", fontsize=15, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    output = FIGURES_DIR / "08_prediction_error_distribution.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")

    print("\nLoading London grid...")
    grid = gpd.read_file(GRID_FILE, layer=GRID_LAYER)
    if grid.crs is None:
        raise ValueError("Grid has no CRS.")
    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(epsg=CRS_EPSG)

    boroughs = gpd.read_file(BOUNDARY_FILE)
    if boroughs.crs is None:
        raise ValueError("Borough boundary layer has no CRS.")
    if boroughs.crs.to_epsg() != CRS_EPSG:
        boroughs = boroughs.to_crs(epsg=CRS_EPSG)

    mapped = grid.merge(
        predictions[["grid_id", "observed_pm25", "linear_prediction", "random_forest_prediction", "random_forest_error"]],
        on="grid_id", how="inner", validate="one_to_one",
    )
    print(f"Mapped test cells: {len(mapped):,}")

    pred_low, pred_high = mapped["random_forest_prediction"].quantile([0.02, 0.98])
    residual_low, residual_high = mapped["random_forest_error"].quantile([0.02, 0.98])

    ax = mapped.plot(
        column="random_forest_prediction", cmap="YlOrRd", figsize=(10, 10), linewidth=0,
        edgecolor="none", vmin=pred_low, vmax=pred_high, legend=True,
        legend_kwds={"label": "Predicted PM2.5 (µg/m³)", "shrink": 0.65},
    )
    add_boundaries(ax, boroughs)
    save_map(
        "09_random_forest_prediction_map.png", "Random Forest Prediction of PM2.5 Across London",
        "Source: LAEI 2022; Random Forest predictions on spatially held-out test cells.",
    )

    ax = mapped.plot(
        column="random_forest_error", cmap="RdBu_r", figsize=(10, 10), linewidth=0,
        edgecolor="none", vmin=residual_low, vmax=residual_high, legend=True,
        legend_kwds={"label": "Observed - predicted PM2.5 (µg/m³)", "shrink": 0.65},
    )
    add_boundaries(ax, boroughs)
    save_map(
        "10_random_forest_residual_map.png", "Random Forest Spatial Prediction Error",
        "Positive values indicate under-prediction; negative values indicate over-prediction.",
    )

    rf_error = predictions["random_forest_error"]
    diagnostic_summary = pd.DataFrame({
        "metric": [
            "test_cells", "rf_prediction_mean", "rf_prediction_std", "rf_residual_mean",
            "rf_residual_std", "rf_residual_mae", "rf_residual_rmse", "rf_residual_min", "rf_residual_max",
        ],
        "value": [
            len(predictions), predictions["random_forest_prediction"].mean(),
            predictions["random_forest_prediction"].std(), rf_error.mean(), rf_error.std(),
            rf_error.abs().mean(), np.sqrt(np.mean(rf_error ** 2)), rf_error.min(), rf_error.max(),
        ],
    })
    diagnostic_file = OUTPUTS_DIR / "model_diagnostics_summary.csv"
    diagnostic_summary.to_csv(diagnostic_file, index=False)
    print(f"\nSaved: {diagnostic_file}")

    print("\n" + "=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)
    print(f"Test cells: {len(predictions):,}")
    print(f"RF mean residual: {rf_error.mean():.4f}")
    print(f"RF MAE: {rf_error.abs().mean():.4f}")
    print(f"RF RMSE: {np.sqrt(np.mean(rf_error ** 2)):.4f}")
    print(f"RF minimum residual: {rf_error.min():.4f}")
    print(f"RF maximum residual: {rf_error.max():.4f}")
    print("\nModel diagnostics complete.")


if __name__ == "__main__":
    main()