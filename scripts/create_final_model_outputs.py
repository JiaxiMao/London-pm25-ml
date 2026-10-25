from pathlib import Path
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CRS_EPSG, FIGURES_DIR, OUTPUTS_DIR, PROCESSED_DIR


GRID_FILE = PROCESSED_DIR / "london_500m_grid.gpkg"
BOUNDARY_FILE = PROJECT_ROOT / "data" / "raw" / "London_Boroughs.gpkg"
FINAL_PREDICTIONS_FILE = OUTPUTS_DIR / "final_model_predictions_all_cells.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "three_model_test_predictions.csv"
MODEL_COMPARISON_FILE = OUTPUTS_DIR / "model_comparison_three_models.csv"
XGB_IMPORTANCE_FILE = OUTPUTS_DIR / "xgboost_feature_importance.csv"
GRID_LAYER = "london_500m_grid"
DPI = 300


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


def main():
    print("=" * 90)
    print("FINAL XGBOOST MODEL OUTPUTS")
    print("=" * 90)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        GRID_FILE, BOUNDARY_FILE, FINAL_PREDICTIONS_FILE,
        TEST_PREDICTIONS_FILE, MODEL_COMPARISON_FILE, XGB_IMPORTANCE_FILE,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    print("\nLoading London grid...")
    grid = gpd.read_file(GRID_FILE, layer=GRID_LAYER)
    if grid.crs is None:
        raise ValueError("London grid has no CRS.")
    if grid.crs.to_epsg() != CRS_EPSG:
        grid = grid.to_crs(epsg=CRS_EPSG)

    print("Loading London borough boundaries...")
    boroughs = gpd.read_file(BOUNDARY_FILE)
    if boroughs.crs is None:
        raise ValueError("Borough layer has no CRS.")
    if boroughs.crs.to_epsg() != CRS_EPSG:
        boroughs = boroughs.to_crs(epsg=CRS_EPSG)

    print("Loading full-London predictions...")
    predictions = pd.read_csv(FINAL_PREDICTIONS_FILE)
    required_prediction_columns = [
        "grid_id", "centroid_x", "centroid_y", "observed_pm25",
        "final_model_prediction", "final_model_residual",
    ]
    missing = [column for column in required_prediction_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    mapped = grid.merge(
        predictions[required_prediction_columns],
        on="grid_id", how="inner", validate="one_to_one",
    )
    print(f"Mapped cells: {len(mapped):,}")

    print("\nCreating full-London XGBoost prediction map...")
    prediction_low, prediction_high = mapped["final_model_prediction"].quantile([0.02, 0.98])
    ax = mapped.plot(
        column="final_model_prediction", cmap="YlOrRd", figsize=(10, 10),
        linewidth=0, edgecolor="none", vmin=prediction_low, vmax=prediction_high,
        legend=True, legend_kwds={"label": "Predicted PM2.5 (µg/m³)", "shrink": 0.65},
    )
    add_boundaries(ax, boroughs)
    save_map(
        "11_final_xgboost_pm25_prediction_map.png",
        "Final XGBoost Prediction of PM2.5 Across London",
        "Source: LAEI 2022; XGBoost model fitted to all 6,670 modelling cells.",
    )

    print("\nCreating XGBoost feature-importance figure...")
    importance = pd.read_csv(XGB_IMPORTANCE_FILE).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Predictor")
    ax.set_title("XGBoost Feature Importance", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    importance_figure = FIGURES_DIR / "12_xgboost_feature_importance.png"
    fig.savefig(importance_figure, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {importance_figure}")

    print("\nCreating model comparison figure...")
    comparison = pd.read_csv(MODEL_COMPARISON_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (column, label, higher_is_better) in zip(
        axes, [("R2", "R²", True), ("RMSE", "RMSE (µg/m³)", False), ("MAE", "MAE (µg/m³)", False)]
    ):
        ordered = comparison.sort_values(column, ascending=not higher_is_better)
        ax.bar(ordered["model"], ordered[column])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparison of PM2.5 Prediction Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    comparison_figure = FIGURES_DIR / "13_model_comparison.png"
    fig.savefig(comparison_figure, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {comparison_figure}")

    print("\nCreating XGBoost test-set observed-vs-predicted figure...")
    test = pd.read_csv(TEST_PREDICTIONS_FILE)
    if "xgboost_prediction" not in test.columns:
        raise ValueError("XGBoost prediction column not found in test predictions.")
    observed = test["observed_pm25"]
    predicted = test["xgboost_prediction"]
    min_value = min(observed.min(), predicted.min())
    max_value = max(observed.max(), predicted.max())
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(observed, predicted, s=16, alpha=0.45)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed modelled PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title("XGBoost: Observed vs Predicted on Spatial Holdout", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    observed_predicted_figure = FIGURES_DIR / "14_xgboost_observed_vs_predicted.png"
    fig.savefig(observed_predicted_figure, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {observed_predicted_figure}")

    best_model = comparison.sort_values("RMSE", ascending=True).iloc[0]
    summary = pd.DataFrame({
        "metric": [
            "final_model", "cells_predicted", "prediction_mean", "prediction_min", "prediction_max",
            "best_test_R2", "best_test_RMSE", "best_test_MAE",
        ],
        "value": [
            best_model["model"], len(predictions), predictions["final_model_prediction"].mean(),
            predictions["final_model_prediction"].min(), predictions["final_model_prediction"].max(),
            best_model["R2"], best_model["RMSE"], best_model["MAE"],
        ],
    })
    summary_file = OUTPUTS_DIR / "final_model_report_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")

    print("\n" + "=" * 90)
    print("FINAL MODEL OUTPUT SUMMARY")
    print("=" * 90)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {best_model['model']}")
    print(f"Predicted cells: {len(predictions):,}")
    print(f"Prediction mean: {predictions['final_model_prediction'].mean():.4f}")
    print(f"Prediction minimum: {predictions['final_model_prediction'].min():.4f}")
    print(f"Prediction maximum: {predictions['final_model_prediction'].max():.4f}")
    print("\nXGBoost feature importance:")
    print(importance.sort_values("importance", ascending=False).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFinal model output generation complete.")


if __name__ == "__main__":
    main()