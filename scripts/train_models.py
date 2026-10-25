from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "london_pm25_final.parquet"
RESULTS_FILE = OUTPUTS_DIR / "model_comparison.csv"
PREDICTIONS_FILE = OUTPUTS_DIR / "test_set_predictions.csv"
RF_IMPORTANCE_FILE = OUTPUTS_DIR / "random_forest_feature_importance.csv"
LINEAR_COEFFICIENT_FILE = OUTPUTS_DIR / "linear_regression_coefficients.csv"
RF_MODEL_FILE = OUTPUTS_DIR / "random_forest_model.joblib"
LINEAR_MODEL_FILE = OUTPUTS_DIR / "linear_regression_model.joblib"

RANDOM_STATE = 42
SPATIAL_BLOCK_SIZE = 5000
TEST_BLOCK_FRACTION = 0.20

FEATURES = [
    "major_mean_aadf",
    "major_hgv_mean_aadf",
    "major_road_length_m",
    "minor_vkm_total",
    "minor_hgv_vkm",
    "green_cover_pct",
]
TARGET = "pm25_mean"


def calculate_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def create_spatial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["block_x"] = np.floor(df["centroid_x"] / SPATIAL_BLOCK_SIZE).astype(int)
    df["block_y"] = np.floor(df["centroid_y"] / SPATIAL_BLOCK_SIZE).astype(int)
    df["spatial_block"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

    blocks = df["spatial_block"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(blocks)
    n_test_blocks = max(1, int(np.ceil(len(blocks) * TEST_BLOCK_FRACTION)))
    test_blocks = set(blocks[:n_test_blocks])

    train = df[~df["spatial_block"].isin(test_blocks)].copy()
    test = df[df["spatial_block"].isin(test_blocks)].copy()

    print(f"Total spatial blocks: {len(blocks):,}")
    print(f"Training blocks: {len(blocks) - n_test_blocks:,}")
    print(f"Testing blocks: {n_test_blocks:,}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    return train, test


def main() -> None:
    print("=" * 90)
    print("LONDON PM2.5 MACHINE-LEARNING MODELS")
    print("=" * 90)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Final modelling dataset not found:\n{INPUT_FILE}")

    print("\nLoading modelling dataset...")
    df = pd.read_parquet(INPUT_FILE)
    required = ["grid_id", TARGET, "centroid_x", "centroid_y", *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    numeric_columns = [TARGET, "centroid_x", "centroid_y", *FEATURES]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    print(f"Usable modelling rows: {len(df):,}")

    print("\nCreating spatial holdout...")
    train, test = create_spatial_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print("\n" + "-" * 90)
    print("LINEAR REGRESSION")
    print("-" * 90)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_metrics = calculate_metrics(y_test, linear_pred)
    print(f"R2:   {linear_metrics['R2']:.4f}")
    print(f"RMSE: {linear_metrics['RMSE']:.4f}")
    print(f"MAE:  {linear_metrics['MAE']:.4f}")

    coefficients = pd.DataFrame({"feature": FEATURES, "coefficient": linear_model.coef_})
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(LINEAR_COEFFICIENT_FILE, index=False)
    joblib.dump(linear_model, LINEAR_MODEL_FILE)

    print("\n" + "-" * 90)
    print("RANDOM FOREST REGRESSION")
    print("-" * 90)
    random_forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    rf_pred = random_forest.predict(X_test)
    rf_metrics = calculate_metrics(y_test, rf_pred)
    print(f"R2:   {rf_metrics['R2']:.4f}")
    print(f"RMSE: {rf_metrics['RMSE']:.4f}")
    print(f"MAE:  {rf_metrics['MAE']:.4f}")

    importance = pd.DataFrame({"feature": FEATURES, "importance": random_forest.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(RF_IMPORTANCE_FILE, index=False)
    joblib.dump(random_forest, RF_MODEL_FILE)

    comparison = pd.DataFrame([
        {"model": "Linear Regression", **linear_metrics},
        {"model": "Random Forest", **rf_metrics},
    ])
    comparison.to_csv(RESULTS_FILE, index=False)

    predictions = test[["grid_id", "centroid_x", "centroid_y", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "observed_pm25"})
    predictions["linear_prediction"] = linear_pred
    predictions["random_forest_prediction"] = rf_pred
    predictions["linear_error"] = predictions["observed_pm25"] - predictions["linear_prediction"]
    predictions["random_forest_error"] = predictions["observed_pm25"] - predictions["random_forest_prediction"]
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nRandom Forest feature importance:")
    print(importance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nOutputs:")
    print(f"Model comparison: {RESULTS_FILE}")
    print(f"Test predictions: {PREDICTIONS_FILE}")
    print(f"RF importance: {RF_IMPORTANCE_FILE}")
    print(f"Linear coefficients: {LINEAR_COEFFICIENT_FILE}")
    print(f"RF model: {RF_MODEL_FILE}")
    print(f"Linear model: {LINEAR_MODEL_FILE}")
    print("\nModel training complete.")


if __name__ == "__main__":
    main()