from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "london_pm25_final.parquet"
COMPARISON_FILE = OUTPUTS_DIR / "model_comparison_three_models.csv"
TEST_PREDICTIONS_FILE = OUTPUTS_DIR / "three_model_test_predictions.csv"
XGB_IMPORTANCE_FILE = OUTPUTS_DIR / "xgboost_feature_importance.csv"
FINAL_PREDICTIONS_FILE = OUTPUTS_DIR / "final_model_predictions_all_cells.csv"
FINAL_MODEL_FILE = OUTPUTS_DIR / "final_pm25_model.joblib"
FINAL_MODEL_META_FILE = OUTPUTS_DIR / "final_model_metadata.txt"

RANDOM_STATE = 42
SPATIAL_BLOCK_SIZE = 5000
TEST_BLOCK_FRACTION = 0.20
TARGET = "pm25_mean"
FEATURES = [
    "major_mean_aadf",
    "major_hgv_mean_aadf",
    "major_road_length_m",
    "minor_vkm_total",
    "minor_hgv_vkm",
    "green_cover_pct",
]


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def make_xgb():
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_spatial_split(df):
    work = df.copy()
    work["block_x"] = np.floor(work["centroid_x"] / SPATIAL_BLOCK_SIZE).astype(int)
    work["block_y"] = np.floor(work["centroid_y"] / SPATIAL_BLOCK_SIZE).astype(int)
    work["spatial_block"] = work["block_x"].astype(str) + "_" + work["block_y"].astype(str)
    blocks = work["spatial_block"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(blocks)
    n_test = max(1, int(np.ceil(len(blocks) * TEST_BLOCK_FRACTION)))
    test_blocks = set(blocks[:n_test])
    helper_columns = ["block_x", "block_y", "spatial_block"]
    train = work[~work["spatial_block"].isin(test_blocks)].drop(columns=helper_columns)
    test = work[work["spatial_block"].isin(test_blocks)].drop(columns=helper_columns)
    return train, test


def main():
    print("=" * 90)
    print("THREE-MODEL LONDON PM2.5 COMPARISON")
    print("=" * 90)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    required = ["grid_id", "centroid_x", "centroid_y", TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required].copy()
    for column in required:
        if column != "grid_id":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required[1:]).reset_index(drop=True)
    print(f"\nTotal modelling cells: {len(df):,}")

    train, test = create_spatial_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Training cells: {len(train):,}")
    print(f"Test cells:     {len(test):,}")

    print("\nTraining Linear Regression...")
    linear = LinearRegression().fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    linear_metrics = metrics(y_test, linear_pred)

    print("Training Random Forest...")
    random_forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    rf_pred = random_forest.predict(X_test)
    rf_metrics = metrics(y_test, rf_pred)

    print("Training XGBoost...")
    xgb = make_xgb().fit(X_train, y_train, verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = metrics(y_test, xgb_pred)

    comparison = pd.DataFrame([
        {"model": "Linear Regression", **linear_metrics},
        {"model": "Random Forest", **rf_metrics},
        {"model": "XGBoost", **xgb_metrics},
    ]).sort_values("RMSE", ascending=True).reset_index(drop=True)
    comparison.to_csv(COMPARISON_FILE, index=False)

    test_predictions = test[["grid_id", "centroid_x", "centroid_y", TARGET]].rename(
        columns={TARGET: "observed_pm25"}
    )
    test_predictions["linear_prediction"] = linear_pred
    test_predictions["random_forest_prediction"] = rf_pred
    test_predictions["xgboost_prediction"] = xgb_pred
    test_predictions.to_csv(TEST_PREDICTIONS_FILE, index=False)

    xgb_importance = pd.DataFrame({"feature": FEATURES, "importance": xgb.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    xgb_importance.to_csv(XGB_IMPORTANCE_FILE, index=False)

    best_model_name = comparison.iloc[0]["model"]
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected final model: {best_model_name}")

    X_all, y_all = df[FEATURES], df[TARGET]
    if best_model_name == "Linear Regression":
        final_model = LinearRegression().fit(X_all, y_all)
    elif best_model_name == "Random Forest":
        final_model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(X_all, y_all)
    else:
        final_model = make_xgb().fit(X_all, y_all, verbose=False)

    all_predictions = df[["grid_id", "centroid_x", "centroid_y", TARGET]].rename(
        columns={TARGET: "observed_pm25"}
    )
    all_predictions["final_model_prediction"] = final_model.predict(X_all)
    all_predictions["final_model_residual"] = (
        all_predictions["observed_pm25"] - all_predictions["final_model_prediction"]
    )
    all_predictions.to_csv(FINAL_PREDICTIONS_FILE, index=False)
    joblib.dump(final_model, FINAL_MODEL_FILE)

    with open(FINAL_MODEL_META_FILE, "w", encoding="utf-8") as file:
        file.write(f"Selected model: {best_model_name}\n")
        file.write("Selection metric: RMSE on spatial holdout\n")
        file.write(f"Training cells for final fit: {len(df):,}\n")
        file.write("Features:\n")
        for feature in FEATURES:
            file.write(f"- {feature}\n")

    print("\n" + "=" * 90)
    print("FINAL MODEL / FULL-LONDON PREDICTIONS")
    print("=" * 90)
    print(f"Model: {best_model_name}")
    print(f"Cells predicted: {len(all_predictions):,}")
    print(f"Predicted mean: {all_predictions['final_model_prediction'].mean():.4f}")
    print(f"Predicted minimum: {all_predictions['final_model_prediction'].min():.4f}")
    print(f"Predicted maximum: {all_predictions['final_model_prediction'].max():.4f}")
    print("\nOutputs:")
    print(f"Comparison: {COMPARISON_FILE}")
    print(f"Test predictions: {TEST_PREDICTIONS_FILE}")
    print(f"XGBoost importance: {XGB_IMPORTANCE_FILE}")
    print(f"Full predictions: {FINAL_PREDICTIONS_FILE}")
    print(f"Final model: {FINAL_MODEL_FILE}")
    print(f"Metadata: {FINAL_MODEL_META_FILE}")
    print("\nThree-model comparison complete.")


if __name__ == "__main__":
    main()