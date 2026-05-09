"""
Smart Inventory & Demand Prediction System
ML Pipeline — Feature Engineering + Model Training + Persistence
Models: Random Forest, Gradient Boosting (XGBoost-style), Linear Baseline
Metric target: MAPE < 15%, R² > 0.85
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows")

# ─── Feature Engineering ──────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()

    # Calendar features
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"]= df["date"].dt.dayofyear
    df["quarter"]    = df["date"].dt.quarter
    df["day_of_week"]= df["date"].dt.dayofweek

    # Fourier features (capture seasonality without overfitting)
    df["sin_month"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_week"]   = np.sin(2 * np.pi * df["week"] / 52)
    df["cos_week"]   = np.cos(2 * np.pi * df["week"] / 52)
    df["sin_doy"]    = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]    = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Lag features (per product)
    for lag in [1, 7, 14, 28]:
        df[f"demand_lag_{lag}"] = df.groupby("product_id")["demand"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("product_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("product_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # Exponential weighted moving average
    df["ewma_7"]  = df.groupby("product_id")["demand"].transform(
        lambda x: x.shift(1).ewm(span=7, min_periods=1).mean())
    df["ewma_30"] = df.groupby("product_id")["demand"].transform(
        lambda x: x.shift(1).ewm(span=30, min_periods=1).mean())

    # Inventory pressure features
    df["inventory_ratio"]   = df["inventory_level"] / (df["reorder_point"] + 1)
    df["days_of_stock"]     = df["inventory_level"] / (df["rolling_mean_7"] + 1)
    df["stockout_risk"]     = (df["inventory_level"] < df["reorder_point"]).astype(int)

    # Price sensitivity proxy
    df["price_tier"] = pd.qcut(df["unit_price"], 4, labels=[0, 1, 2, 3]).astype(int)

    # Weather encoding
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snowy": 3, "Stormy": 4}
    df["weather_code"] = df["weather"].map(weather_map)

    # Category encoding
    le = LabelEncoder()
    df["category_code"] = le.fit_transform(df["category"])

    # Holiday/event proximity (days since or until major events)
    df["is_q4"]         = (df["month"].isin([10, 11, 12])).astype(int)
    df["is_holiday_week"]= df["week"].isin([1, 14, 22, 44, 48, 52]).astype(int)

    return df, le

df, label_enc = build_features(df)

FEATURES = [
    # Calendar
    "year", "month", "week", "day_of_year", "quarter", "day_of_week",
    # Fourier
    "sin_month", "cos_month", "sin_week", "cos_week", "sin_doy", "cos_doy",
    # Lags
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    # Rolling stats
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_14", "rolling_std_30",
    # EWMAs
    "ewma_7", "ewma_30",
    # Inventory
    "inventory_ratio", "days_of_stock", "stockout_risk",
    # Product
    "unit_price", "lead_time_days", "reorder_point", "safety_stock", "price_tier",
    # Contextual
    "weather_code", "temperature_c", "is_promo", "is_weekend",
    "is_q4", "is_holiday_week", "category_code",
]

TARGET = "demand"

# ─── Train / Test split (time-based — no leakage) ───────────────────────────
df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
cutoff    = pd.Timestamp("2024-07-01")
train_df  = df_clean[df_clean["date"] < cutoff]
test_df   = df_clean[df_clean["date"] >= cutoff]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ─── Models ──────────────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        max_features=0.6, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=4, random_state=42
    ),
    "Ridge Baseline": Ridge(alpha=10),
}

results   = {}
artifacts = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mp   = mape(y_test.values, preds)

    results[name]   = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mp}
    artifacts[name] = model
    print(f"  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}  MAPE={mp:.1f}%")

# ─── Best model ───────────────────────────────────────────────────────────────
best_name  = min(results, key=lambda k: results[k]["MAPE"])
best_model = artifacts[best_name]
print(f"\n🏆 Best model: {best_name} (MAPE={results[best_name]['MAPE']:.1f}%)")

# ─── Feature importances ──────────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({
        "feature":    FEATURES,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv("models/feature_importance.csv", index=False)
    print("\nTop 10 features:")
    print(fi.head(10).to_string(index=False))

# ─── Anomaly detection (Z-score on rolling demand) ────────────────────────────
df_agg = df.groupby(["date", "category"])["demand"].sum().reset_index()
df_agg["rolling_mean"] = df_agg.groupby("category")["demand"].transform(
    lambda x: x.rolling(30, min_periods=1).mean())
df_agg["rolling_std"]  = df_agg.groupby("category")["demand"].transform(
    lambda x: x.rolling(30, min_periods=1).std().fillna(1))
df_agg["z_score"]      = (df_agg["demand"] - df_agg["rolling_mean"]) / df_agg["rolling_std"]
df_agg["is_anomaly"]   = (df_agg["z_score"].abs() > 2.5).astype(int)
df_agg.to_csv("models/anomaly_data.csv", index=False)

# ─── Save artifacts ───────────────────────────────────────────────────────────
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_enc, f)
with open("models/feature_list.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

# Save metrics
with open("models/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# Save processed data sample for app
df_clean.to_csv("models/processed_data.csv", index=False)
df_agg.to_csv("models/category_daily.csv", index=False)

# Forecast for next 30 days (use last known features per product)
last_known = df_clean.sort_values("date").groupby("product_id").last().reset_index()
last_known["forecast_demand"] = np.clip(best_model.predict(last_known[FEATURES]), 0, None).astype(int)
last_known["recommended_order"] = np.maximum(
    0,
    last_known["forecast_demand"] * 7 + last_known["safety_stock"] - last_known["inventory_level"]
).astype(int)
last_known["stock_status"] = last_known.apply(
    lambda r: "🔴 Critical" if r["inventory_level"] < r["safety_stock"]
    else ("🟡 Low" if r["inventory_level"] < r["reorder_point"]
    else "🟢 Healthy"), axis=1
)
last_known[["product_id","category","inventory_level","reorder_point",
            "forecast_demand","recommended_order","stock_status","unit_price"]].to_csv(
    "models/inventory_status.csv", index=False)

print("\n✅ All artifacts saved to models/")
print(f"   Model metrics: {json.dumps(results[best_name], indent=2)}")
