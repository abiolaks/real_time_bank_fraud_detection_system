import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# =====================================
# 1 Log Transformation
# =====================================
def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log(1 + x) to transaction amount to reduce skewness."""
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    return df

# =====================================
# 2 Feature Engineering
# =====================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral and temporal features for fraud detection."""
    df = df.copy()
    
    # Ensure datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Per-customer aggregate features
    user_agg = df.groupby("customer_id").agg(
        avg_amount_per_user=("amount", "mean"),
        std_amount_per_user=("amount", "std"),
        unique_devices_per_user=("device_id", "nunique"),
        unique_locations_per_user=("location", "nunique")
    ).reset_index()

    # Merge back
    df = df.merge(user_agg, on="customer_id", how="left")

    # Derived behavioral features
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_per_user"] + 1e-6)
    df["is_high_deviation"] = (
        abs(df["amount"] - df["avg_amount_per_user"]) > 3 * df["std_amount_per_user"]
    ).astype(int)

    # Fill missing std values (new users)
    df["std_amount_per_user"].fillna(0, inplace=True)

    return df

# =====================================
# 3 Column Lists
# =====================================
numeric_cols = [
    "amount_log", "hour_of_day", "day_of_week",
    "avg_amount_per_user", "std_amount_per_user",
    "amount_to_avg_ratio"
]

categorical_cols = ["merchant_category", "merchant_id", "device_id", "location"]

# =====================================
# 4 Transformers
# =====================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# =====================================
# 5 Column Transformer
# =====================================
column_transformer = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

# =====================================
# 6 Full Preprocessing Pipeline
# =====================================
preprocessing_pipeline = Pipeline(steps=[
    ("log_transform", FunctionTransformer(log_transform_amount)),
    ("feature_engineering", FunctionTransformer(engineer_features)),
    ("transform", column_transformer)
])
