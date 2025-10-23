# ⚙️ Step 2: Data Preprocessing & Feature Engineering Pipeline

## 🎯 Objective
To transform raw synthetic transaction data into **high-quality, machine-learning-ready features**.  
This step ensures that the model can learn **patterns of normal customer behavior** and **identify anomalies** effectively.

---

## 🧹 Data Preprocessing Steps

### 1. **Data Cleaning**
Before modeling, we’ll ensure the dataset is consistent and free of invalid values.

| Task | Description |
|------|--------------|
| Handle Missing Values | Fill or drop null entries (e.g., missing `device_id` or `geo_location`) |
| Format Timestamps | Convert `timestamp` to pandas datetime |
| Encode Time Features | Extract `hour`, `day_of_week`, `month`, etc. |
| Standardize Categories | Normalize text labels for merchant, channel, and device |
| Remove Duplicates | Drop repeated transactions by `transaction_id` |
| Handle Outliers | Cap extreme transaction amounts using IQR or log scaling |

---

### 2. **Data Transformation**
Convert raw values into model-usable formats:

| Transformation | Description |
|----------------|--------------|
| **Label Encoding** | Convert categorical features (e.g., transaction_type) into integers |
| **One-Hot Encoding** | For low-cardinality features like `channel`, `day_of_week` |
| **Normalization / Scaling** | Standardize numeric values (amounts, frequencies) |
| **Feature Filtering** | Drop redundant or uninformative columns |

---

## 🧠 Feature Engineering

Feature engineering is the **core of fraud detection** — this is where the system learns to understand customer behavior.

We’ll create **three key categories** of features:

---

### 🏦 1. Transactional Features
Describe the **current transaction** itself.

| Feature | Description |
|----------|--------------|
| `transaction_amount` | Amount of current transaction |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Monday–Sunday |
| `transaction_type_encoded` | Encoded type (POS, ATM, Online) |
| `merchant_category_encoded` | Encoded merchant type |
| `is_weekend` | 1 if transaction occurs on Saturday/Sunday |

---

### 👤 2. Behavioral Features
Describe the **customer’s typical transaction patterns** over a historical window (e.g., last 7 or 30 days).

| Feature | Description |
|----------|--------------|
| `avg_txn_amount_7d` | Average transaction amount in last 7 days |
| `std_txn_amount_7d` | Spending variation over last 7 days |
| `txn_count_24h` | Number of transactions in past 24 hours |
| `txn_count_7d` | Number of transactions in past 7 days |
| `avg_hour_activity` | Mean hour of customer’s recent transactions |
| `unique_devices_used` | Number of distinct devices used recently |
| `unique_locations_used` | Number of distinct locations used recently |
| `last_txn_hours_diff` | Hours since last transaction |

These features are **computed per customer ID** and **joined** with the current transaction record.

---

### 🌐 3. Risk & Anomaly Features
Capture sudden or abnormal changes that might indicate fraud.

| Feature | Description |
|----------|--------------|
| `amount_deviation` | Difference between current and average spend |
| `new_device_flag` | 1 if current device is unseen for this customer |
| `new_location_flag` | 1 if transaction location differs significantly |
| `unusual_time_flag` | 1 if transaction happens outside usual hours |
| `rapid_fire_flag` | 1 if multiple transactions occur in a short window |
| `is_high_risk_country` | 1 if transaction originates from a flagged region |

---

## 🧩 Feature Pipeline Design

Below is a high-level view of how features will be generated and merged.

             ┌──────────────────────────────┐
             │  Raw Transactions (CSV)      │
             └──────────┬───────────────────┘
                        │
                        ▼
             ┌──────────────────────────────┐
             │ Data Cleaning & Formatting    │
             └──────────┬───────────────────┘
                        │
                        ▼
             ┌──────────────────────────────┐
             │ Feature Computation per User  │
             │ (rolling time window stats)   │
             └──────────┬───────────────────┘
                        │
                        ▼
             ┌──────────────────────────────┐
             │ Merge Engineered Features     │
             │ with Raw Transactions         │
             └──────────┬───────────────────┘
                        │
                        ▼
             ┌──────────────────────────────┐
             │ Train / Test Split            │
             └──────────────────────────────┘

---

## 🧮 Example Feature Engineering Script (Simplified)

```python
import pandas as pd
import numpy as np

# Load synthetic data
df = pd.read_csv("synthetic_transactions.csv", parse_dates=["timestamp"])
df.sort_values(by=["customer_id", "timestamp"], inplace=True)

# Extract temporal features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

# Compute behavioral features
df["avg_txn_amount_7d"] = (
    df.groupby("customer_id")["transaction_amount"]
      .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
df["txn_count_24h"] = (
    df.groupby("customer_id")["timestamp"]
      .transform(lambda x: x.diff().dt.total_seconds().lt(86400).cumsum())
)

# Compute anomaly flags
df["amount_deviation"] = (
    df["transaction_amount"] / (df["avg_txn_amount_7d"] + 1e-6)
)
df["new_device_flag"] = (
    df.groupby("customer_id")["device_id"]
      .transform(lambda x: x.ne(x.shift()).astype(int))
)

# Final dataset ready for modeling
df.to_csv("processed_features.csv", index=False)


🧾 Output

File: processed_features.csv

Rows: same as input (~100,000)

Columns: 25–40 engineered features

Purpose: Model training, evaluation, and real-time scoring.