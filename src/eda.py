"""
eda_exploration.py

Exploratory Data Analysis (EDA) for synthetic_bank_transactions.csv
Step-by-step: data overview, cleaning checks, univariate & bivariate analysis,
time-series checks, customer-level summaries, and artifact export.

Dependencies:
    pandas, numpy, matplotlib, seaborn, plotly (optional), scipy
Install with:
    pip install pandas numpy matplotlib seaborn plotly scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional for interactive graphs:
# import plotly.express as px

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

DATA_PATH = "synthetic_bank_transactions.csv"
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 0. Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print("Loaded:", df.shape)
print(df.head())

# -----------------------------
# 1. Quick overview / sanity checks
# -----------------------------
print("\n=== Basic info ===")
print(df.dtypes)
print("\n=== Missing values ===")
print(df.isnull().sum())

# Unique counts
print("\nUnique counts:")
print("Customers:", df["customer_id"].nunique())
print(
    "Merchants:",
    df["merchant_id"].nunique()
    if "merchant_id" in df.columns
    else "merchant_id not present",
)
print("Devices:", df["device_id"].nunique())
print("Locations:", df["location"].nunique())

# -----------------------------
# 2. Target distribution (fraud ratio)
# -----------------------------
print("\n=== Fraud target distribution ===")
fraud_counts = df["is_fraud"].value_counts()
print(fraud_counts)
fraud_ratio = fraud_counts.get(1, 0) / len(df)
print(f"Fraud ratio: {fraud_ratio:.4f}")

# Plot fraud ratio
plt.figure()
ax = sns.countplot(x="is_fraud", data=df)
ax.set_title("Fraud (1) vs Legit (0) counts")
plt.savefig(os.path.join(OUT_DIR, "fraud_counts.png"))
plt.close()

# -----------------------------
# 3. Transaction amount distribution
# -----------------------------
print("\n=== Amount distribution summary ===")
print(df["amount"].describe())

# Histogram + log-histogram
plt.figure()
sns.histplot(df["amount"], bins=100, kde=False)
plt.title("Transaction amount distribution")
plt.xlim(0, df["amount"].quantile(0.99))  # zoom to 99th percentile
plt.savefig(os.path.join(OUT_DIR, "amount_hist.png"))
plt.close()

plt.figure()
sns.histplot(np.log1p(df["amount"]), bins=100, kde=False)
plt.title("Log(1+amount) distribution")
plt.savefig(os.path.join(OUT_DIR, "amount_log_hist.png"))
plt.close()

# Amount vs is_fraud (box or violin)
plt.figure()
sns.boxplot(
    x="is_fraud", y="amount", data=df[df["amount"] <= df["amount"].quantile(0.99)]
)
plt.yscale("log")
plt.title("Amount by Fraud label (zoomed, log scale)")
plt.savefig(os.path.join(OUT_DIR, "amount_by_fraud_box.png"))
plt.close()

# -----------------------------
# 4. Time patterns
# -----------------------------
# add time features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["date"] = df["timestamp"].dt.date

# hourly volume & fraud rate by hour
hourly = df.groupby("hour").agg(
    total_txn=("transaction_id", "count"), fraud_txn=("is_fraud", "sum")
)
hourly["fraud_rate"] = hourly["fraud_txn"] / hourly["total_txn"]

hourly.to_csv(os.path.join(OUT_DIR, "hourly_stats.csv"))
plt.figure()
ax = hourly["total_txn"].plot(kind="bar")
ax.set_title("Transactions per hour of day")
plt.savefig(os.path.join(OUT_DIR, "txns_per_hour.png"))
plt.close()

plt.figure()
ax = hourly["fraud_rate"].plot(kind="bar", color="orange")
ax.set_title("Fraud rate by hour of day")
plt.savefig(os.path.join(OUT_DIR, "fraud_rate_by_hour.png"))
plt.close()

# weekday pattern
weekday = df.groupby("day_of_week").agg(
    total=("transaction_id", "count"), fraud=("is_fraud", "sum")
)
weekday["fraud_rate"] = weekday["fraud"] / weekday["total"]
weekday = weekday.reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
weekday.to_csv(os.path.join(OUT_DIR, "weekday_stats.csv"))

plt.figure()
weekday["total"].plot(kind="bar")
plt.title("Transactions per weekday")
plt.savefig(os.path.join(OUT_DIR, "txns_per_weekday.png"))
plt.close()

plt.figure()
weekday["fraud_rate"].plot(kind="bar", color="red")
plt.title("Fraud rate per weekday")
plt.savefig(os.path.join(OUT_DIR, "fraud_rate_per_weekday.png"))
plt.close()

# -----------------------------
# 5. Merchant & device analysis
# -----------------------------
# top merchants by volume and fraud rate
top_merchants = (
    df.groupby("merchant_id")
    .agg(total=("transaction_id", "count"), fraud=("is_fraud", "sum"))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"])
    .sort_values("total", ascending=False)
)
top_merchants.head(20).to_csv(os.path.join(OUT_DIR, "top_merchants.csv"))

# plot top 10 merchants volume
plt.figure()
top_merchants.head(10)["total"].plot(kind="bar")
plt.title("Top 10 merchants by transaction volume")
plt.savefig(os.path.join(OUT_DIR, "top10_merchants_volume.png"))
plt.close()

# devices
top_devices = (
    df.groupby("device_id")
    .agg(total=("transaction_id", "count"), fraud=("is_fraud", "sum"))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"])
    .sort_values("total", ascending=False)
)
top_devices.head(20).to_csv(os.path.join(OUT_DIR, "top_devices.csv"))

# -----------------------------
# 6. Customer-level summaries
# -----------------------------
# compute per-customer aggregates: total txns, total spend, fraud count, unique devices, active days
cust_agg = (
    df.groupby("customer_id")
    .agg(
        total_txn=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        avg_txn_amt=("amount", "mean"),
        fraud_count=("is_fraud", "sum"),
        unique_devices=("device_id", "nunique"),
        active_days=("date", "nunique"),
    )
    .assign(fraud_rate=lambda x: x["fraud_count"] / x["total_txn"])
)
cust_agg.to_csv(os.path.join(OUT_DIR, "customer_aggregates.csv"))
print("\nCustomer aggregates sample:")
print(cust_agg.describe().transpose())

# inspect customers with fraud_count>0
cust_with_fraud = cust_agg[cust_agg["fraud_count"] > 0].sort_values(
    "fraud_count", ascending=False
)
cust_with_fraud.head(20).to_csv(os.path.join(OUT_DIR, "customers_with_fraud.csv"))

# -----------------------------
# 7. Bivariate relationships & correlation
# -----------------------------
# correlation among numeric features, including amount, total_spend etc.
num_cols = ["amount", "hour", "total_txn"]  # total_txn placeholder if present
# create quick numeric df for correlation
corr_df = df[["amount", "hour", "is_fraud"]].copy()
corr_matrix = corr_df.corr()
corr_matrix.to_csv(os.path.join(OUT_DIR, "corr_matrix.csv"))
print("\nCorrelation matrix:")
print(corr_matrix)

# scatter amount vs hour colored by fraud
plt.figure()
sns.scatterplot(
    data=df.sample(min(2000, len(df))), x="hour", y="amount", hue="is_fraud", alpha=0.6
)
plt.ylim(0, df["amount"].quantile(0.95))
plt.title("Amount vs Hour (sampled)")
plt.savefig(os.path.join(OUT_DIR, "amount_vs_hour_scatter.png"))
plt.close()

# -----------------------------
# 8. Simple feature checks for planned features
# -----------------------------
# Check: new-device signal (how many transactions per device per customer)
device_per_cust = (
    df.groupby(["customer_id", "device_id"]).size().reset_index(name="count")
)
# device seen once per customer (candidate new device behavior)
single_device_usage = (
    device_per_cust[device_per_cust["count"] == 1]
    .groupby("customer_id")
    .size()
    .reset_index(name="single_device_count")
)
single_device_usage.describe().to_csv(
    os.path.join(OUT_DIR, "single_device_usage_desc.csv")
)

# Check amount deviation (we'll compute avg per customer and ratio)
cust_mean = df.groupby("customer_id")["amount"].mean().rename("cust_avg")
df = df.merge(cust_mean, on="customer_id", how="left")
df["amount_to_avg_ratio"] = df["amount"] / (df["cust_avg"] + 1e-9)
df[["amount", "cust_avg", "amount_to_avg_ratio"]].head(10).to_csv(
    os.path.join(OUT_DIR, "amount_ratio_sample.csv")
)

# Distribution of amount_to_avg_ratio for fraud vs non-fraud
plt.figure()
sns.kdeplot(
    data=df[df["is_fraud"] == 0],
    x="amount_to_avg_ratio",
    label="legit",
    bw_adjust=1,
    clip=(0, 50),
)
sns.kdeplot(
    data=df[df["is_fraud"] == 1],
    x="amount_to_avg_ratio",
    label="fraud",
    bw_adjust=1,
    clip=(0, 50),
)
plt.xlim(0, 50)
plt.legend()
plt.title("Amount / Personal Average: fraud vs legit")
plt.savefig(os.path.join(OUT_DIR, "amount_to_avg_ratio_kde.png"))
plt.close()

# -----------------------------
# 9. Time-series overview (daily)
# -----------------------------
daily = df.groupby("date").agg(
    total_txn=("transaction_id", "count"), fraud_txn=("is_fraud", "sum")
)
daily["fraud_rate"] = daily["fraud_txn"] / daily["total_txn"]
daily.to_csv(os.path.join(OUT_DIR, "daily_stats.csv"))

plt.figure(figsize=(12, 4))
daily["total_txn"].plot()
plt.title("Daily transaction volume")
plt.savefig(os.path.join(OUT_DIR, "daily_txn_volume.png"))
plt.close()

plt.figure(figsize=(12, 4))
daily["fraud_rate"].plot(color="red")
plt.title("Daily fraud rate")
plt.savefig(os.path.join(OUT_DIR, "daily_fraud_rate.png"))
plt.close()

# -----------------------------
# 10. Save cleaned sample and EDA summary
# -----------------------------
# Save a cleaned, processed sample for later steps (feature engineering)
df.to_csv(os.path.join(OUT_DIR, "transactions_with_basic_features.csv"), index=False)

# Basic EDA summary
summary = {
    "n_rows": len(df),
    "n_customers": int(df["customer_id"].nunique()),
    "fraud_ratio": float(fraud_ratio),
    "amount_mean": float(df["amount"].mean()),
    "amount_median": float(df["amount"].median()),
    "amount_99pct": float(df["amount"].quantile(0.99)),
}
import json

with open(os.path.join(OUT_DIR, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nEDA complete. Outputs saved to:", OUT_DIR)
