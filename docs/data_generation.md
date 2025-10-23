# 🧩 Step 1: Data Schema & Synthetic Data Generation Plan

## 🎯 Objective
To build and test the fraud detection model, we first need a **realistic transaction dataset** that simulates how bank customers interact with different payment channels.  
Because real banking data is confidential, we’ll **generate synthetic data** that closely mirrors real-world behavior — including both **legitimate** and **fraudulent** transactions.

---

## 🧱 Data Schema

Each record represents a single **bank transaction** made by a customer.  
The dataset will include **behavioral, transactional, and device-level information**.

| Feature Name | Description | Example | Data Type |
|---------------|-------------|----------|------------|
| `transaction_id` | Unique identifier for each transaction | TXN_001 | string |
| `customer_id` | Unique ID for customer | C1021 | string |
| `timestamp` | Date and time of transaction | 2025-09-20 14:32:10 | datetime |
| `transaction_amount` | Value of transaction (₦) | 25,000 | float |
| `transaction_type` | Type of transaction | POS, ATM, Online | string |
| `merchant_category` | Merchant classification | Groceries, Electronics | string |
| `device_id` | Unique device or card fingerprint | DVC_09A | string |
| `ip_address` | IP used for transaction | 102.23.55.12 | string |
| `geo_location` | Latitude & longitude of transaction | (6.5244, 3.3792) | tuple |
| `channel` | Source of transaction | MobileApp, POS, Web | string |
| `time_of_day` | Encoded as Morning/Afternoon/Evening/Night | Night | string |
| `day_of_week` | Day transaction occurred | Friday | string |
| `num_prev_txn_24h` | No. of previous transactions in last 24 hours | 5 | integer |
| `avg_txn_amount_7d` | Average amount over last 7 days | 15,000 | float |
| `new_device_flag` | 1 if device is unseen for customer | 0/1 | integer |
| `new_location_flag` | 1 if location differs from recent pattern | 0/1 | integer |
| `is_high_risk_country` | 1 if from risky region | 0/1 | integer |
| `is_fraud` | Target variable — 1 = Fraudulent, 0 = Legitimate | 1 | integer |

---

## ⚙️ Synthetic Data Generation Logic

### 1. **Customer Profiles**
- Simulate ~5,000 unique customers.  
- Each customer has typical **spending ranges** and **transaction habits**.
  - E.g., Customer A usually spends ₦5k–₦30k, mostly daytime, same device.
  - Customer B spends ₦200k+ weekly, often international.

### 2. **Normal (Legitimate) Transactions**
- 98–99% of all records.
- Follow customer’s typical behavior:
  - Common device, known location.
  - Average spend within range.
  - Transaction frequency consistent with historical data.

### 3. **Fraudulent Transactions**
- 1–2% of records are labeled `is_fraud = 1`.
- Simulate **abnormal behavior** such as:
  - Unusual amount (too high or too low).
  - New or unseen device.
  - Location far from last known area.
  - Late-night transaction when customer is usually inactive.
  - High transaction frequency within a short window.

### 4. **Time-Series Generation**
- Generate transactions chronologically (by `timestamp`).
- Include seasonality and daily patterns:
  - More POS on weekends.
  - More online/mobile during work hours.

---

## 🧮 Tools & Libraries
Use Python to generate data:
- **NumPy / Pandas** → generate structured data
- **Faker** → generate realistic IDs, timestamps, IPs, and device strings
- **Datetime** → create time-based patterns
- **Random module** → inject controlled noise and anomalies

---

## 🧠 Example Data Generation Pseudocode

```python
from faker import Faker
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

faker = Faker()

customers = [f"CUST_{i}" for i in range(1, 5001)]
devices = [f"DVC_{i}" for i in range(1, 200)]
merchant_categories = ["Groceries", "Electronics", "Bills", "Clothing", "Travel", "Fuel", "Food"]
channels = ["POS", "Online", "ATM", "MobileApp"]

transactions = []

for i in range(100000):
    cust = random.choice(customers)
    base_amount = np.random.normal(20000, 10000)
    amount = max(1000, np.random.normal(base_amount, 8000))
    device = random.choice(devices)
    category = random.choice(merchant_categories)
    channel = random.choice(channels)
    timestamp = faker.date_time_between(start_date="-90d", end_date="now")

    # Inject fraud 1-2%
    is_fraud = np.random.choice([0, 1], p=[0.985, 0.015])
    if is_fraud:
        amount *= np.random.uniform(2, 8)
        device = f"NEW_{random.randint(100, 999)}"
        channel = random.choice(["Online", "Web"])
    
    transactions.append({
        "transaction_id": f"TXN_{i+1}",
        "customer_id": cust,
        "timestamp": timestamp,
        "transaction_amount": round(amount, 2),
        "transaction_type": channel,
        "merchant_category": category,
        "device_id": device,
        "ip_address": faker.ipv4(),
        "geo_location": (faker.latitude(), faker.longitude()),
        "is_fraud": is_fraud
    })

df = pd.DataFrame(transactions)
df.to_csv("synthetic_transactions.csv", index=False)
