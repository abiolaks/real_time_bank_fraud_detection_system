import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
NUM_CUSTOMERS = 5000
TRANSACTIONS_PER_CUSTOMER = (20, 50)
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

MERCHANT_CATEGORIES = [
    "Groceries",
    "Electronics",
    "Fashion",
    "Restaurants",
    "Travel",
    "Utilities",
    "Health",
    "Entertainment",
    "Fuel",
    "Online Shopping",
]
LOCATIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "Enugu", "Kaduna"]
DEVICES = [
    "iPhone_12",
    "Samsung_S21",
    "Infinix_Hot",
    "Tecno_Spark",
    "Windows_PC",
    "MacBook_Pro",
]

# -------------------------------
# FIXED RANDOM SEED FOR REPRODUCIBILITY
# -------------------------------
random.seed(42)
np.random.seed(42)


# -------------------------------
# GENERATE CUSTOMER PROFILES
# -------------------------------
def generate_customer_profiles(num_customers):
    customers = []
    for cust_id in range(1, num_customers + 1):
        income = np.random.normal(250000, 100000)
        spending_ratio = np.clip(np.random.normal(0.3, 0.1), 0.1, 0.6)
        location = random.choice(LOCATIONS)
        preferred_categories = random.sample(
            MERCHANT_CATEGORIES, k=random.randint(2, 5)
        )
        customers.append(
            {
                "customer_id": cust_id,
                "income": max(50000, income),
                "spending_ratio": spending_ratio,
                "location": location,
                "preferred_categories": preferred_categories,
            }
        )
    return pd.DataFrame(customers)


# -------------------------------
# GENERATE TRANSACTIONS
# -------------------------------
def generate_transactions(customers):
    transactions = []
    txn_id = 1

    for _, cust in customers.iterrows():
        num_txn = random.randint(*TRANSACTIONS_PER_CUSTOMER)
        for _ in range(num_txn):
            txn_time = START_DATE + timedelta(
                days=random.randint(0, (END_DATE - START_DATE).days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            base_amount = np.random.normal(
                cust["income"] * cust["spending_ratio"], 20000
            )
            amount = abs(base_amount) * random.uniform(0.3, 1.3)
            merchant_category = random.choice(cust["preferred_categories"])
            merchant_id = f"M{random.randint(1000, 9999)}"
            location = cust["location"]
            device = random.choice(DEVICES)

            # Fraud logic
            fraud_prob = 0.02
            if amount > cust["income"] * 0.8:
                fraud_prob += 0.2
            if random.random() < 0.05:
                location = random.choice(
                    [loc for loc in LOCATIONS if loc != cust["location"]]
                )
                fraud_prob += 0.15
            if random.random() < 0.05:
                device = random.choice([d for d in DEVICES if d != device])
                fraud_prob += 0.1
            if txn_time.hour < 5 or txn_time.hour > 23:
                fraud_prob += 0.1
            if random.random() < 0.03:
                merchant_category = random.choice(
                    [
                        cat
                        for cat in MERCHANT_CATEGORIES
                        if cat not in cust["preferred_categories"]
                    ]
                )
                fraud_prob += 0.1

            is_fraud = np.random.rand() < fraud_prob

            transactions.append(
                {
                    "transaction_id": txn_id,
                    "customer_id": cust["customer_id"],
                    "timestamp": txn_time,
                    "amount": round(amount, 2),
                    "merchant_category": merchant_category,
                    "merchant_id": merchant_id,
                    "device_id": device,
                    "location": location,
                    "is_fraud": int(is_fraud),
                }
            )
            txn_id += 1

    return pd.DataFrame(transactions)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
if __name__ == "__main__":
    print("Generating final synthetic bank transaction dataset...")

    customers_df = generate_customer_profiles(NUM_CUSTOMERS)
    transactions_df = generate_transactions(customers_df)
    transactions_df = transactions_df.sort_values("timestamp").reset_index(drop=True)

    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)
    transactions_df.to_csv(
        os.path.join(output_dir, "bank_transactions.csv"), index=False
    )
    print(
        f"Dataset created: {len(transactions_df)} transactions for {len(customers_df)} customers."
    )
    print("Saved as synthetic_bank_transactions.csv")
