# Fraud Detection Model – Planning Phase

## Project Objective
Build a **machine learning–powered fraud detection POC** for a bank that:
- Detects **fraudulent or suspicious transactions in real time**.
- Learns **each customer’s normal behavior**.
- Provides a **fraud risk score** and explanation for decision-making.
- Can be **extended into production** (with APIs, dashboards, etc.).

---

##  Scope of the POC
The POC will demonstrate:
1. **Data pipeline** → ingest and preprocess historical transaction data.  
2. **Feature engineering** → compute behavioral and transaction-based features.  
3. **Model training and evaluation** → train supervised/unsupervised ML models.  
4. **Inference pipeline** → simulate real-time detection on new transactions.  
5. **Logging + dashboard** → store predictions and visualize fraud alerts.

 **Goal:** Show a working system prototype — not a full production deployment — but designed in a modular, scalable, and realistic way.

---

##  Data Understanding

### a. Data Sources
We’ll use (or generate) a realistic transaction dataset containing:
- `customer_id`
- `transaction_id`
- `timestamp`
- `transaction_amount`
- `transaction_type` (POS, ATM, Online)
- `merchant_category`
- `device_id`
- `ip_address` / `geo_location`
- `is_fraud` (label)

### b. Data Volume
For POC: 10,000–100,000 transactions (synthetic or anonymized real data).

---

##  System Architecture (High-Level)
            ┌────────────────────────┐
            │   Transaction Source    │
            │ (historical + new txns) │
            └──────────┬──────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │ Data Preprocessing &    │
            │ Feature Engineering     │
            └──────────┬──────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  ML Fraud Detection     │
            │   Model (XGBoost etc.)  │
            └──────────┬──────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  Inference Pipeline     │
            │ (API or Batch Scoring)  │
            └──────────┬──────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  Logging & Monitoring   │
            │ (Database / Dashboard)  │
            └────────────────────────┘

---

##  Modeling Strategy

We’ll consider **two modeling paths**:

### Option 1: Supervised ML
Use labeled historical transactions (`is_fraud = 1/0`).

Model candidates:
- XGBoost / LightGBM  
- Random Forest  
- Logistic Regression (baseline)

### Option 2: Unsupervised ML (Anomaly Detection)
If labels are unavailable or incomplete:
- Isolation Forest  
- One-Class SVM  
- Autoencoders (Deep Learning)

---

##  Feature Engineering Plan

We’ll design features in **three categories**:

###  Transactional Features
- Transaction amount  
- Transaction type (encoded)  
- Time of day  
- Merchant category  

###  Behavioral Features
- Average transaction amount (per user)  
- Standard deviation of spend  
- Frequency per day/week  
- Usual transaction time window  
- Common location/device  

###  Risk Features
- New device or IP?  
- Location distance from last transaction  
- Amount deviation from usual pattern  
- Multiple transactions in short period  

---

##  Tech Stack

| Layer | Tools |
|-------|--------|
| **Data** | Pandas, NumPy, SQL / synthetic generator |
| **Modeling** | Scikit-learn, XGBoost, LightGBM |
| **Logging & Storage** | SQLite / PostgreSQL / CSV |
| **Visualization** | Streamlit dashboard |
| **Deployment (optional)** | FastAPI or Flask for inference API |
| **Environment** | Python 3.10+, modular OOP code structure |

---

##  Evaluation Metrics
Depending on data balance:
- **Precision, Recall, F1-score** (for fraud detection)
- **ROC-AUC** (discrimination power)
- **Confusion Matrix** (false positives vs false negatives)

>  Business focus: prioritize **Recall (catch all fraud)** while controlling **False Positives** (avoid annoying legit users).

---

##  POC Workflow

| Phase | Description | Deliverables |
|--------|--------------|---------------|
| **1. Data Preparation** | Load, clean, and preprocess dataset | Clean dataset |
| **2. Feature Engineering** | Create behavioral and transactional features | Feature dataset |
| **3. Model Training** | Train baseline + tuned ML models | Trained model |
| **4. Evaluation** | Measure metrics on test set | Metrics report |
| **5. Inference Pipeline** | Real-time fraud detection simulation | Inference script |
| **6. Dashboard** | Streamlit dashboard to visualize results | Fraud detection UI |

---

