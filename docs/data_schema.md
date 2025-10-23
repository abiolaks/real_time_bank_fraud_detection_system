# Final & Approved Data Schema (to Use Throughout the POC)

This schema is realistic, consistent, and perfectly aligns with our planned fraud detection pipeline (including behavioral features, inference, and model serving).

## Final Schema: synthetic_bank_transactions.csv
Column	Type	Example	Description
transaction_id	int	10234	Unique identifier for each transaction
customer_id	int	451	Customer unique ID
timestamp	datetime	2024-06-12 14:45:00	Date and time of transaction
amount	float	18500.75	Transaction amount in ₦
merchant_category	string	"Groceries"	Merchant or service category
merchant_id	string	"M1234"	Unique merchant identifier
device_id	string	"iPhone_12"	Device used for the transaction
location	string	"Lagos"	Location or branch of the transaction
is_fraud	int (0 or 1)	0	Label — 1 if fraud, else 0
## Notes

merchant_id was added — banks always track it and it’s useful for merchant risk features (e.g., some merchants get more fraud).

Everything else stays stable for all later phases.

Future features (e.g., is_new_device, avg_amount_last_7d, etc.) will be engineered, not part of this raw data.

## Why This Schema Works
Step	Why it’s needed
EDA	To explore transaction patterns, frequency, and fraud distribution.
Feature Engineering	Allows computing behavioral indicators like amount deviation, time patterns, device frequency, etc.
Model Training	Contains both event-level and contextual data required for fraud detection.
Inference Pipeline	Allows real-time scoring for new transactions with historical lookup.

New data fields
These can be generated from your existing data to provide richer context for machine learning models. 
New Column Name 	Type	Description
transaction_hour	int	The hour of the day the transaction occurred. Fraudulent transactions often happen at odd hours.
time_since_last_transaction	int	The time difference (e.g., in seconds or minutes) since the last transaction for the same customer_id. A short interval could indicate high-velocity fraud.
same_merchant_since_last	int (0 or 1)	A flag to check if the last transaction was with the same merchant. Unusual sequences of merchants are a common fraud indicator.
same_location_since_last	int (0 or 1)	A flag for consecutive transactions from the same location. This helps identify geographic shifts.
merchant_is_known	int (0 or 1)	A flag to indicate if the customer has transacted with this merchant before.
amount_is_unusual	int (0 or 1)	A flag to indicate if the current amount is an unusual size compared to the customer's historical average.
merchant_velocity	int	The count of transactions for a specific merchant_id within a recent time window (e.g., last hour). An unusually high count could be suspicious.
device_velocity	int	The count of transactions for a specific device_id within a recent time window. High velocity can indicate misuse.
risk_score	float	A score derived from external APIs, such as an IP address risk score, if you use them.
customer_history_age	int	The number of days since the customer_id was first seen in the data. New customers are often higher risk.
Additional considerations for your PoC
Handle class imbalance: Your dataset will be heavily skewed towards non-fraudulent transactions. Use techniques like oversampling the minority class (is_fraud = 1) or assigning higher class weights during model training.
Model explainability: For fraud detection, understanding why a transaction was flagged is critical for operations teams. Consider using an interpretable model like a Gradient Boosting Decision Tree (GBDT) for your PoC, which provides insights into feature importance.
Real-time requirements: Your schema supports real-time analysis by allowing you to generate velocity and affinity features on the fly for incoming transactions.
Iterate and enrich: The best approach is to start with a minimal viable product (MVP) and then gradually enrich the schema with more complex features. You can even combine your transaction data with user behavior or identity information for a more comprehensive model. 