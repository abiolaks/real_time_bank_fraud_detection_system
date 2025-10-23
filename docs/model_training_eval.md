# 🤖 Step 3: Model Training, Evaluation & Selection

## 🎯 Objective
To build, train, and evaluate machine learning models that can accurately **detect fraudulent transactions** based on engineered behavioral and transactional features.

This step focuses on model experimentation — comparing several algorithms, understanding their strengths, and selecting the one that best balances **accuracy and interpretability** for banking use cases.

---

## 🧱 Training Approach Overview

Fraud detection is a **binary classification** problem:

| Label | Description |
|--------|--------------|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

The model learns from historical data labeled as fraud or non-fraud and generalizes patterns to identify new fraud attempts.

---

## 📊 1. Data Splitting

Before training, we’ll split the processed dataset:

| Split | Percentage | Purpose |
|--------|-------------|----------|
| **Train Set** | 70% | Model learns fraud patterns |
| **Validation Set** | 15% | Hyperparameter tuning and model selection |
| **Test Set** | 15% | Final model performance evaluation |

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


🧠 2. Model Candidates

We’ll experiment with multiple algorithms, progressing from simple to advanced:

Model	Type	Why It’s Used
Logistic Regression	Linear	Fast, interpretable baseline
Random Forest	Tree-based	Handles non-linearities well
XGBoost / LightGBM	Gradient Boosting	High accuracy and scalability
Isolation Forest / One-Class SVM	Unsupervised	Detects outliers when labels are limited
Neural Network (Optional)	Deep Learning	Can learn complex fraud patterns
⚙️ 3. Training Workflow

Load and preprocess feature dataset.

Handle class imbalance using:

SMOTE (Synthetic Minority Over-sampling Technique), or

Class-weight adjustment in the model.

Train candidate models on the training set.

Evaluate using the validation set.

Select top-performing model based on metrics.

Test final model on unseen data.

📈 4. Evaluation Metrics

Fraud detection is an imbalanced classification problem — only a small fraction of transactions are fraudulent.
Hence, accuracy alone is misleading. We focus on recall, precision, and F1-score.

Metric	Description	Goal
Precision	% of predicted frauds that are truly fraud	Avoid false positives
Recall (Sensitivity)	% of actual frauds correctly detected	Avoid missing real frauds
F1-score	Balance between precision and recall	Overall performance
AUC-ROC	Measures discrimination between fraud and non-fraud	Higher is better
Confusion Matrix	Breakdown of TP, FP, FN, TN	Understand model behavior
from sklearn.metrics import classification_report, roc_auc_score

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

🧮 5. Model Explainability

In banking, transparency is crucial — especially for compliance and audit.

To make the model explainable:

Use SHAP or LIME to show which features influenced each prediction.

Provide feature importance dashboards for investigators.

import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


This helps answer:

“Why did the model flag this transaction as fraud?”

🏆 6. Model Selection Criteria

We’ll choose the final model based on:

High Recall: Detects most fraudulent transactions.

Acceptable Precision: Minimizes false alarms.

Operational Efficiency: Fast inference in real-time systems.

Explainability: Easy to justify to compliance teams.

In most fraud POCs, XGBoost or LightGBM provides the best trade-off.

🧾 Example Training Script (Simplified)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=10,  # handle class imbalance
    eval_metric="auc"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

📤 7. Model Output

After training and evaluation:

Output	Description
fraud_model.pkl	Serialized trained model (for deployment)
feature_list.json	List of features used during training
evaluation_report.json	Key performance metrics and confusion matrix
shap_summary_plot.png	Feature importance visualization