# train_and_register.py
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from azureml.core import Workspace, Run

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import your preprocessing pipeline
from pipeline_preprocessing import preprocessing_pipeline


# ---------------------------
# Azure ML Job Context Setup
# ---------------------------
# Get the Azure ML run context (so logs are visible in the portal)
run = Run.get_context()

if run.id.startswith("OfflineRun"):
    # Running locally
    ws = Workspace.from_config(path="./config.json")
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
else:
    # Running on Azure ML Compute
    ws = run.experiment.workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

mlflow.set_experiment("fraud-detection-training")

print(" Connected to Azure ML Workspace:", ws.name)


# ---------------------------
# Load data
# ---------------------------
data_path = os.getenv("DATA_PATH", "./data/transactions.csv")
print(f"📂 Loading data from: {data_path}")

df = pd.read_csv(data_path, parse_dates=["timestamp"])
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Handle class imbalance
# ---------------------------
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
weights = dict(zip(np.unique(y_train), class_weights))
print("Computed class weights:", weights)


# ---------------------------
# Define Models
# ---------------------------
models = {
    "RandomForest": RandomForestClassifier(
        class_weight=weights, random_state=42, n_estimators=200
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=float(weights[1]),
        random_state=42,
        n_estimators=300,
        use_label_encoder=False,
        eval_metric="logloss",
    ),
    "LightGBM": LGBMClassifier(
        scale_pos_weight=float(weights[1]), random_state=42, n_estimators=300
    ),
    "LogisticRegression": LogisticRegression(
        class_weight=weights, max_iter=1000, random_state=42
    ),
    "DecisionTree": DecisionTreeClassifier(class_weight=weights, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}


# ---------------------------
# Helper function
# ---------------------------
def compute_metrics_dict(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


# ---------------------------
# Train + Evaluate + Log to MLflow
# ---------------------------
best_auc = -1.0
best_model_name = None
best_run_id = None
best_pipeline = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n--- Training {name} ---")

        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessing_pipeline), ("classifier", model)]
        )

        # Train
        model_pipeline.fit(X_train, y_train)

        # Predict
        y_train_pred = model_pipeline.predict(X_train)
        y_test_pred = model_pipeline.predict(X_test)

        y_train_proba = None
        y_test_proba = None
        if hasattr(model_pipeline.named_steps["classifier"], "predict_proba"):
            y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]
            y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        train_metrics = compute_metrics_dict(y_train, y_train_pred, y_train_proba)
        test_metrics = compute_metrics_dict(y_test, y_test_pred, y_test_proba)

        # Log metrics
        mlflow.log_metrics(
            {f"train_{k}": v for k, v in train_metrics.items() if v is not None}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if v is not None}
        )

        # Log params
        mlflow.log_params(
            {
                k: v
                for k, v in model.get_params().items()
                if isinstance(v, (int, float, str, bool))
            }
        )

        # Log model
        example_input = X_train.head(3)
        try:
            example_output = model_pipeline.predict(example_input)
            signature = infer_signature(example_input, example_output)
        except Exception:
            signature = None

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model_pipeline",
            signature=signature,
            input_example=example_input,
        )

        print(f"{name} - Test metrics: {test_metrics}")

        score_for_selection = test_metrics.get("roc_auc") or test_metrics.get(
            "f1_score"
        )
        if score_for_selection and score_for_selection > best_auc:
            best_auc = score_for_selection
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id
            best_pipeline = model_pipeline


# ---------------------------
# Register best model
# ---------------------------
if best_run_id:
    print(f"\n Best model: {best_model_name} (score={best_auc})")
    model_uri = f"runs:/{best_run_id}/model_pipeline"
    registered_model_name = "fraud_detection_best_pipeline"

    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"Model registered to Azure ML: {registered_model_name}")
else:
    print(" No successful run to register.")
