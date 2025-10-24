# train_and_register.py
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from azureml.core import Workspace

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

# models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# import your preprocessing pipeline (must be accessible)
from pipeline_preprocessing import preprocessing_pipeline

# ---------------------------
# Load data
# ---------------------------
import os

os.chdir("../fraud_detection_project_azure")
print("Current working directory:", os.getcwd())
df = pd.read_csv("./data/transactions.csv", parse_dates=["timestamp"])

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# class weights
# ---------------------------
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
weights = dict(zip(np.unique(y_train), class_weights))
print("Computed class weights:", weights)

# ---------------------------
# models to evaluate
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
# connect MLflow to Azure ML
# ---------------------------
ws = Workspace.from_config(path="../config.json")
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("fraud-detection-training")


# ---------------------------
# helper: compute metrics dict
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
# training loop: train & evaluate on train+test, log to MLflow
# ---------------------------
best_auc = -1.0
best_model_name = None
best_run_id = None
best_pipeline = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        print(f"\n--- Training {name} (run_id={run_id}) ---")

        # unified pipeline: preprocessing + model
        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessing_pipeline), ("classifier", model)]
        )

        # fit
        model_pipeline.fit(X_train, y_train)

        # predictions on train
        y_train_pred = model_pipeline.predict(X_train)
        y_train_proba = None
        if hasattr(model_pipeline.named_steps["classifier"], "predict_proba"):
            y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]

        # predictions on test
        y_test_pred = model_pipeline.predict(X_test)
        y_test_proba = None
        if hasattr(model_pipeline.named_steps["classifier"], "predict_proba"):
            y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # compute metrics
        train_metrics = compute_metrics_dict(y_train, y_train_pred, y_train_proba)
        test_metrics = compute_metrics_dict(y_test, y_test_pred, y_test_proba)

        # log metrics (prefix with train_ / test_)
        for k, v in train_metrics.items():
            if v is None:
                continue
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            if v is None:
                continue
            mlflow.log_metric(f"test_{k}", v)

        # log params
        try:
            mlflow.log_params(model.get_params())
        except Exception:
            # some models have complex params that aren't JSON-serializable
            pass

        # Log example input & signature (use small sample)
        example_input = X_train.head(3)
        # predict on sample for signature output (labels)
        try:
            example_output = model_pipeline.predict(example_input)
            signature = infer_signature(example_input, example_output)
        except Exception:
            signature = None

        # log the pipeline model artifact into this run
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model_pipeline",
            signature=signature,
            input_example=example_input,
        )

        # print quick console report
        print(f"{name} - Train metrics: {train_metrics}")
        print(f"{name} - Test metrics : {test_metrics}")
        print("Classification report (test):")
        print(classification_report(y_test, y_test_pred, zero_division=0))

        # update best model by test roc_auc (if available), fallback to test f1
        model_auc = test_metrics.get("roc_auc")
        model_f1 = test_metrics.get("f1_score")
        score_for_selection = model_auc if model_auc is not None else model_f1

        if score_for_selection is not None and score_for_selection > best_auc:
            best_auc = score_for_selection
            best_model_name = name
            best_run_id = run_id
            best_pipeline = model_pipeline

# ---------------------------
# after loop: register best model pipeline
# ---------------------------
if best_run_id is None:
    raise RuntimeError("No successful run to register as best model.")

print(
    f"\nBest model chosen: {best_model_name} (selection score={best_auc}), run_id={best_run_id}"
)

# register the best model artifact (the model_pipeline artifact from the best run)
model_uri = f"runs:/{best_run_id}/model_pipeline"
registered_model_name = "fraud_detection_best_pipeline"

# register model in MLflow registry (which maps to Azure ML model registry when tracking URI is Azure)
mlflow.register_model(model_uri=model_uri, name=registered_model_name)
print(
    f"Registered model '{registered_model_name}' from run {best_run_id} at uri {model_uri}"
)

# optionally, transition model to stage "Staging"/"Production" via MLflow client (manual or automated later)
