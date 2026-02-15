"""
Train and save 6 classification models + evaluation metrics for Assignment-2.

Dataset: Breast Cancer Wisconsin (Diagnostic) from UCI (via sklearn.datasets.load_breast_cancer).
- Instances: 569
- Features: 30 numeric features
- Target: 0 = malignant, 1 = benign

Run:
    python train_models.py

Outputs:
    model/*.pkl              (saved models)
    model/metrics.json       (evaluation metrics on held-out test set)
    sample_test_data_*.csv   (example test files for Streamlit upload)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def evaluate_binary(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback (rare here)
        scores = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_prob)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        ),
    }

    metrics_summary = {}
    detailed = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        fname = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
        )
        joblib.dump(model, model_dir / f"{fname}.pkl")

        det = evaluate_binary(model, X_test, y_test)
        detailed[name] = det
        metrics_summary[name] = {k: det[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}

    # Save metrics
    (model_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    (model_dir / "test_confusion_and_report.json").write_text(json.dumps(detailed, indent=2))

    # Save sample test files (for Streamlit upload)
    test_with_target = X_test.copy()
    test_with_target["target"] = y_test.values
    test_with_target.to_csv(out_dir / "sample_test_data_with_target.csv", index=False)
    X_test.to_csv(out_dir / "sample_test_data_no_target.csv", index=False)

    print("Training complete. Saved models + metrics to:", model_dir)


if __name__ == "__main__":
    main()
