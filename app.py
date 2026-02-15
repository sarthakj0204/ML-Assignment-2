"""
Streamlit app for Assignment-2: Multi-model Classification Demo

Features (as required):
- Dataset upload option (CSV) (upload test data due to Streamlit free-tier limits)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix / classification report
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
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

st.set_page_config(page_title="ML Assignment-2 | Classification Models", layout="wide")


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"


@st.cache_resource
def load_registry():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}, {}
    metrics = json.loads(metrics_path.read_text())

    model_paths = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
        "Decision Tree": MODEL_DIR / "decision_tree.pkl",
        "KNN": MODEL_DIR / "knn.pkl",
        "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
        "Random Forest": MODEL_DIR / "random_forest.pkl",
        "XGBoost": MODEL_DIR / "xgboost.pkl",
    }

    loaded = {}
    for k, p in model_paths.items():
        if p.exists():
            loaded[k] = joblib.load(p)

    return metrics, loaded


def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
        "Confusion Matrix": confusion_matrix(y, y_pred),
        "Classification Report": classification_report(y, y_pred, output_dict=False),
    }


st.title("Machine Learning Assignment-2: Multi-Model Classification")
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic) — UCI (via sklearn). Upload **test CSV** to evaluate with any selected model.")

metrics_registry, models = load_registry()

if not models:
    st.error("No trained models found. Please run train_models.py first to generate model/*.pkl and model/metrics.json.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()), index=0)
uploaded = st.sidebar.file_uploader("Upload test CSV (same feature columns). Optional: include `target` column.", type=["csv"])
show_training_metrics = st.sidebar.checkbox("Show stored (held-out test) metrics from training", value=True)

# Training metrics table
if show_training_metrics and metrics_registry:
    st.subheader("Model Comparison (stored metrics from training hold-out test set)")
    comp = (
        pd.DataFrame(metrics_registry)
        .T.reset_index()
        .rename(columns={"index": "ML Model Name"})
    )
    st.dataframe(comp, use_container_width=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Upload Evaluation")
    st.write("Upload a CSV containing the same feature columns used during training.")
    st.write("If the CSV includes a `target` column, the app will compute all evaluation metrics + confusion matrix.")
    st.write("If `target` is missing, the app will output predictions and (if available) probabilities.")

    if uploaded is None:
        st.info("Tip: Use the provided sample files in the repo: `sample_test_data_with_target.csv` or `sample_test_data_no_target.csv`.")
    else:
        df = pd.read_csv(uploaded)

        if "target" in df.columns:
            X = df.drop(columns=["target"])
            y = df["target"].astype(int)
            model = models[model_name]
            out = compute_metrics(model, X, y)

            # Metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Accuracy", f"{out['Accuracy']:.4f}")
            mcol2.metric("AUC", f"{out['AUC']:.4f}")
            mcol3.metric("F1", f"{out['F1']:.4f}")

            mcol4, mcol5, mcol6 = st.columns(3)
            mcol4.metric("Precision", f"{out['Precision']:.4f}")
            mcol5.metric("Recall", f"{out['Recall']:.4f}")
            mcol6.metric("MCC", f"{out['MCC']:.4f}")

            st.markdown("#### Confusion Matrix")
            st.dataframe(pd.DataFrame(out["Confusion Matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=False)

            st.markdown("#### Classification Report")
            st.code(out["Classification Report"])
        else:
            X = df.copy()
            model = models[model_name]
            preds = model.predict(X)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]

            out_df = df.copy()
            out_df["prediction"] = preds
            if proba is not None:
                out_df["probability_class1"] = proba

            st.success("Predictions generated (no `target` provided, so evaluation metrics are not computed).")
            st.dataframe(out_df.head(50), use_container_width=True)

with col2:
    st.subheader("About this app")
    st.markdown(
        """
**Models included**
1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)
        """
    )

st.divider()
st.caption("Note: For Streamlit free-tier capacity, upload only test data as suggested in the assignment.")
