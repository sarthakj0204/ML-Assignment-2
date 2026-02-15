# Machine Learning Assignment-2 — Multi-Model Classification + Streamlit Deployment

## a. Problem statement
Build and evaluate **six different classification models** on a single public dataset, compare them using standard classification metrics, and deploy an **interactive Streamlit web application** where users can upload **test CSV data**, select a model, and view metrics + confusion matrix / classification report.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository (accessible via `sklearn.datasets.load_breast_cancer`).

- **Type:** Binary classification (0 = malignant, 1 = benign)
- **Instances:** 569 (>= 500)
- **Features:** 30 numeric features (>= 12)
- **Public source:** UCI dataset (commonly mirrored in scikit-learn)

## c. Models used
The following **6 models** are trained on the same dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table (Evaluation Metrics)
Metrics computed on a held-out test set (25% split, stratified, random_state=42).

| ML Model Name       |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC Score |
|:--------------------|-----------:|------------:|------------:|---------:|-----------:|------------:|
| Logistic Regression |     0.986  |      0.9977 |      0.9889 |   0.9889 |     0.9889 |      0.97   |
| Decision Tree       |     0.9231 |      0.9234 |      0.954  |   0.9222 |     0.9379 |      0.8378 |
| KNN                 |     0.979  |      0.9923 |      0.9677 |   1      |     0.9836 |      0.9555 |
| Naive Bayes         |     0.9371 |      0.9893 |      0.9263 |   0.9778 |     0.9514 |      0.865  |
| Random Forest       |     0.958  |      0.9941 |      0.9565 |   0.9778 |     0.967  |      0.9098 |
| XGBoost             |     0.965  |      0.9964 |      0.957  |   0.9889 |     0.9727 |      0.9251 |

### Observations on model performance

| ML Model Name            | Observation about model performance                                                                           |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Strong overall performance with the best AUC and MCC; linear decision boundary works well on this dataset.    |
| Decision Tree            | Lower generalization compared to other models; single-tree tends to overfit, reducing AUC/MCC.                |
| kNN                      | Very high recall (1.0) with strong accuracy; performance depends on scaling and choice of k.                  |
| Naive Bayes              | High AUC but lower precision/accuracy; independence assumption is imperfect though still competitive.         |
| Random Forest (Ensemble) | Improves over single tree with better stability; strong AUC and balanced precision/recall.                    |
| XGBoost (Ensemble)       | Very strong performance close to Logistic Regression; gradient boosting captures non-linearities effectively. |

---

## Project structure (as required)
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   ├── *.pkl
│   ├── metrics.json
│   └── test_confusion_and_report.json
│-- train_models.py
│-- sample_test_data_with_target.csv
│-- sample_test_data_no_target.csv
```

## How to run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Streamlit app features checklist
- Dataset upload option (CSV) ✅ (recommended: upload **test** data only)
- Model selection dropdown ✅
- Display of evaluation metrics ✅
- Confusion matrix / classification report ✅

## Deployment on Streamlit Community Cloud
1. Push this repository to GitHub  
2. Go to Streamlit Community Cloud → New app  
3. Select repo + branch → choose `app.py` → Deploy
