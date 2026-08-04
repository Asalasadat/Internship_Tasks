# Day 3 — Logistic Regression & Classification Metrics
 
## Overview
This notebook trains and evaluates a **Logistic Regression** model to predict customer churn using the Telco Customer Churn dataset (`churn/churn.csv`).
 
## Tasks
 
### Task 1: Train a Logistic Regression Model
- Loaded the churn dataset into a Pandas DataFrame and checked for missing values.
- Label-encoded all categorical (object dtype) columns to numeric form.
- Split features (`X`) and target (`y = Churn`).
- Split into training (80%) and test (20%) sets using `train_test_split` with `stratify=y` to preserve the churn/no-churn ratio.
- Trained a `LogisticRegression` model (`max_iter=1000`).
### Task 2: Predictions & Confusion Matrix
Generated predictions on the test set and computed the confusion matrix:
 
|                | Predicted: No Churn | Predicted: Churn |
|----------------|----------------------|-------------------|
| **Actual: No Churn** | 917 (TN) | 118 (FP) |
| **Actual: Churn**    | 180 (FN) | 194 (TP) |
 
The model correctly identifies just over half of churned customers (recall 51.9%), missing 180 (FN) — the costliest error type for retention efforts. Precision of 62.2% means roughly a third of predicted churners (118, FP) are false alarms. The model outperforms random guessing but would benefit from threshold tuning or class weighting to reduce missed churners.
 
### Task 3: Precision, Recall & F1 (classification_report)
 
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Churn | 0.84 | 0.89 | 0.86 | 1035 |
| Churn | 0.62 | 0.52 | 0.57 | 374 |
| **Accuracy** | | | **0.79** | 1409 |
| Macro avg | 0.73 | 0.70 | 0.71 | 1409 |
| Weighted avg | 0.78 | 0.79 | 0.78 | 1409 |
 
The Churn class performs notably weaker than No Churn, reflecting class imbalance (374 vs. 1035 samples). The gap between macro avg (0.71) and weighted avg (0.78) confirms the model is biased toward the majority class — accuracy alone overstates real-world performance.
 
### Task 4: Precision vs. Recall — Which Matters More?
**Recall matters more** for this problem. A missed churner (false negative) represents lost recurring revenue that may be unrecoverable, while a false positive costs only a modest, low-risk retention incentive spent on a loyal customer. This asymmetry means the model should be optimized to minimize false negatives — e.g. via `class_weight='balanced'`, threshold adjustment, or resampling — rather than treating overall accuracy as the target metric.
 
### Task 5: AUC-ROC
 
**AUC-ROC = 0.838**
 
This indicates strong discriminative power: the model ranks an actual churner above a non-churner about 84% of the time, well above the 0.5 random baseline. This is notably higher than the recall (0.52) observed at the default 0.5 threshold, showing the model captures real signal but the default threshold underuses it. The ROC curve's steep early rise (~50% TPR at only ~10% FPR) suggests lowering the threshold to ~0.3 would meaningfully improve recall — the priority metric for this use case — at a modest precision cost.
 
## Key Takeaway
The model has solid underlying discriminative ability (AUC 0.838), but its default 0.5 decision threshold underperforms on recall, the metric that matters most for churn prevention. Threshold tuning or class weighting is the recommended next step to better align the model's decisions with the business objective of catching at-risk customers.
 
## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn (`LogisticRegression`, `train_test_split`, classification metrics, ROC/AUC)
- Matplotlib
- Jupyter Notebook
