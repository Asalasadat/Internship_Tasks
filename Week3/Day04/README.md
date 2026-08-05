## Day 4 — Model Comparison, Evaluation & Feature Importance

### Overview
In this day, I trained and compared four different classification models 
on the same telecom customer churn dataset and train/test split, evaluated 
them using a consistent metric (F1-score), interpreted feature importances 
from the best model, and identified which model generalized best and why.

### Topics Covered
- Training multiple classifiers (Decision Tree, Random Forest, SVM, k-NN) 
  on the same data split
- Evaluating models consistently using accuracy and weighted F1-score
- Assembling model results into a single comparison table
- Extracting and interpreting feature importances from a Random Forest
- Diagnosing overfitting via the train/test performance gap
- Justifying model selection based on generalization, not just raw accuracy

### Key Results

**Model Comparison (F1-score, weighted)**

| Model | Train F1 | Test F1 | Gap |
|---|---|---|---|
| Random Forest | 0.9980 | 0.7804 | 0.2177 |
| Decision Tree | 0.9980 | 0.7353 | 0.2627 |
| k-NN | 0.8204 | 0.7283 | 0.0921 |
| SVM | 0.6223 | 0.6222 | 0.0001 |

**Top Feature Importances (Random Forest)**

| Feature | Importance |
|---|---|
| MonthlyCharges | 0.177 |
| tenure | 0.172 |
| TotalCharges | 0.165 |
| Contract | 0.083 |
| PaymentMethod | 0.052 |

### Key Findings

**Random Forest was the best-performing model**, achieving the highest 
test F1-score. Its ensemble nature (averaging many trees) makes it more 
robust to overfitting than a single Decision Tree, which memorized the 
training data almost perfectly but generalized noticeably worse.

**SVM showed near-zero overfitting** (train ≈ test F1), indicating strong 
generalization, but its overall performance was the lowest — suggesting 
its decision boundary was too simple for this dataset.

**Billing and tenure dominate churn prediction**: `MonthlyCharges`, 
`tenure`, and `TotalCharges` were the top three predictors, followed by 
`Contract` type — consistent with typical telecom churn behavior (newer, 
higher-billed, short-contract customers churn more).

**Conclusion:** Random Forest is the recommended model for this dataset. 
Its remaining overfitting gap could likely be reduced further by tuning 
`max_depth`, `min_samples_leaf`, or `n_estimators`.

### Tools Used
Pandas • Scikit-learn • Jupyter Notebook
