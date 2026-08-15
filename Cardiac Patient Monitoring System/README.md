# Cardiac Patient Monitoring System — AI & Machine Learning Project

## Project Overview
This project builds a machine-learning classification pipeline to predict the presence of heart disease using the **UCI Heart Disease dataset** (combined records from Cleveland, Hungary, Switzerland, and VA Long Beach, 920 patients total). The analysis covers data cleaning, exploratory data analysis, supervised classification, cross-validation, hyperparameter tuning, and model comparison.

## Objective
Predict whether a patient has heart disease (binary classification: disease present / not present) based on clinical and demographic features, and identify which factors most strongly drive that prediction.

## Dataset
- **Source:** UCI Heart Disease dataset (public)
- **Size:** 920 records × 16 columns (before cleaning)
- **Target variable:** `num` — originally a 5-level disease severity score (0 = no disease, 1–4 = increasing severity), reframed as binary (`0` = no disease, `1` = disease present) after multi-class modeling proved unreliable (see Findings below)

## Workflow Summary

### 1. Data Cleaning
- Dropped the non-predictive `id` column.
- Identified and imputed missing values: numeric columns filled with median (robust to outliers), categorical columns filled with mode. Missingness ranged from 0 in most columns up to ~66% in `ca`.
- Detected and removed disguised missing data: `trestbps` and `chol` contained physiologically impossible `0` values (a resting blood pressure or cholesterol of 0 is not survivable), identified via box plots and removed as data errors.
- One-hot encoded all categorical columns (`drop_first=True` to avoid multicollinearity).

### 2. Exploratory Data Analysis
- Visualized distributions of all numeric and categorical features (histograms, bar charts, box plots).
- Built a correlation heatmap across all features. Strongest correlations with the target: `oldpeak` (+0.5), `ca` (+), `thalch` (−), consistent with later model coefficients.
- Identified severe class imbalance in the original 5-class target: class 0 = 52%, class 1 = 27%, classes 2–4 combined = only 20% (class 4 = 2.94%, 22 patients).

### 3. Feature Engineering
Three new features were derived to capture clinically meaningful relationships not present in any single raw column:
- **`hr_reserve`** = (220 − age) − thalch — adjusts max heart rate achieved for age-predicted maximum, since a raw heart rate value means little without accounting for the patient's age.
- **`bp_chol_product`** = trestbps × chol — an interaction term capturing compounding cardiovascular risk when both blood pressure and cholesterol are elevated, which a linear model cannot otherwise learn without an explicit interaction term.
- **`high_risk_flag`** — a binary flag combining three established risk indicators (exercise-induced angina, ST depression > 1.0, and affected vessel count > 0) into a single "multiple red flags present" signal.

### 4. Supervised Learning — Baseline
- **Key finding:** modeling `num` as a 5-class problem with Random Forest + `StratifiedKFold` cross-validation yielded a poor **Mean F1 (macro) = 0.30**, driven by the sparse severity classes.
- Reframing the target as **binary** (disease present or not) and rerunning the identical model and cross-validation setup raised **Mean F1 to 0.79** — a 160% improvement, confirming the original issue was class imbalance, not weak features or a poor model choice.

### 5. Model Comparison
Four classifiers were compared via 5-fold stratified cross-validation (F1 scoring):

| Model | Mean F1 | Std |
|---|---|---|
| **Logistic Regression** | **0.8046** | 0.046 |
| Random Forest | 0.7881 | 0.033 |
| Gradient Boosting | 0.7654 | 0.036 |
| SVM | 0.6155 | 0.095 |

Logistic Regression performed best, suggesting the underlying relationship is largely linear. SVM underperformed, likely due to unscaled features.

### 6. Hyperparameter Tuning
`GridSearchCV` (and `RandomizedSearchCV` as a faster alternative) were used to tune Logistic Regression:
- **Best parameters:** `C=1`, `penalty='l2'`, `class_weight='balanced'`
- **Best CV F1:** 0.8115 (a modest gain over the untuned default of 0.8046)
- `class_weight='balanced'` directly addressed the false-negative problem observed in the Random Forest baseline.

### 7. Final Model Evaluation

| Metric | Random Forest (baseline) | Tuned Logistic Regression | Change |
|---|---|---|---|
| Test F1 | 0.7111 | **0.7660** | +0.055 |
| Accuracy | 0.74 | **0.78** | +0.04 |
| False Negatives | 23 | **17** | −6 |
| Recall (disease class) | 0.68 | **0.76** | +0.08 |

The tuned Logistic Regression model was selected as final: it caught 6 more true disease cases than Random Forest with no drop in precision — an important trade-off in a medical screening context, where missing a real case (false negative) is costlier than a false alarm.

### 8. Overfitting Check
Comparing train vs. test F1 for the final model (Train F1: 0.8268, Test F1: 0.7660, Gap: 0.0608) shows a small, healthy gap — the model generalizes well and is not overfitting. This was cross-checked against a validation curve across `max_depth` values for Random Forest, which showed clear overfitting beyond `max_depth ≈ 6` (Train F1 reaching 1.00 while Test F1 declined) — reinforcing the decision to favor the simpler, regularized Logistic Regression model.

### 9. Feature Interpretation
Coefficients from the final Logistic Regression model:

**Increases predicted risk:** `sex_Male`, `exang_True` (exercise-induced angina), `ca` (number of affected vessels), `slope_flat`, `oldpeak` (ST depression).

**Decreases predicted risk:** `cp_atypical angina` (strongest effect overall), `cp_non-anginal`, `cp_typical angina`, `thal_normal` — reflecting that non-asymptomatic chest pain types and normal thallium test results are protective relative to the baseline.

**Negligible effect:** `age`, `trestbps`, `chol`, `thalch` — their signal appears to be captured indirectly through correlated clinical features like `cp`, `exang`, and `ca`.

## Key Findings
1. Framing the target as binary rather than multi-class severity was essential — the original 5-class problem was unworkable due to class imbalance.
2. A simple, regularized Logistic Regression outperformed more complex models (Random Forest, Gradient Boosting, SVM), suggesting the relationship between features and disease presence is largely linear.
3. `class_weight='balanced'` meaningfully reduced false negatives — the most clinically important error type for a screening tool.
4. Chest pain type (`cp`) and exercise-induced angina (`exang`) are the most clinically interpretable and influential predictors, consistent with established cardiology knowledge.

## Limitations
- This is an educational/analytical project, not a clinical diagnostic tool — it must not be used for real patient diagnosis or treatment decisions.
- The dataset combines four hospital sources with differing data-collection practices and missingness patterns, which may introduce site-specific bias (e.g. `dataset_VA Long Beach` showed a coefficient effect that is more likely a population/site artifact than a genuine clinical signal).
- Median/mode imputation was used for missing values; more sophisticated imputation methods were not explored.
- The final model was evaluated on a single train/test split for the reported test metrics; cross-validation was used during model selection but the final numbers reflect one held-out split.

## Tools Used
- Python, Jupyter Notebook
- Pandas, NumPy — data loading, cleaning, feature engineering
- Matplotlib, Seaborn — visualization (distributions, box plots, correlation heatmap)
- Scikit-learn — `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `train_test_split`, `StratifiedKFold`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, classification metrics

## How to Run
1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the dataset (`heart_disease_uci.csv`) in the `data/` folder.
4. Open and run `Demo.ipynb` from top to bottom in Jupyter or VS Code.

## Files
- `Demo.ipynb` — full notebook: data cleaning, EDA, modeling, evaluation, and interpretation
- `requirements.txt` — exact library versions for reproducibility
