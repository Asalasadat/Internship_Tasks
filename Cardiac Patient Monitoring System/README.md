# Cardiac Patient Monitoring System — AI & Machine Learning Project

## Overview
Heart disease prediction sounds straightforward until you actually look at the data: severity scores that barely have any patients in the higher categories, blood pressure readings of zero that are medically impossible, and four different hospitals each recording data slightly differently. This project builds a classification pipeline on the UCI Heart Disease dataset (920 patients combined from Cleveland, Hungary, Switzerland, and VA Long Beach) that works through each of these problems in turn — cleaning, exploring, engineering features, comparing models, and tuning the winner — landing on a model that catches real disease cases reliably enough to matter in a screening context.

## Objective
Predict whether a patient has heart disease (present / not present) from their clinical and demographic profile, and identify which factors actually drive that prediction — not just build a model that runs, but one whose reasoning holds up to scrutiny.

## Dataset
- **Source:** UCI Heart Disease dataset (public)
- **Size:** 920 records × 16 columns before cleaning
- **Target:** `num` — originally a 5-level severity score (0 = no disease, 1–4 = increasing severity), reframed as binary after the multi-class version proved unworkable (see below)

## The Work, Step by Step

### 1. Cleaning: Two Different Kinds of "Missing"
The dataset's missingness wasn't uniform — some columns were untouched, while `ca` was missing in nearly two-thirds of rows. Numeric columns were imputed with the median (robust to outliers) and categorical columns with the mode. But a second, sneakier problem showed up during EDA: `trestbps` and `chol` both contained values of exactly `0` — a resting blood pressure or cholesterol reading of zero isn't a rare case, it's not survivable. These weren't missing values pandas would flag automatically; they were disguised as real numbers. Once spotted via box plots, they were treated as data errors and removed, along with the non-predictive `id` column. All categorical columns were then one-hot encoded (`drop_first=True` to avoid redundant, perfectly correlated columns).

### 2. What the Data Actually Looks Like
Distributions, correlations, and class balance were examined before touching any model. The correlation heatmap pointed early to `oldpeak`, `ca`, and `thalch` as the features most linked to the target — a pattern that later showed up again, independently, in the trained model's coefficients. But the more consequential discovery was in the target itself: the original 5-class severity score was badly imbalanced. Class 0 made up 52% of patients, class 1 another 27%, and classes 2 through 4 combined were just 20% — with class 4 sitting at a mere 2.94% (22 patients total). That imbalance would turn out to be the central problem of the whole project.

### 3. Engineering Features the Raw Columns Don't Capture
Three features were built to encode relationships no single raw column expresses on its own:
- **`hr_reserve`** = (220 − age) − thalch — a raw max heart rate means little without adjusting for what's expected at a given age; this feature does that adjustment directly.
- **`bp_chol_product`** = trestbps × chol — an interaction term for compounding risk when both blood pressure and cholesterol are elevated together, a pattern a linear model can't otherwise learn without it being handed the interaction explicitly.
- **`high_risk_flag`** — a single binary signal combining three established risk indicators (exercise-induced angina, ST depression > 1.0, affected-vessel count > 0) into one "multiple red flags" marker.

### 4. Why Binary, Not Five Classes
The first real modeling attempt — Random Forest on the full 5-class target, evaluated with stratified cross-validation — returned a **Mean F1 (macro) of just 0.30**. With so few examples of severity classes 2 through 4, the model had almost nothing to learn their patterns from. Reframing the exact same problem as binary (disease present or not) and rerunning the identical model and evaluation setup raised **Mean F1 to 0.79** — a 160% improvement using the same features and the same algorithm. This confirmed the original score wasn't a sign of weak features or a bad model choice; it was purely a class-imbalance problem, solved by asking a more clinically relevant question in the first place.

### 5. Letting Four Models Compete Fairly
With the target reframed, four classifiers were compared on identical 5-fold stratified cross-validation:

| Model | Mean F1 | Std |
|---|---|---|
| **Logistic Regression** | **0.8046** | 0.046 |
| Random Forest | 0.7881 | 0.033 |
| Gradient Boosting | 0.7654 | 0.036 |
| SVM | 0.6155 | 0.095 |

The simplest model won. Logistic Regression beat two ensemble methods, which suggests the relationship between these features and heart disease is largely linear rather than requiring complex non-linear splits. SVM performed worst by a wide margin, likely penalized by unscaled features.

### 6. Tuning the Winner
`GridSearchCV` (with `RandomizedSearchCV` tested as a faster alternative) searched Logistic Regression's hyperparameters directly. The winning combination — `C=1`, `penalty='l2'`, `class_weight='balanced'` — pushed cross-validated F1 from 0.8046 to **0.8115**, a modest but meaningful gain. More importantly, `class_weight='balanced'` was a targeted fix: it directly addresses the kind of error that matters most in this context, discussed next.

### 7. The Error That Actually Matters
Accuracy and F1 tell part of the story, but in a screening tool, false negatives — patients with real disease predicted as healthy — are the costly mistake, not false alarms. Comparing the untuned Random Forest baseline against the tuned Logistic Regression on the held-out test set:

| Metric | Random Forest (baseline) | Tuned Logistic Regression | Change |
|---|---|---|---|
| Test F1 | 0.7111 | **0.7660** | +0.055 |
| Accuracy | 0.74 | **0.78** | +0.04 |
| False Negatives | 23 | **17** | −6 |
| Recall (disease class) | 0.68 | **0.76** | +0.08 |

Six more real disease cases caught, with no drop in precision. This is the result the tuning was actually aimed at — not a slightly better score for its own sake, but fewer missed diagnoses.

### 8. Checking the Model Isn't Fooling Itself
A model that performs well on paper can still be quietly overfitting. The final model's train-vs-test gap (Train F1: 0.8268, Test F1: 0.7660, Gap: 0.0608) is small — a healthy sign of genuine generalization rather than memorization. A validation curve across Random Forest's `max_depth` made the contrast obvious: past `max_depth ≈ 6`, train F1 climbed to a perfect 1.00 while test F1 declined, a textbook overfitting signature. That comparison reinforced, independently, why the simpler and regularized Logistic Regression was the right choice.

### 9. What the Model Actually Learned
Reading the final coefficients turns the model from a black box into something a clinician could reason about:

**Increases predicted risk:** `sex_Male`, `exang_True` (exercise-induced angina), `ca` (affected vessel count), `slope_flat`, `oldpeak` (ST depression).

**Decreases predicted risk:** `cp_atypical angina` (the single strongest effect in the model), `cp_non-anginal`, `cp_typical angina`, `thal_normal` — non-asymptomatic chest pain types and normal thallium results are protective relative to the baseline category.

**Negligible effect:** `age`, `trestbps`, `chol`, `thalch` — despite their reputation as classic cardiac risk factors, their signal appears to already be captured indirectly through more specific correlated features like `cp`, `exang`, and `ca`.

## Key Findings
1. Reframing the target as binary rather than 5-class severity was the single most consequential decision in the project — the original framing was unworkable due to class imbalance, no matter how good the model or features were.
2. A simple, regularized Logistic Regression outperformed Random Forest, Gradient Boosting, and SVM, pointing to a largely linear relationship between these features and disease presence.
3. `class_weight='balanced'` measurably reduced false negatives — the clinically costliest error type for a screening tool.
4. Chest pain type and exercise-induced angina are the most influential and clinically interpretable predictors, consistent with established cardiology knowledge.

## Limitations
- This is an educational/analytical project, not a clinical diagnostic tool — it must not be used for real patient diagnosis or treatment decisions.
- The dataset combines four hospital sources with differing data-collection practices, which may introduce site-specific bias (`dataset_VA Long Beach` showed a coefficient effect that looks more like a population/site artifact than a genuine clinical signal).
- Missing values were imputed with median/mode; more sophisticated imputation methods weren't explored.
- Final test metrics reflect a single held-out split; cross-validation informed model selection, but the reported test numbers come from one split, not an average.

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
