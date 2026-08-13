# Day 5 — Pipelines & End-to-End Hyperparameter Search

Builds a leak-safe `sklearn` pipeline (preprocessing + model) for the 
California housing dataset, then tunes both preprocessing and model 
hyperparameters jointly with `GridSearchCV`, and compares the tuned 
pipeline against an untuned baseline on a held-out test set.

## Dataset

`housing.csv` (California Housing dataset), sampled down to 2,000 rows 
for tractable search times. Target: `median_house_value`.

## Workflow

1. **Load & sample** — read the CSV, sample 2,000 rows (`random_state=42`).
2. **Handle missing values** — `total_bedrooms` imputed with the median 
   (robust to outliers).
3. **Build the pipeline**
   - **Numeric** (`longitude`, `latitude`, `housing_median_age`, 
     `total_rooms`, `total_bedrooms`, `population`, `households`, 
     `median_income`): median imputation → `StandardScaler`.
   - **Categorical** (`ocean_proximity`): most-frequent imputation → 
     `OneHotEncoder(handle_unknown='ignore')`.
   - Combined via `ColumnTransformer`, then chained with 
     `RandomForestRegressor` into one `full_pipeline`.
   - Wrapping everything in a `Pipeline` ensures imputation/scaling/encoding 
     are refit only on each fold's training data — no leakage from 
     validation/test data into the transforms.
4. **Feature engineering** — added three ratio features: 
   `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
5. **Joint hyperparameter search** — 80/20 train/test split, then 
   `GridSearchCV` (5-fold) over both preprocessing and model parameters:
   - `preprocessor__num__imputer__strategy`: `median`, `mean`
   - `model__n_estimators`: `100`, `200`
   - `model__max_depth`: `15`, `25`, `None`
   - `model__min_samples_split`: `2`, `5`
   
   (24 combinations total, fit on `X_train` only — the earlier bug of 
   fitting on the full `X` is fixed here.)
6. **Baseline comparison** — trained an untuned pipeline 
   (`n_estimators=50`, default depth) and compared RMSE, MAE, and R² 
   against the tuned pipeline, both scored on the same held-out `X_test`.

## Results

**Best Parameters:** `{'model__max_depth': None, 'model__min_samples_split': 5, 'model__n_estimators': 100, 'preprocessor__num__imputer__strategy': 'median'}`

| Metric | Value |
|---|---|
| Test RMSE | $60,078.13 |
| Test R² | 0.7390 |

`max_depth=None` paired with a higher `min_samples_split=5` suggests the 
trees regularize via split thresholds rather than an explicit depth cap. 
`median` imputation outperformed `mean`, consistent with the right-skew 
in income and room/bedroom counts.

**Baseline vs. Tuned (held-out test set):**

| Metric | Baseline Model | Tuned Pipeline |
|---|---|---|
| RMSE ($) | 60,433.32 | 60,078.13 |
| MAE ($) | 41,861.78 | 41,928.54 |
| R² Score | 0.7359 | 0.7390 |

**Total RMSE Improvement:** 0.59% — a small, practically negligible gain. 
MAE slightly worsened with tuning even as RMSE improved, suggesting the 
tuned model reduced a few large errors (RMSE is more sensitive to these) 
at a marginal cost to typical-case accuracy. Overall, this confirms the 
untuned baseline was already close to this ceiling — the search added 
limited real-world value.

## Tools

- pandas
- scikit-learn
