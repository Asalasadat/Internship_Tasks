 # Day 4 — Feature Engineering & Hyperparameter Tuning
 
Predicting California median house values with a tuned Random Forest 
Regressor, using engineered features and both `GridSearchCV` and 
`RandomizedSearchCV`.
 
## Dataset
 
`Housing Prices/housing.csv` — the California Housing dataset (block-group 
level), with the following columns:
 
| Column | Description |
|---|---|
| `longitude`, `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms`, `total_bedrooms` | Total room/bedroom counts |
| `population`, `households` | Block population and household counts |
| `median_income` | Median household income |
| `median_house_value` | **Target** — median house value |
| `ocean_proximity` | Categorical distance to ocean (dropped for this run) |
 
## Workflow
 
1. **Load & clean** — read the CSV; `total_bedrooms` had 207 missing 
   values, filled with the column median (robust to outliers).
2. **Feature engineering** — three ratio-based features added to surface 
   per-unit signal hidden in raw totals:
   - `rooms_per_household` = `total_rooms / households`
   - `bedrooms_per_room` = `total_bedrooms / total_rooms`
   - `population_per_household` = `population / households`
3. **Prep for modeling** — dropped `ocean_proximity`, sampled 2,000 rows 
   for tractable search times, split 80/20 train/test.
4. **Hyperparameter search** — defined a `RandomForestRegressor` param 
   grid (`n_estimators`, `max_depth`, `min_samples_split`, 
   `min_samples_leaf`, `max_features`); 324 total combinations.
   - `GridSearchCV` (5-fold, all 324 combos = 1,620 fits) was interrupted 
     manually — too slow for this environment.
   - Switched to `RandomizedSearchCV` (30 sampled combinations, 150 fits), 
     scored on `neg_root_mean_squared_error`.
5. **Baseline comparison** — evaluated an untuned `RandomForestRegressor` 
   (`n_estimators=100`, defaults) under the same 5-fold CV for a fair 
   comparison against the tuned model.
6. **Diagnostics** — extracted `feature_importances_` from the best 
   estimator and grouped `cv_results_` by each hyperparameter to see 
   which one moved the score most.
## Results
 
**Best Parameters:** `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 1.0, 'max_depth': 20}`
 
| Model | CV RMSE |
|---|---|
| Baseline (`n_estimators=100`, default depth) | 62,180.64 |
| Tuned (RandomizedSearchCV) | 61,895.55 |
| **Improvement** | **285.09 (0.46%)** |
 
Tuning produced no meaningful improvement over the default configuration — 
the gain is within normal cross-validation noise, suggesting the defaults 
were already close to optimal for this dataset.
 
**Top features:** `median_income` dominates (~52% importance), followed 
by the engineered `population_per_household` (~14%) — which outranks 
several raw features, confirming it added real signal.
 
**Most impactful hyperparameter:** `max_features` (RMSE spread ~1,446 
across `1.0` / `sqrt` / `log2`) — using all features per split beat both 
restricted options. `n_estimators` mattered least, indicating the 
ensemble had already converged by 100–200 trees.
 
## Tools
 
```
pandas
numpy
scikit-learn
Jupyter Notebook
```
 
## Usage
 
Open `Day4.ipynb` in Jupyter and run the cells top to bottom. Update the 
`pd.read_csv(...)` path if `housing.csv` is stored elsewhere. Note: the 
full `GridSearchCV` cell is slow (1,620 fits) — skip it and go straight 
to the `RandomizedSearchCV` cell if you just want the tuning results.
 
