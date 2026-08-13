# Day 2 — Cross-Validation

Evaluating a Random Forest Regressor's reliability on the California 
housing dataset by comparing a single train/test split against 5-fold 
cross-validation.

## Dataset

`Housing Prices/housing.csv` — the California Housing dataset, with 
`median_house_value` as the regression target.

## Workflow

1. **Load & clean**
   - Loaded the CSV.
   - Label-encoded the categorical `ocean_proximity` column.
   - Filled 207 missing values in `total_bedrooms` with the column 
     median.
2. **Split the data** — three-way split via two calls to 
   `train_test_split`:
   - 60% train
   - 20% validation
   - 20% test
3. **Scale features** — `StandardScaler` fit on the training set only, 
   then applied to validation and test sets to avoid data leakage.
4. **5-Fold Cross-Validation** — trained a `RandomForestRegressor` 
   (`n_estimators=100`, default depth) and scored it with 
   `cross_val_score` (`neg_root_mean_squared_error`, 5 folds), converting 
   the negative scores back to positive RMSE.
5. **Compare against a single split** — checked whether the single 
   train/test RMSE agreed with the more robust cross-validated estimate.
6. **Discuss Stratified K-Fold** — explained why `StratifiedKFold` 
   doesn't apply to this regression task.

## Results

**Per-fold RMSE:** `[50013.90, 53111.04, 47589.86, 52448.82, 51921.87]`

| Metric | Value |
|---|---|
| Mean RMSE (5-fold CV) | 51,017.10 |
| Standard Deviation | 2,000.27 (~4% of the mean) |

The low standard deviation across folds indicates stable performance, 
not one lucky data split.

| Evaluation Method | RMSE |
|---|---|
| Single train/test split | 51,339.26 |
| 5-Fold Cross-Validation (Mean) | 51,017.10 |

The two estimates differ by only ~322 (under 1%), both within the CV 
standard deviation — confirming the single-split result wasn't an 
outlier and closely matches the more reliable cross-validated score.

**Why not Stratified K-Fold?** `StratifiedKFold` preserves class ratios 
across folds and only applies to classification. Since 
`median_house_value` is continuous, standard `KFold` (the `cross_val_score` 
default) is used instead — there's no class balance to preserve, only 
numeric variance, which RMSE already accounts for.

## Tools

- pandas
- numpy
- scikit-learn

## Usage

Open `Day2.ipynb` in Jupyter and run the cells top to bottom. Update the 
`pd.read_csv(...)` path if `housing.csv` is stored elsewhere.
