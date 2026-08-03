# Day 2 — Linear Regression

## Overview
This notebook builds and evaluates a **Linear Regression** model to predict median house values using the [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) dataset.

## Tasks

### Task 1: Train a Linear Regression Model
- Loaded the housing dataset (`Housing Prices/housing.csv`) into a Pandas DataFrame.
- Checked for missing values — found 207 missing entries in `total_bedrooms`.
- Filled missing values in `total_bedrooms` using the **median** (chosen over the mean because it is robust to outliers and better represents the typical value).
- Dropped the categorical column `ocean_proximity` to keep the model strictly numeric.
- Split the data into training (80%) and test (20%) sets using `train_test_split` with `random_state=42`.
- Trained a `LinearRegression` model from scikit-learn.

**Result:**
- Train R²: `0.640`
- Test R²: `0.614`

### Task 2: Model Coefficients
Reported each feature's coefficient and identified the feature with the strongest effect on the predicted target.

| Feature | Coefficient |
|---|---|
| longitude | -42,632.39 |
| latitude | -42,450.07 |
| median_income | 40,538.40 |
| housing_median_age | 1,182.81 |
| total_bedrooms | 116.26 |
| households | 46.34 |
| population | -38.49 |
| total_rooms | -8.19 |

- **Intercept:** -3,578,224.23
- **Strongest feature (by absolute magnitude):** `longitude` (-42,632.39)

> ⚠️ Note: Coefficient magnitudes should be interpreted with caution since the features are on different scales. Standardizing the features would allow a more meaningful comparison of their relative importance.

### Task 3: Model Evaluation
Evaluated the trained model on the test set using three metrics:

| Metric | Value |
|---|---|
| MAE | 51,810.48 |
| RMSE | 71,133.17 |
| R² | 0.614 |

**Interpretation:** The model explains about **61.4%** of the variance in house prices. On average, predictions are off by roughly $51,810 (MAE), with RMSE indicating that some predictions carry noticeably larger errors.

### Task 4: Baseline Comparison
Compared the model against a naive baseline that predicts the training mean for every test sample.

| Model | RMSE |
|---|---|
| Linear Regression | 71,133.17 |
| Baseline (mean prediction) | 114,485.64 |

**Conclusion:** The linear regression model has a substantially lower RMSE than the baseline, confirming that it **adds real predictive value** over simply guessing the average house price.

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn (`LinearRegression`, `train_test_split`, evaluation metrics)
- Jupyter Notebook
