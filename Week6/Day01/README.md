# Day 1 — Sprint 1 Planning & Neural Network Architecture

## Project Overview
This notebook kicks off the Phase 3 capstone (**House Price Prediction**) by exploring the King County house sales dataset, planning Sprint 1, and establishing a simple baseline regression model to beat with an upcoming neural network.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15` (all numeric — `id` and `date` were dropped as non-predictive)

## Workflow

### 1. Data Loading & Cleanup
- Loaded `kc_house_data.csv` and dropped the `id` and `date` columns, which carry no predictive value for price.
- Selected all remaining numeric columns for analysis.

### 2. Exploratory Data Analysis
- Computed descriptive statistics (`df.describe()`) and inspected each numeric feature via boxplots and individual histograms.
- **Target distribution:** `price` has a mean of **~$540,088** and a median of **$450,000** — the gap between mean and median indicates a right-skewed distribution.
- **Feature ranges:**
  - `bedrooms`: mean 3.37, range 0–33
  - `bathrooms`: mean 2.11, range 0–8
  - `sqft_living`: mean ~2,080 sq ft, max 13,540 sq ft
  - `sqft_lot`: mean ~15,107 sq ft, max 1,651,359 sq ft (highly variable)
  - `yr_built`: 1900–2015; `yr_renovated` contains many zeros (never renovated)
  - `sqft_living15` and `sqft_lot15` also show substantial variation
- **Outliers:** Boxplots flagged likely outliers in `price`, `sqft_living`, `sqft_lot`, `sqft_living15`, and `sqft_lot15`. These were kept rather than removed, since they may represent genuinely large or expensive properties rather than data errors.
- Checked for missing values with `df.isnull().sum()`.
- The large differences in feature scale motivate standardizing features before training the neural network in a later step.

### 3. Sprint 1 Planning
**Sprint Goal:** Understand the dataset, complete initial preprocessing, and establish a baseline regression model to compare against the neural network.

**Backlog:**
1. Dataset selection and understanding (load, inspect features, check missing values/outliers)
2. Data preprocessing (handle missing values, encode categoricals, scale numerics, train/val/test split)
3. Baseline model (`DummyRegressor`, evaluated with MAE/RMSE/R²)
4. Neural network baseline (Keras regression model, trained and compared to the baseline)

**Acceptance Criteria:**
- Dataset loaded and inspected successfully
- Preprocessing completed without data leakage
- Baseline regression model trained and evaluated
- MAE, RMSE, and R² recorded
- Neural network trained and evaluated
- All results documented in the notebook

### 4. Preprocessing & Train/Val/Test Split
- Separated features (`X`) from the target (`y = price`).
- Built a `ColumnTransformer` / `Pipeline` for the numeric columns: median imputation (`SimpleImputer`) followed by `StandardScaler`.
- Split the data **60/20/20** into train / validation / test sets (`train_test_split`, `random_state=42`), fitting the preprocessor only on the training set to avoid leakage.
- Resulting shapes: **Train (12,967, 18)**, **Validation (4,323, 18)**, **Test (4,323, 18)**.

### 5. Baseline Model
- Trained a `DummyRegressor` (strategy = `mean`) on the processed training data as the reference model.

| Metric | Value |
|---|---|
| MAE | 242,663.5953 |
| RMSE | 410,080.4851 |
| R² | -0.0009 |

The baseline simply predicts the mean `price` for every observation, so it serves only as a floor to beat — the near-zero (slightly negative) R² is expected for a `DummyRegressor`. The planned neural network should achieve lower MAE/RMSE and a meaningfully positive R².

## Key Findings
1. `price` is right-skewed, and several features (`sqft_lot`, `sqft_living`, `price` itself) contain outlier values retained as plausible real data.
2. Feature scales vary enormously (e.g. `sqft_lot` in the thousands vs. `waterfront` as 0/1), making scaling essential before neural network training.
3. The `DummyRegressor` baseline (MAE ≈ 242.7K, RMSE ≈ 410.1K, R² ≈ 0) establishes the minimum bar the upcoming Keras neural network must clear.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading, cleaning, descriptive statistics
- Matplotlib / Seaborn — boxplots and histograms
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics
- (Imported for later steps in the sprint: `KMeans`, `silhouette_score`, `PCA`, `TSNE`, `IsolationForest`, `OneHotEncoder`)

## Files
- `Day1.ipynb` — full notebook: data loading, EDA, Sprint 1 plan, preprocessing pipeline, train/val/test split, and baseline model
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility
