# Day 2 — Activations, Forward Propagation & Loss

## Project Overview
This notebook continues the Phase 3 **House Price Prediction** capstone. It repeats Day 1's data loading, EDA, Sprint 1 plan, preprocessing, and baseline model, then moves into neural network fundamentals: comparing activation functions, choosing the right output layer/loss for a regression task, and manually implementing a forward pass in NumPy.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** all remaining numeric columns (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`)

## Workflow

### 1. Data Loading, EDA & Sprint 1 Recap
- Loaded `kc_house_data.csv`, dropped `id` and `date`, and re-ran the Day 1 EDA (descriptive statistics, boxplots, histograms per numeric feature, missing-value check).
- Same key observations as Day 1: `price` is right-skewed (mean ≈ $540,088, median $450,000); outliers persist in `price`, `sqft_living`, `sqft_lot`, `sqft_living15`, `sqft_lot15` and are retained as plausible real data; large scale differences across features motivate scaling before training.
- Restated the Sprint 1 goal, backlog, and acceptance criteria from Day 1 (understand the data, preprocess without leakage, establish a `DummyRegressor` baseline, then build a neural network to beat it).

### 2. Preprocessing & Baseline (recap)
- Rebuilt the `ColumnTransformer` / `Pipeline` (median imputation + `StandardScaler`) and the 60/20/20 train/validation/test split (`random_state=42`), fitting the preprocessor on the training set only.
- Retrained the `DummyRegressor` (mean strategy) baseline:

| Metric | Value |
|---|---|
| MAE | 242,663.5953 |
| RMSE | 410,080.4851 |
| R² | -0.0009 |

This remains the floor the upcoming neural network must beat.

### 3. Activation Functions
- Plotted and compared **ReLU**, **Sigmoid**, and **Tanh** over the input range [-5, 5].
- **ReLU** — outputs 0 for negative inputs, passes positive inputs unchanged; used in hidden layers.
- **Sigmoid** — squashes inputs to (0, 1); suited to binary classification outputs.
- **Tanh** — squashes inputs to (-1, 1), centered at zero.
- **Decision for this project:** ReLU in the hidden layers; no activation (`Dense(1)`) on the output layer, since the target is a continuous price.

### 4. Output Layer & Loss Function
- Because the capstone is a **regression** task, the output layer uses a single neuron with **no activation** (`Dense(1)`), allowing predictions across the full range of house prices.
- **Loss function:** Mean Squared Error (`loss="mse"`) — penalizes larger errors more heavily than smaller ones, encouraging the network to reduce large mispredictions.
- Final configuration: Task = Regression · Output layer = `Dense(1)` · Output activation = None (linear) · Loss = MSE. This is contrasted with binary classification, which would instead use a sigmoid output and binary cross-entropy.

### 5. Manual Forward Pass (NumPy)
- Implemented a 2-layer forward pass by hand (no deep-learning framework) to demonstrate the mechanics behind a neural network prediction:
  - Input: `x = [[2.0, 3.0]]`
  - Layer 1: `z1 = x @ W1 + b1`, followed by ReLU activation `a1`
  - Layer 2 (output): `z2 = a1 @ W2 + b2`, with linear (no) activation since this mirrors the regression setup
- **Results:**
  - Layer 1 pre-activation (`z1`): `[[2.3, 1.5, 2.3]]`
  - Layer 1 activation (`a1`, post-ReLU): `[[2.3, 1.5, 2.3]]` (all inputs were already positive, so ReLU had no effect)
  - Final output (`y_pred`): `[[2.92]]`
- This traces the full flow: **input → weighted sum → activation → next layer → prediction**.

## Key Findings
1. EDA and baseline results are unchanged from Day 1 — the `DummyRegressor` (MAE ≈ 242.7K, RMSE ≈ 410.1K, R² ≈ 0) remains the benchmark.
2. ReLU (hidden layers) + linear output + MSE loss is the correct activation/loss combination for this continuous-price regression task, as opposed to the sigmoid + cross-entropy setup used for classification.
3. A manual NumPy forward pass through two layers confirms the underlying mechanics before the same architecture is implemented in Keras.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — boxplots, histograms, activation function plots
- NumPy — manual forward-pass implementation
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics
- (Imported for later steps: `KMeans`, `silhouette_score`, `PCA`, `TSNE`, `IsolationForest`, `OneHotEncoder`)

## Files
- `Day2.ipynb` — full notebook: EDA recap, Sprint 1 plan recap, baseline model, activation function comparison, output/loss design decision, manual NumPy forward pass
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility
