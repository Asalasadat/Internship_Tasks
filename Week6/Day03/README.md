# Day 3 — Backpropagation, Gradient Descent & Optimizers

## Project Overview
This notebook continues the Phase 3 **House Price Prediction** capstone. It repeats the Day 1/2 data loading, EDA, Sprint 1 plan, preprocessing, and baseline model, then moves deeper into neural network training mechanics: the four-step training loop, a learning-rate experiment with real Keras models, and the theory behind backpropagation and the chain rule.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** all remaining numeric columns (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`)

## Workflow

### 1. Data Loading, EDA & Sprint 1 Recap
- Loaded `kc_house_data.csv`, dropped `id` and `date`, and re-ran the standard EDA (descriptive statistics, boxplots, per-feature histograms, missing-value check).
- Same findings as prior days: `price` is right-skewed (mean ≈ $540,088, median $450,000); outliers persist in `price`, `sqft_living`, `sqft_lot`, `sqft_living15`, `sqft_lot15` and are retained as plausible real data; large scale differences across features motivate feature scaling.
- Restated the Sprint 1 goal, backlog, and acceptance criteria (understand the data, preprocess without leakage, establish a `DummyRegressor` baseline, then build a neural network to beat it).

### 2. Preprocessing & Baseline (recap)
- Rebuilt the `ColumnTransformer` / `Pipeline` (median imputation + `StandardScaler`) and the 60/20/20 train/validation/test split (`random_state=42`), fitting the preprocessor on the training set only.
- Retrained the `DummyRegressor` (mean strategy) baseline:

| Metric | Value |
|---|---|
| MAE | 242,663.5953 |
| RMSE | 410,080.4851 |
| R² | -0.0009 |

### 3. Activation Functions & Output/Loss Design (recap)
- Compared **ReLU**, **Sigmoid**, and **Tanh** — ReLU for hidden layers, no activation (`Dense(1)`) on the regression output.
- Confirmed the regression configuration: linear output layer + **MSE loss** (`loss="mse"`), as opposed to sigmoid + binary cross-entropy for classification.

### 4. Manual Forward Pass (recap)
- Re-ran the 2-layer NumPy forward pass (`x=[[2.0, 3.0]]` → ReLU hidden layer → linear output), producing `y_pred = [[2.92]]`, to ground the mechanics before training a real model.

### 5. The Four-Step Training Loop
Outlined the training cycle every neural network follows each batch/epoch:
1. **Forward Pass** — Input → Network → Prediction
2. **Calculate Loss** — Prediction vs. Actual → Loss
3. **Backpropagation** — Loss → Gradients
4. **Update Weights** — Gradients + Adam → Updated Weights

...repeated for each subsequent batch and epoch.

### 6. Learning Rate Experiment
- Trained three identical small Keras networks (`Dense(16, relu)` → `Dense(1)`, Adam optimizer, MSE loss, 30 epochs, batch size 32) — varying only the **learning rate**:
  - **0.00001 (Too Low)**
  - **0.001 (Good)**
  - **0.1 (Too High)**
- Plotted training/validation loss curves for all three.
- **Findings:**
  - **0.00001:** loss decreases very slowly — weight updates too small for efficient learning.
  - **0.001:** loss decreases steadily — good balance of speed and stability.
  - **0.1:** loss drops quickly in the first few epochs, then plateaus; no severe instability was observed here, but the rate is large and may be less reliable on different data/initializations.
- **Conclusion:** a learning rate of **0.001** is the reasonable choice for this project, balancing convergence speed and stability.

### 7. Backpropagation & the Chain Rule
- Explained how backpropagation computes the gradient of the loss with respect to each weight/bias by working backward through the network after the forward pass and loss calculation.
- Because each layer's output feeds the next layer's input, the chain rule links these gradients across layers:

  `dLoss/dWeight = dLoss/dOutput × dOutput/dHidden × dHidden/dWeight`

- These gradients are what an optimizer (e.g., Adam) uses to update the weights and reduce the loss.

## Key Findings
1. EDA and baseline results are unchanged from prior days — the `DummyRegressor` (MAE ≈ 242.7K, RMSE ≈ 410.1K, R² ≈ 0) remains the benchmark to beat.
2. A learning rate of **0.001** gives stable, steady convergence for this dataset/architecture, while 0.00001 is too slow and 0.1 risks instability.
3. The four-step training loop (forward pass → loss → backpropagation → weight update) and the chain rule underpin how the network will actually learn to predict house prices in later steps.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — boxplots, histograms, activation and loss-curve plots
- NumPy — manual forward-pass implementation
- TensorFlow / Keras — Sequential model, `Adam` optimizer, learning-rate experiment
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics
- (Imported for later steps: `KMeans`, `silhouette_score`, `PCA`, `TSNE`, `IsolationForest`, `OneHotEncoder`)

## Files
- `Day3.ipynb` — full notebook: EDA/baseline recap, activation & loss recap, four-step training loop, learning-rate experiment (Keras), backpropagation/chain-rule explanation
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility
