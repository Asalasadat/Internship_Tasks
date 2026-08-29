# Day 4 — Building & Training a Network in Keras

## Project Overview
This notebook completes the core modeling work of the Phase 3 **House Price Prediction** capstone. It repeats the Day 1–3 data prep, EDA, and neural-network fundamentals, then builds, trains, and evaluates a full Keras regression model — including a Dropout-regularized variant — and compares the final results against the Day 1 `DummyRegressor` baseline on the held-out test set.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** all remaining numeric columns (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`)

## Workflow

### 1. Data Loading, EDA, Sprint 1 & Baseline (recap)
- Loaded `kc_house_data.csv`, dropped `id`/`date`, and re-ran the standard EDA (descriptive stats, boxplots, histograms, missing-value check) — same findings as prior days: `price` is right-skewed, outliers retained in `price`/`sqft_living`/`sqft_lot`/`sqft_living15`/`sqft_lot15`, large scale differences motivate feature scaling.
- Restated the Sprint 1 plan (goal, backlog, acceptance criteria).
- Rebuilt the preprocessing pipeline (median imputation + `StandardScaler`) and the 60/20/20 train/validation/test split (`random_state=42`).
- Retrained the `DummyRegressor` baseline: **MAE 242,663.60 · RMSE 410,080.49 · R² -0.0009**.

### 2. Neural Network Fundamentals (recap)
- Compared **ReLU / Sigmoid / Tanh** activations; confirmed ReLU for hidden layers and a linear (no-activation) `Dense(1)` output for this regression task, with **MSE** as the loss.
- Re-ran the manual NumPy forward pass and the four-step training loop (forward pass → loss → backpropagation → weight update).
- Repeated the learning-rate experiment (0.00001 / 0.001 / 0.1), confirming **0.001** as the stable choice, and reviewed backpropagation and the chain rule.

### 3. Neural Network Architecture
Built the main Keras model:

```text
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
```
- **Optimizer:** Adam · **Loss:** MSE · **Metric:** MAE
- **Trainable parameters:** 3,329 (no non-trainable parameters)
- No activation on the output layer, since regression needs an unrestricted continuous output.

### 4. Training the Model
- Trained for **50 epochs**, batch size 32, on `X_train_processed` / `y_train` with `X_val_processed` / `y_val` for validation.
- **Training loss:** ~413.0B → ~32.1B (MSE) · **Validation loss:** ~420.3B → ~35.6B
- **Training MAE:** ~536,846 → ~114,116 · **Validation MAE:** ~538,549 → ~115,931
- Loss and MAE curves for train vs. validation were plotted; the two tracked closely with only a small, expected gap (validation slightly higher than training).
- **Conclusion:** no clear sign of severe overfitting over 50 epochs — validation loss kept decreasing rather than increasing — though `EarlyStopping` was noted as a natural next step to auto-select the best epoch.

### 5. Dropout-Regularized Model
Built and trained a second model with Dropout regularization:

```text
Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1)
```
- Same optimizer (Adam), loss (MSE), 50 epochs, batch size 32.
- **Validation loss:** ~37.56B by epoch 50 · **Validation MAE:** ~120,164
- Compared training/validation loss curves of the original vs. Dropout model: the regularized model has slightly higher training loss (expected, since Dropout reduces effective capacity during training), but its validation loss stays close to its training loss — reasonable generalization with no clear evidence of severe overfitting in either model.

### 6. Test Set Evaluation: Neural Network vs. Day 1 Baseline
Evaluated the original (non-Dropout) neural network on the held-out test set and compared it to the Day 1 baseline:

| Metric | Day 1 Baseline | Neural Network |
|---|---:|---:|
| MAE | 242,663.5953 | **120,246.7089** |
| RMSE | 410,080.4851 | **209,799.3855** |
| R² | -0.0009 | **0.7380** |

- **MAE** improved ~50.4% (242,663.60 → 120,246.71)
- **RMSE** improved ~48.8% (410,080.49 → 209,799.39)
- **R²** rose from essentially 0 to **0.7380** — the neural network explains ~73.8% of the variance in test-set house prices.

## Key Findings
1. The `Dense(64,ReLU) → Dense(32,ReLU) → Dense(1)` network trained with Adam/MSE substantially outperforms the Day 1 `DummyRegressor` baseline on every metric (MAE, RMSE, R²).
2. Both the original and Dropout-regularized models show healthy, non-overfitting training behavior across 50 epochs, with validation metrics tracking training metrics closely.
3. Dropout (0.3 / 0.2) trades a bit of training performance for regularization but does not clearly outperform the original model in this run — final model choice should rest on head-to-head test-set MAE/RMSE/R², which is the natural next step.
4. A learning rate of 0.001 remains the stable choice; 0.00001 is too slow and 0.1 risks instability.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — boxplots, histograms, activation/loss/MAE curve plots
- NumPy — manual forward-pass implementation
- TensorFlow / Keras — `Sequential`, `Dense`, `Dropout`, `Adam` optimizer, model training and evaluation
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics (MAE, RMSE, R²)
- (Imported for later steps: `KMeans`, `silhouette_score`, `PCA`, `TSNE`, `IsolationForest`, `OneHotEncoder`)

## Files
- `Day4.ipynb` — full notebook: EDA/baseline/fundamentals recap, main Keras model build & training, Dropout-regularized model, original-vs-Dropout comparison, final test-set evaluation vs. Day 1 baseline
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility
