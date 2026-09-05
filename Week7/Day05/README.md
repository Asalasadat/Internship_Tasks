# Day 5 — Advancing the Core Model & Sprint Review

## Project Overview
This notebook is the **Sprint 2 close-out** of the House Price Prediction capstone. It re-establishes the full pipeline (EDA, preprocessing, Day 1 baseline, and the Sprint 1 neural network), then runs three systematic tuning experiments (learning rate, batch size, dropout) to try to beat the Sprint 1 model, and closes with a full Sprint Review and Retrospective.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** all remaining numeric columns (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`)

## Workflow

### 1. Core Architecture Selection
Since the project uses tabular data to predict a continuous target (`Selling_Price`/`price`), a **Dense Neural Network** was confirmed as the core architecture. The Sprint 1 model's results were restated as the benchmark to beat:
- MAE: 120,246.71 · RMSE: 209,799.39 · R²: 0.7380

### 2. Recap: EDA, Preprocessing & Baseline
- Re-ran the standard EDA (descriptive statistics, boxplots, per-feature histograms, missing-value check) — same findings as prior days: `price` is right-skewed, outliers retained in `price`/`sqft_living`/`sqft_lot`/`sqft_living15`/`sqft_lot15`.
- Rebuilt the preprocessing pipeline (median imputation + `StandardScaler`) and a 60/20/20 train/validation/test split.
- Retrained the `DummyRegressor` baseline: **MAE 242,663.60 · RMSE 410,080.49 · R² -0.0009**.
- Recapped activation functions (ReLU/Sigmoid/Tanh), the linear-output + MSE-loss regression setup, the manual NumPy forward pass, the four-step training loop, the learning-rate experiment (0.001 confirmed best), and backpropagation/chain rule.

### 3. Sprint 1 Neural Network (recap)
Rebuilt and retrained `Dense(64,ReLU)→Dense(32,ReLU)→Dense(1)` (3,329 params) for 50 epochs, and its Dropout-regularized variant, reproducing the Sprint 1 test results:
**MAE 120,246.71 · RMSE 209,799.39 · R² 0.7380**

### 4. Sprint 2 Experiment Log
Three systematic one-variable-at-a-time experiments were run against the Sprint 1 model:

**Experiment 1 — Learning Rate (0.001 → 0.0005)**

| Model | Learning Rate | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Baseline NN | 0.001 | **120,246.71** | **209,799.39** | **0.7380** |
| Experiment 1 | 0.0005 | 160,547.19 | 242,658.57 | 0.6495 |

→ 0.0005 was too conservative (slower learning); **0.001 remains better**.

**Experiment 2 — Batch Size (16 / 32 / 64)**

| Batch Size | Test MAE | Test RMSE | Test R² |
|---:|---:|---:|---:|
| **16** | **114,687.92** | **203,523.72** | **0.7535** |
| 32 | 117,511.51 | 206,624.95 | 0.7459 |
| 64 | 147,024.55 | 230,387.33 | 0.6841 |

→ **batch_size=16 won**, improving on the Sprint 1 model across all three metrics.

**Experiment 3 — Dropout Regularization (0.3 / 0.2, on top of batch_size=16)**

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Previous Best (batch=16, no dropout) | **114,687.92** | **203,523.72** | **0.7535** |
| Dropout Model | 115,122.50 | 207,468.61 | 0.7438 |

→ Dropout **did not help** here; the non-regularized batch=16 model stayed the best.

### 5. Final Sprint 2 Model
```text
Architecture: Dense(64,ReLU) → Dense(32,ReLU) → Dense(1,Linear)
Optimizer: Adam · Learning rate: 0.001 · Batch size: 16 · Epochs: 50 · Dropout: None
```
Training/validation loss and MAE curves for this configuration showed steady improvement with no clear sign of severe overfitting.

## Final Model Comparison

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Week 6 Baseline (`DummyRegressor`) | 242,663.60 | 410,080.49 | -0.0009 |
| Sprint 1 Neural Network | 120,246.71 | 209,799.39 | 0.7380 |
| **Sprint 2 Best Model** | **114,687.92** | **203,523.72** | **0.7535** |

- **vs. Week 6 baseline:** ~52.7% lower MAE, ~50.4% lower RMSE
- **vs. Sprint 1 NN:** MAE improved from 120,246.71 → 114,687.92; RMSE from 209,799.39 → 203,523.72; R² from 0.7380 → 0.7535

## Sprint 2 Review & Retrospective

**Sprint Goal:** Improve the Sprint 1 core model through systematic tuning experiments.

**What went well:**
- The Dense Neural Network architecture remained appropriate for this tabular regression task.
- Experiments were run one variable at a time (learning rate → batch size → dropout), isolating each effect cleanly.
- The batch-size experiment produced a real, measurable improvement over Sprint 1.
- Training/validation curves were checked at each step to confirm healthy learning behavior.

**What could be improved:**
- Explore more hyperparameter combinations (network sizes, optimizer configurations).
- Rely on validation performance for model selection, reserving the test set strictly for final evaluation.
- Adopt a dedicated experiment-tracking tool (e.g., MLflow) instead of manual logging.

**Sprint 3 focus:** Improve generalization and reliability further, using validation performance (not test performance) to drive model-selection decisions, while keeping the test set for the final evaluation only.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — EDA plots, activation/loss/MAE curves
- NumPy — manual forward-pass implementation
- TensorFlow / Keras — `Sequential`, `Dense`, `Dropout`, `Adam`, systematic hyperparameter experiments
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics (MAE, RMSE, R²)

## Files
- `Day05.ipynb` — full notebook: core architecture recap, EDA/baseline/Sprint 1 NN recap, 3 Sprint 2 tuning experiments, final model comparison, Sprint Review & Retrospective
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility