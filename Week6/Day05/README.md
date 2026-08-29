# Day 5 — Tuning, Evaluation & Sprint Review

## Project Overview
This notebook closes out Sprint 1 of the Phase 3 **House Price Prediction** capstone. It repeats the Day 1–4 data prep, EDA, and neural-network fundamentals, then runs a systematic hyperparameter search, trains a final "best" model with `EarlyStopping`, and compiles the Sprint 1 evidence package: baseline-vs-network comparison, final architecture, training curves, and final test metrics.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** all remaining numeric columns (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`)

## Workflow

### 1. Data Loading, EDA, Sprint 1 Plan & Baseline (recap)
- Loaded `kc_house_data.csv`, dropped `id`/`date`, re-ran the standard EDA (descriptive stats, boxplots, histograms, missing-value check) — same findings as prior days.
- Restated the Sprint 1 plan (goal, backlog, acceptance criteria).
- Rebuilt the preprocessing pipeline (median imputation + `StandardScaler`) and the 60/20/20 train/validation/test split (`random_state=42`).
- Retrained the `DummyRegressor` baseline: **MAE 242,663.60 · RMSE 410,080.49 · R² -0.0009**.

### 2. Neural Network Fundamentals & First Models (recap)
- Compared ReLU/Sigmoid/Tanh activations, confirmed the linear-output + MSE-loss regression setup, re-ran the manual NumPy forward pass, reviewed the four-step training loop and backpropagation/chain rule.
- Rebuilt and retrained the original `Dense(64,ReLU)→Dense(32,ReLU)→Dense(1)` network (3,329 params) and the Dropout-regularized variant, reproducing the Day 4 training curves and metrics.
- Re-confirmed on the test set: **Neural Network — MAE 120,246.71 · RMSE 209,799.39 · R² 0.7380** vs. the Day 1 baseline.

### 3. Step 1: Systematic Hyperparameter Tuning
Built a reusable `run_experiment()` function (learning rate, hidden layer sizes, dropout rates, batch size all configurable) and ran **six experiments**, changing one hyperparameter at a time from the baseline:

| Run | Configuration | Learning Rate | Hidden Layers | Dropout | Batch Size | Best Val Loss | Best Val MAE |
|---|---|---:|---|---|---:|---:|---:|
| 0 | Baseline | 0.001 | (64, 32) | (0.0, 0.0) | 32 | 35,265,970,000 | 115,209.93 |
| 1 | LR 0.0001 | 0.0001 | (64, 32) | (0.0, 0.0) | 32 | 357,774,300,000 | 494,533.91 |
| 2 | LR 0.01 | 0.01 | (64, 32) | (0.0, 0.0) | 32 | **28,383,160,000** | **101,635.63** |
| 3 | Larger Network | 0.001 | (128, 64) | (0.0, 0.0) | 32 | 33,188,510,000 | 110,985.96 |
| 4 | Dropout | 0.001 | (64, 32) | (0.3, 0.2) | 32 | 36,545,210,000 | 116,761.31 |
| 5 | Batch Size 16 | 0.001 | (64, 32) | (0.0, 0.0) | 16 | 32,971,650,000 | 109,857.19 |

**Conclusion:** Run 2 (**learning rate = 0.01**) achieved the best validation performance — lowest validation loss (2.84×10¹⁰) and lowest validation MAE (101,635.63). The lower learning rate (0.0001) was far too slow; the larger network and dropout gave modest gains over baseline but neither beat the higher learning rate. **Learning rate 0.01** was selected for the final model.

### 4. Final Model with EarlyStopping
Built the "best" model using the winning hyperparameters:

```text
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
Optimizer: Adam(learning_rate=0.01) · Loss: MSE · Metric: MAE
```
- Trained for up to 100 epochs with `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
- **Training stopped at epoch 98**; **best epoch was 93** with a **validation loss of 25,653,786,624**.
- Since validation loss did not improve for 5 consecutive epochs after epoch 93, training halted automatically, and the final model's weights were restored to those from epoch 93 — confirming `EarlyStopping` worked as intended and avoided unnecessary extra training.
- Training vs. validation MAE curves showed a large improvement in the first few epochs, then a gradual decrease and stabilization, with the two curves staying close together (validation MAE finishing ≈100K, training slightly lower) — indicating effective learning with no severe overfitting.

### 5. Sprint 1 Evidence Summary
Compiled the full evidence package for Sprint 1 sign-off:

**Baseline vs. Neural Network**

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Day 1 Baseline | 242,663.5953 | 410,080.4851 | -0.0009 |
| Neural Network | **120,246.7089** | **209,799.3855** | **0.7380** |

**Architecture:** Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, linear) · Adam · MSE · 3,329 trainable parameters

**EarlyStopping:** stopped at epoch 98, best epoch 93, best validation loss 25,653,786,624

**Final test-set evaluation** (re-run via `model.evaluate` / `model.predict`):
- **Test MAE:** 120,246.70
- **Test RMSE:** 209,799.39
- **Test R²:** 0.7380

## Key Findings
1. Of six tuning experiments, **learning rate = 0.01** (with the original 64/32 architecture, no dropout, batch size 32) gave the best validation loss and MAE — outperforming a larger network, dropout regularization, and a smaller batch size.
2. `EarlyStopping` (patience=5, restore best weights) correctly identified epoch 93 as the best checkpoint and prevented 5 epochs of unnecessary further training.
3. The final neural network cuts MAE and RMSE by roughly half versus the Day 1 `DummyRegressor` baseline and explains ~73.8% of the variance in house price (R² = 0.7380) — a substantial, well-evidenced improvement that closes out Sprint 1.
4. Training and validation MAE curves stay close throughout, with no sign of severe overfitting, supporting the model's readiness for further tuning or deployment in the next sprint.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — boxplots, histograms, loss/MAE curve plots
- NumPy — manual forward-pass implementation
- TensorFlow / Keras — `Sequential`, `Dense`, `Dropout`, `Adam`, `EarlyStopping`, systematic experiment runner
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics (MAE, RMSE, R²)
- (Imported for later steps: `KMeans`, `silhouette_score`, `PCA`, `TSNE`, `IsolationForest`, `OneHotEncoder`)

## Files
- `Day5.ipynb` — full notebook: EDA/baseline/fundamentals recap, six-run hyperparameter search, final `EarlyStopping` model, and the complete Sprint 1 evidence writeup
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility
