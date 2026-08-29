# Sprint 1 — House Price Prediction (Phase 3 Capstone)

## Project Overview
This repository documents Sprint 1 of the Phase 3 capstone project: building a neural network to predict house sale prices from the King County House Sales dataset. Across five days, the project moves from raw data exploration and a naive baseline, through neural network theory and training mechanics, to a fully trained, hyperparameter-tuned Keras model that substantially outperforms the baseline.

## Dataset
- **Source:** King County House Sales dataset (`kc_house_data.csv`)
- **Size:** 21,613 records, 21 columns (19 after dropping `id` and `date`)
- **Target:** `price`
- **Features:** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`

## Sprint 1 Goal
Understand the housing dataset, complete preprocessing without data leakage, establish a simple baseline regression model, and build/train/tune a neural network that beats it — with all results documented and reproducible in notebooks.

---

## Day-by-Day Summary

### Day 1 — Sprint 1 Planning & Neural Network Architecture
- Loaded and cleaned the dataset (dropped `id`, `date`); ran EDA via descriptive statistics, boxplots, and per-feature histograms.
- **Key EDA finding:** `price` is right-skewed (mean ≈ $540,088, median $450,000); outliers in `price`, `sqft_living`, `sqft_lot`, `sqft_living15`, `sqft_lot15` were retained as plausible real data; wide differences in feature scale motivate scaling.
- Wrote the Sprint 1 plan: goal, backlog (dataset understanding → preprocessing → baseline → neural network), and acceptance criteria.
- Built the preprocessing pipeline (median imputation + `StandardScaler`) and a 60/20/20 train/validation/test split (`random_state=42`).
- Trained a `DummyRegressor` (mean strategy) as the baseline: **MAE 242,663.60 · RMSE 410,080.49 · R² -0.0009**.

### Day 2 — Activations, Forward Propagation & Loss
- Compared **ReLU**, **Sigmoid**, and **Tanh** activation functions; selected ReLU for hidden layers.
- Decided the regression output layer needs **no activation** (`Dense(1)`, linear) with **MSE** loss — contrasted with the sigmoid + cross-entropy setup used for classification.
- Hand-implemented a 2-layer forward pass in NumPy (`x=[[2.0, 3.0]]` → ReLU hidden layer → linear output), producing `y_pred = [[2.92]]`, to ground the mechanics.

### Day 3 — Backpropagation, Gradient Descent & Optimizers
- Introduced the **four-step training loop**: forward pass → calculate loss → backpropagation → update weights (repeated per batch/epoch).
- Ran a **learning-rate experiment** with three small Keras models (0.00001 / 0.001 / 0.1): 0.00001 was too slow, 0.1 risked instability, **0.001** gave steady, stable convergence.
- Explained **backpropagation and the chain rule** — how gradients of the loss w.r.t. each weight are computed by working backward through the network.

### Day 4 — Building & Training a Network in Keras
- Built the main model: `Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)`, Adam optimizer, MSE loss, **3,329 trainable parameters**.
- Trained for 50 epochs; training/validation loss and MAE decreased steadily with no sign of severe overfitting.
- Built and trained a **Dropout-regularized variant** (0.3 / 0.2 dropout) for comparison — similar learning pattern, slightly higher training loss as expected.
- **Test set evaluation vs. Day 1 baseline:**

| Metric | Day 1 Baseline | Neural Network |
|---|---:|---:|
| MAE | 242,663.5953 | **120,246.7089** |
| RMSE | 410,080.4851 | **209,799.3855** |
| R² | -0.0009 | **0.7380** |

  A ~50% reduction in MAE/RMSE and an R² of 0.738 (explaining ~74% of price variance).

### Day 5 — Tuning, Evaluation & Sprint Review
- Ran a **systematic 6-run hyperparameter search** (learning rate, hidden layer sizes, dropout, batch size), varying one factor at a time:

| Run | Learning Rate | Hidden Layers | Dropout | Batch Size | Best Val Loss | Best Val MAE |
|---|---:|---|---|---:|---:|---:|
| Baseline | 0.001 | (64, 32) | (0.0, 0.0) | 32 | 35.3B | 115,209.93 |
| LR 0.0001 | 0.0001 | (64, 32) | (0.0, 0.0) | 32 | 357.8B | 494,533.91 |
| **LR 0.01** | **0.01** | (64, 32) | (0.0, 0.0) | 32 | **28.4B** | **101,635.63** |
| Larger Network | 0.001 | (128, 64) | (0.0, 0.0) | 32 | 33.2B | 110,985.96 |
| Dropout | 0.001 | (64, 32) | (0.3, 0.2) | 32 | 36.5B | 116,761.31 |
| Batch Size 16 | 0.001 | (64, 32) | (0.0, 0.0) | 16 | 33.0B | 109,857.19 |

  **Winner: learning rate = 0.01**, keeping the original 64/32 architecture, no dropout, batch size 32.
- Trained the final model with **`EarlyStopping`** (patience=5, restore best weights, up to 100 epochs): training stopped at **epoch 98**, best weights restored from **epoch 93** (validation loss 25,653,786,624).
- Compiled the full **Sprint 1 evidence package**: architecture summary, EarlyStopping behavior, training/validation MAE curves (no severe overfitting), and final test metrics confirming:
  **MAE 120,246.70 · RMSE 209,799.39 · R² 0.7380**

---

## Overall Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Day 1 Baseline (`DummyRegressor`) | 242,663.60 | 410,080.49 | -0.0009 |
| Final Neural Network | **120,246.70** | **209,799.39** | **0.7380** |

The tuned neural network cuts prediction error roughly in half compared to the naive baseline and explains ~74% of the variance in house prices on unseen test data.

## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and cleaning
- Matplotlib / Seaborn — EDA plots, activation/loss/MAE curves
- NumPy — manual forward-pass implementation
- TensorFlow / Keras — `Sequential`, `Dense`, `Dropout`, `Adam`, `EarlyStopping`, hyperparameter experiments
- Scikit-learn — `StandardScaler`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `DummyRegressor`, evaluation metrics (MAE, RMSE, R²)

## Files
- `Day1.ipynb` — data loading, EDA, Sprint 1 plan, preprocessing, baseline model
- `Day2.ipynb` — activation functions, output/loss design, manual NumPy forward pass
- `Day3.ipynb` — four-step training loop, learning-rate experiment, backpropagation/chain rule
- `Day4.ipynb` — full Keras model build & training, Dropout comparison, test-set evaluation
- `Day5.ipynb` — hyperparameter tuning sweep, final `EarlyStopping` model, Sprint 1 evidence writeup
- `kc_house_data.csv` — King County house sales dataset
- `requirements.txt` — exact library versions for reproducibility

## Next Steps
- Further hyperparameter tuning around the winning learning rate (e.g. finer LR search, learning rate schedules).
- Feature engineering (e.g. house age, renovation flag, geographic clustering from `lat`/`long`).
- Deployment: wrap the final model in a Streamlit or FastAPI app with a public URL, per the Phase 3 Definition of Done.
