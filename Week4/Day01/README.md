## Day 1 — Train / Validation / Test Splits

### Overview
Every model faces the same risk: if you evaluate it on the data it trained 
on, it can look excellent while having simply memorized the data rather than 
learned a real pattern. A basic train/test split addresses this, but tuning 
a model's settings introduces a second, subtler risk — if those settings are 
chosen by checking test performance, the test set is effectively "used up" 
before the final evaluation ever happens. This notebook works through that 
problem end-to-end: splitting the California housing dataset three ways, 
tuning a Random Forest's `max_depth` using only a validation set, and 
reserving the test set for one single, honest evaluation at the very end.

### What Was Done
- Loaded the housing dataset, encoded `ocean_proximity`, and filled the 207 
  missing `total_bedrooms` values with the median
- Split the data 60/20/20 into train, validation, and test sets with a fixed 
  `random_state` for reproducibility
- Scaled features using `StandardScaler`, fitting only on the training set 
  to avoid leaking test statistics into preprocessing
- Trained a Random Forest across five candidate `max_depth` values, judging 
  each one only by validation RMSE
- Evaluated the final model on the test set exactly once, after all tuning 
  decisions were already locked in
- Reasoned through what would go wrong if the test set had been used to 
  tune `max_depth` instead

### Key Result

| Metric | Score |
|---|---|
| MAE | 33,459.55 |
| RMSE | 51,339.26 |
| R² | 0.8078 |

**Best `max_depth`:** `None` (unlimited depth), selected purely from 
validation performance — error dropped sharply up to depth 15, then 
plateaued, with diminishing gains beyond that.

### Why This Matters
The core lesson isn't the specific RMSE number — it's the discipline behind 
it. Tuning against the validation set and evaluating against the test set 
only once keeps the final score honest: it reflects how the model would 
actually perform on data it's never influenced in any way. Tuning against 
the test set instead would have quietly leaked information into the model 
selection process, producing a score that looks better on paper than the 
model would actually achieve in the real world — the exact failure mode 
this three-way split is designed to prevent.

### Tools Used
Pandas • Scikit-learn • NumPy • Jupyter Notebook
