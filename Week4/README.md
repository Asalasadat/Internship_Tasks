# 📚 Week 4 — Model Validation, Diagnosis & Tuning

## Overview
A model that scores well on one run of data isn't automatically a model you can trust. Week 4 is built around a single uncomfortable question that every earlier week left unanswered: **how do you know if a good score is real, or just luck?** Each day chips away at a different version of that question — whether tuning decisions secretly leaked information, whether a model memorized instead of learned, whether one train/test split was a fluke, and whether all the hyperparameter searching actually earned its keep. By the end of the week, the answer isn't a better score — it's a repeatable process for trusting the score you already have.

---

## Day 1 — Train / Validation / Test Splits

Every model faces the same risk: if you evaluate it on the data it trained on, it can look excellent while having simply memorized the data rather than learned a real pattern. A basic train/test split addresses that — but tuning a model's settings introduces a second, subtler risk: if those settings are chosen by checking test performance, the test set is effectively "used up" before the final evaluation ever happens. This day works through that problem end-to-end on the California housing dataset — splitting 60/20/20, tuning a Random Forest's `max_depth` using only the validation set, and reserving the test set for one single, honest evaluation at the very end.

**Result:** MAE = 33,459.55, RMSE = 51,339.26, R² = 0.8078, with `max_depth=None` selected purely from validation performance.

**The takeaway:** tuning against validation and evaluating against test only once keeps the final score honest — it reflects real performance, not a number quietly shaped by data that was supposed to stay unseen.

---

## Day 2 — Cross-Validation

A single train/test split raises its own question: what if that one split just happened to be easy, or hard? Day 2 answers this by running 5-fold cross-validation on the same Random Forest and checking whether the single-split score was trustworthy or a coincidence.

**Result:** Mean 5-fold CV RMSE = 51,017.10 (SD = 2,000.27, ~4% of the mean) — nearly identical to Day 1's single-split RMSE of 51,339.26, a difference of under 1%.

**The takeaway:** the low spread across folds and the close agreement with the single split confirms Day 1's result wasn't a lucky draw — it reflects genuinely stable model performance. (StratifiedKFold was ruled out since the target is continuous, not categorical.)

---

## Day 3 — Bias-Variance & Diagnosing Model Fit

A single accuracy number doesn't say *why* a model is underperforming. Day 3 deliberately builds both failure modes to make the diagnosis visible: a tree too simple to learn anything (`max_depth=1`), and a tree so complex it memorizes the training set outright (`max_depth=None`). Comparing train vs. validation error for each reveals which problem is which — and regularization is then applied to fix the overfit case.

| Regime | Train RMSE | Validation RMSE | Gap | Diagnosis |
|---|---|---|---|---|
| Underfit (depth=1) | 79,057.22 | 77,118.37 | 1,938.85 | Too simple |
| Overfit (depth=None) | 0.00 | 62,153.07 | 62,153.07 | Memorized training data |
| Regularized (depth=10) | 45,580.26 | 50,954.80 | 5,374.54 | Best generalization |

**The takeaway:** a small gap with high error on both sides means "too simple" (add complexity); a large gap with near-perfect training performance means "memorized" (add constraints). Regularization cut the gap by 91% here — proof that less complexity, not more, was the right fix.

---

## Day 4 — Feature Engineering & Hyperparameter Tuning

With diagnosis tools in hand, Day 4 asks a more ambitious question: can engineered features and a proper hyperparameter search meaningfully beat a default model? Three ratio features were added (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`), and a `RandomForestRegressor` was searched across 324 combinations — first via `GridSearchCV` (too slow, abandoned), then `RandomizedSearchCV` (30 sampled combinations).

**Result:** Tuned CV RMSE = 61,895.55 vs. untuned baseline RMSE = 62,180.64 — a 0.46% improvement, within normal cross-validation noise. `median_income` dominated feature importance (~52%), and the engineered `population_per_household` outranked several raw features (~14%), confirming it added real signal even though the broader search didn't.

**The takeaway:** feature engineering can matter more than exhaustive tuning — the defaults were already close to optimal here, and the real gain came from a smarter feature, not a better hyperparameter combination.

---

## Day 5 — Pipelines & End-to-End Hyperparameter Search

Manually re-fitting scalers and encoders on each fold is easy to get wrong — Day 5 closes the week by wrapping the entire preprocessing + modeling flow into a single `sklearn` `Pipeline`, so imputation, encoding, and scaling are refit correctly on every fold's training data automatically, closing a data-leakage bug from earlier attempts. Preprocessing choices (imputation strategy) and model hyperparameters were then tuned jointly via `GridSearchCV`, and the tuned pipeline was compared against an untuned baseline on a held-out test set.

| Metric | Baseline | Tuned Pipeline |
|---|---|---|
| RMSE ($) | 60,433.32 | 60,078.13 |
| MAE ($) | 41,861.78 | 41,928.54 |
| R² | 0.7359 | 0.7390 |

**The takeaway:** a 0.59% RMSE improvement — and MAE actually got slightly worse — confirms the untuned baseline was already near this ceiling. But the real value of the day wasn't the score: it was building a pipeline architecture that makes leakage structurally impossible, regardless of how much tuning happens on top of it.

---

## 🔑 What the Week Adds Up To

Every day answered a different version of "can I trust this number?" — clean splits (Day 1), repeated splits (Day 2), diagnosing *why* a score is bad (Day 3), and testing whether more effort (features, tuning) actually pays off (Days 4-5). The recurring pattern across Days 4 and 5 — tuning producing under 1% improvement both times — is itself a lesson: a well-validated default model is often close to the ceiling, and knowing *that* is as valuable as the model itself.

## 🛠️ Tools Used
Pandas • NumPy • Matplotlib • Scikit-learn • Jupyter Notebook
