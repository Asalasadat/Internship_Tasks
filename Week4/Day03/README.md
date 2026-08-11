## Day 3 — Bias-Variance & Diagnosing Model Fit

### Overview
A model's accuracy number alone doesn't tell you whether it's actually 
learning or just memorizing — and it definitely doesn't tell you how to fix 
it if something's wrong. This notebook works through both failure modes on 
the California housing dataset: deliberately building a model too simple to 
learn anything (underfitting), then one so complex it memorizes the 
training set outright (overfitting), comparing both against the training 
vs. validation gap to diagnose which is happening — and finally applying 
regularization to fix the overfit model and confirming the gap actually 
shrinks.

### What Was Done
- Split the housing data into train/validation/test sets and scaled 
  features (fit on train only)
- Trained an extremely simple decision tree (`max_depth=1`) to confirm the 
  underfitting signature: high error on both train and validation, with a 
  near-zero gap between them
- Trained an unconstrained decision tree (`max_depth=None`) to confirm the 
  overfitting signature: near-zero train error against a much higher 
  validation error
- Plotted train vs. validation RMSE across a full range of tree depths 
  (1 to unlimited), visualizing the transition from underfitting through a 
  good fit to overfitting in a single curve
- Applied regularization to the overfit model (`max_depth=10`, 
  `min_samples_leaf=20`) and measured how much the train-validation gap 
  shrank as a result

### Key Results

| Regime | Train RMSE | Validation RMSE | Gap | Diagnosis |
|---|---|---|---|---|
| Underfit (depth=1) | 79,057.22 | 77,118.37 | 1,938.85 | Too simple to learn |
| Overfit (depth=None) | 0.00 | 62,153.07 | 62,153.07 | Memorized training data |
| Regularized (depth=10) | 45,580.26 | 50,954.80 | 5,374.54 | Best generalization |

Regularization cut the train-validation gap by **91%** and improved 
validation RMSE by 18% — proof that reducing model complexity, not 
increasing it, was the right fix for the overfit tree.

### Why This Matters
A single accuracy score can't tell you *why* a model is underperforming — 
only the train-validation comparison can. A small gap with high error on 
both sides points to underfitting (add complexity); a large gap with 
near-perfect training performance points to overfitting (add constraints). 
This notebook turns that diagnosis into a repeatable workflow: plot both 
curves across a range of complexity settings, locate where validation 
error stops improving, and use regularization to pull an overfit model back 
toward that sweet spot — exactly the judgment call every model in this 
program will require going forward.

### Tools Used
Pandas • NumPy • Matplotlib • Scikit-learn • Jupyter Notebook
