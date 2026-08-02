## Day 1 — Supervised Learning Concepts & the Scikit-learn API

### Overview
In this day, I learned the fundamentals of supervised learning workflow using Scikit-learn — specifically how to prepare a dataset for training by separating features from the target variable and splitting the data into training and testing sets.

### Topics Covered
- Separating a dataset into features (`X`) and a target variable (`y`)
- Performing an 80/20 train/test split using `train_test_split`
- Using `random_state` to ensure reproducible splits
- Verifying the shapes of `X_train`, `X_test`, `y_train`, and `y_test`
- Understanding why a model must never see the test set during training

### Key Concepts

**Features (X) vs. Target (y)**
The dataset is divided into input features (`X`) that the model uses to make predictions, and a target variable (`y`) that the model is trained to predict. In this notebook, `math score` was selected as the target, so it was dropped from `X` and stored separately in `y`.

**Train/Test Split**
The data was split so that 80% is used for training the model and 20% is held out for testing on unseen data. Setting `random_state=42` ensures the same split can be reproduced on every run.

**Why the Test Set Must Stay Unseen**
A model can memorize its training data and appear to perform perfectly, while actually failing on new, unseen data. Keeping the test set completely separate during training is what allows an honest evaluation of how well the model generalizes.

### Result
X_train: (800, 7)
X_test: (200, 7)
y_train: (800,)
y_test: (200,)

### Tools Used
Pandas • Scikit-learn • Jupyter Notebook
