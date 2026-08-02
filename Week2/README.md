# 📚 Week 2 Summary — Statistics, Probability & Exploratory Data Analysis

## 📊 Descriptive Statistics

Machine learning models learn patterns from data, so understanding a dataset before modeling is essential. Descriptive statistics help summarize a dataset by describing its **center**, **spread**, and **distribution**.

### Key Concepts
- **Measures of Central Tendency**
  - **Mean:** Average value of the dataset.
  - **Median:** Middle value after sorting the data.
  - **Mode:** Most frequently occurring value.
  - Comparing the mean and median helps identify skewness and the presence of outliers.

- **Measures of Spread**
  - **Variance:** Measures how far values are spread from the mean.
  - **Standard Deviation:** Square root of the variance, expressed in the same units as the data.
  - **Interquartile Range (IQR):** Measures the spread of the middle 50% of the data and is less affected by outliers.

Understanding these statistics provides the first insight into a dataset before creating any visualizations.

---

## 🎲 Probability & Distributions

Probability measures uncertainty and forms the foundation of predictive machine learning models.

### Key Concepts
- **Probability Rules**
  - Complement Rule
  - Addition Rule
  - Multiplication Rule

- **Conditional Probability**
  - Represents the probability of an event occurring given that another event has already occurred.
  - Forms the basis of many classification problems.

- **Bayes' Theorem**
  - Combines prior knowledge with new evidence to calculate updated probabilities.
  - Provides the mathematical foundation of the **Naive Bayes** classifier.

### Common Probability Distributions
- **Normal (Gaussian):** Symmetric, bell-shaped distribution commonly found in real-world data.
- **Binomial:** Models the number of successes in repeated independent trials.
- **Uniform:** Every outcome has an equal probability.

The normal distribution is particularly important because many statistical techniques and ML algorithms assume approximately normal data.

---

## 📐 Linear Algebra for Machine Learning

Linear algebra is the mathematical language behind machine learning. Every dataset, feature set, and model parameter can be represented using vectors and matrices.

### Key Concepts
- **Vectors:** Represent the features of a single data sample.
- **Matrices:** Represent entire datasets, where rows are samples and columns are features.
- **Dot Product:** Computes predictions in linear models using:

```text
Prediction = Dot(Features, Weights) + Bias
```

- **Matrix Multiplication:** Enables predictions for multiple samples simultaneously while following the rule that inner dimensions must match.

Understanding matrix shapes is essential for avoiding shape mismatch errors in machine learning implementations.

---

## 📈 EDA Part 1 — Distributions & Outliers

Exploratory Data Analysis (EDA) is the first step of every machine learning project. It helps identify patterns, understand distributions, and detect data quality issues before modeling.

### Key Concepts
- Statistical visualization using **Seaborn**
- **Histogram:** Shows the distribution of numeric variables.
- **Box Plot:** Displays quartiles and identifies potential outliers.
- **Count Plot:** Displays frequencies of categorical variables.
- **KDE Plot:** Shows a smooth estimate of the data distribution.

### Outlier Detection
The **Interquartile Range (IQR)** method identifies potential outliers using:

```text
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Outliers should always be investigated before deciding whether to keep, modify, or remove them.

---

## 📊 EDA Part 2 — Correlation & Data Storytelling

The second stage of EDA focuses on discovering relationships between variables and communicating insights effectively.

### Key Concepts
- **Bivariate Analysis**
  - Scatter plots for relationships between numerical variables.
  - Grouped box plots for comparing numeric values across categories.

- **Correlation Analysis**
  - Measures the strength and direction of relationships between variables.
  - Correlation heatmaps provide a quick overview of all pairwise relationships.

> **Important:** Correlation does **not** imply causation.

- **Pairplot**
  - Provides a complete overview of relationships among numerical variables.

### Data Storytelling

The final goal of EDA is not just creating charts, but communicating insights through a clear narrative that explains:
- What the dataset contains.
- Which patterns and relationships were discovered.
- What data quality issues exist.
- What these findings imply for future machine learning models.

> **A correlation without interpretation is just a number—the story behind the data is what makes it valuable.**
