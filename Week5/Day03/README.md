# Day 3 — Dimensionality Reduction & Clustering

This project explores **dimensionality reduction and unsupervised learning techniques** using a car dataset. The notebook focuses on preparing numerical features, applying feature scaling, comparing clustering approaches, and using **Principal Component Analysis (PCA)** to reduce the dimensionality of the data.

## 📌 Project Overview

The dataset contains information about used cars, including:

* Car name
* Year
* Selling price
* Present price
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Number of previous owners

The notebook works with **301 observations** and uses five numerical features for clustering and PCA:

```text
Year
Selling_Price
Present_Price
Kms_Driven
Owner
```

## 🎯 Objectives

The main objectives of this project are:

1. Load and inspect the car dataset.
2. Select relevant numerical features.
3. Standardize the features using `StandardScaler`.
4. Determine a suitable number of clusters for K-Means.
5. Apply K-Means clustering.
6. Evaluate clusters using the Silhouette Score.
7. Apply PCA for dimensionality reduction.
8. Determine the number of PCA components required to preserve at least 95% of the variance.
9. Apply DBSCAN clustering.
10. Apply hierarchical clustering.
11. Compare the different clustering methods.

## 🛠️ Technologies & Libraries

The project is implemented in Python using:

* **Python**
* **Pandas** — data loading and manipulation
* **NumPy** — numerical operations
* **Matplotlib** — visualization
* **Seaborn** — visualization
* **Scikit-learn** — scaling, clustering, evaluation, and PCA

The notebook imports `StandardScaler`, `KMeans`, `silhouette_score`, and `PCA` from scikit-learn.

## 📂 Dataset

The notebook loads the dataset from:

```text
car data.csv
```

The dataset contains 301 rows and includes numerical and categorical variables related to used cars.

## 🔄 Data Preparation

Five numerical variables are selected:

```python
numeric_cols = [
    'Year',
    'Selling_Price',
    'Present_Price',
    'Kms_Driven',
    'Owner'
]
```

The selected features are then standardized using `StandardScaler`:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
```

After scaling, the features have approximately a mean of **0** and a standard deviation of **1**.

### Why Scaling?

The numerical variables have very different scales. For example, `Kms_Driven` contains much larger raw values than `Owner`. Without scaling, distance-based techniques such as K-Means and PCA can be disproportionately influenced by variables with larger numerical magnitudes.

## 🔵 K-Means Clustering

K-Means clustering is applied to the standardized numerical features.

### Elbow Method

The elbow analysis shows that inertia decreases sharply up to approximately **k = 4**, after which the improvement becomes smaller. The notebook therefore identifies **k = 4** as the approximate elbow point.

### Silhouette Analysis

The notebook compares two candidate values:

| Number of Clusters | Silhouette Score |
| -----------------: | ---------------: |
|                  3 |           0.4379 |
|                  4 |           0.4122 |

Although the elbow method suggests approximately four clusters, **k = 3** is selected because it produces the higher Silhouette Score.

```text
k = 3 → 0.4379
k = 4 → 0.4122
```

The final K-Means model therefore uses **3 clusters**.

## 📉 Principal Component Analysis (PCA)

PCA is used to reduce the dimensionality of the standardized feature space while retaining as much information as possible.

When two principal components are used, they capture:

* **PC1:** 38.30% of the variance
* **PC2:** 32.51% of the variance
* **Combined:** approximately **70.81%**

The notebook also calculates the cumulative explained variance to determine how many components are needed to preserve at least 95% of the variance.

### PCA Component Selection

The analysis selects **4 principal components**, which retain approximately **98.48% of the total variance**.

| Measure             |     Result |
| ------------------- | ---------: |
| Original features   |          5 |
| Selected components |          4 |
| Variance retained   |     98.48% |
| Original shape      | `(301, 5)` |
| Reduced shape       | `(301, 4)` |

Thus, PCA reduces the feature space from five dimensions to four while preserving most of the information.
A separate two-dimensional PCA representation is also created for visualization of the clusters.

## 🟢 DBSCAN Clustering

DBSCAN is applied to the same standardized numerical features using:

```python
DBSCAN(eps=0.8, min_samples=5)
```

The resulting analysis identifies:

* **2 clusters**
* **35 noise points**

Unlike K-Means, DBSCAN explicitly identifies observations that do not belong to sufficiently dense clusters.

## 🌳 Hierarchical Clustering

Hierarchical clustering is performed using **Ward linkage** on the standardized numerical features.

A dendrogram is generated and a **cut height of 10** is selected.

The resulting clustering produces:

* **Cut height:** 10
* **Number of clusters:** 9

## 📊 Clustering Comparison

The notebook compares the three clustering approaches:

| Method                  | Number of Clusters | Noise Points |
| ----------------------- | -----------------: | -----------: |
| K-Means                 |                  4 |            0 |
| DBSCAN                  |                  2 |           35 |
| Hierarchical Clustering |                  9 |            0 |

K-Means assigns every observation to a cluster, DBSCAN identifies both dense clusters and noise points, while hierarchical clustering produces nine clusters at the selected dendrogram cut height.

> **Note:** The notebook's clustering comparison reports K-Means as **4 clusters**, while the earlier silhouette analysis selects **k = 3** as the preferred K-Means value. This README preserves both results as they appear in the notebook rather than silently changing one of them.

## 📈 Visualizations

The notebook includes visual analysis such as:

* Elbow plot for K-Means
* PCA projections
* Cluster visualizations
* DBSCAN clustering results
* Hierarchical clustering dendrogram
* Comparison of clustering methods

A 2D PCA projection is used to visualize observations according to their first two principal components and cluster assignments.

## 🚀 How to Run

### 1. Clone or download the project

Make sure the following files are in the same directory:

```text
Day3.ipynb
car data.csv
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the notebook

Using Jupyter Notebook:

```bash
jupyter notebook Day3.ipynb
```

Or open the notebook in **Google Colab**.

### 4. Run the cells

Run the notebook cells sequentially so that the dataset, scaled features, clustering models, and PCA transformations are created in the correct order.

## 🧠 Key Takeaways

* Numerical features were standardized before applying clustering and PCA.
* The elbow method suggested an approximate clustering structure around **k = 4**.
* Silhouette analysis favored **k = 3**, with a score of **0.4379**.
* PCA reduced the five-dimensional feature space to four components while retaining **98.48% of the variance**.
* DBSCAN identified **2 clusters and 35 noise points**.
* Hierarchical clustering produced **9 clusters** using a dendrogram cut height of **10**.
* Different clustering algorithms produce different structures because they use different definitions of similarity and cluster formation.

## 📁 Project Structure

```text
.
├── Day3.ipynb
├── car data.csv
└── README.md
```

## 📌 Project Focus

This notebook is part of a machine-learning learning sequence and focuses on **dimensionality reduction and unsupervised clustering**, with PCA and multiple clustering algorithms applied to the car dataset.
