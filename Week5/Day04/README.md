# Day 4 — t-SNE & Anomaly Detection

This project explores **unsupervised machine learning techniques** for analyzing a used-car dataset. The notebook focuses on clustering, dimensionality reduction, visualization, and anomaly detection.

## Dataset

The analysis uses the `car data.csv` dataset, containing **301 car records** with information such as:

* `Car_Name`
* `Year`
* `Selling_Price`
* `Present_Price`
* `Kms_Driven`
* `Fuel_Type`
* `Seller_Type`
* `Transmission`
* `Owner`

The analysis primarily uses these numerical features:

```text
Year
Selling_Price
Present_Price
Kms_Driven
Owner
```

## Objectives

The main objectives of this notebook are to:

1. Standardize the numerical features.
2. Determine suitable clusters using K-Means.
3. Compare different clustering techniques.
4. Visualize high-dimensional data using PCA and t-SNE.
5. Detect potential anomalies using Isolation Forest.
6. Interpret unusual observations in the dataset.

## Technologies & Libraries

The notebook uses Python and the following libraries:

* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

Important machine learning algorithms and techniques include:

* `StandardScaler`
* K-Means
* DBSCAN
* Hierarchical Clustering
* PCA
* t-SNE
* Isolation Forest
* Silhouette Score

## Data Preprocessing

The numerical features are standardized using `StandardScaler`.

Scaling is important because the features have very different numerical ranges. For example, `Year` is a four-digit value, `Owner` ranges from 0–3, while `Kms_Driven` can reach much larger values.

Standardization transforms the features so that each has approximately:

* Mean = 0
* Standard deviation = 1

This prevents distance-based algorithms such as K-Means from being dominated by features with larger raw values.

## K-Means Clustering

The Elbow Method was first used to examine possible values of `k`. The notebook identifies an elbow around **k = 4**.

Silhouette scores were then calculated for `k = 3` and `k = 4`:

| k | Silhouette Score |
| - | ---------------: |
| 3 |           0.4379 |
| 4 |           0.4122 |

Because `k = 3` produced the higher silhouette score, the final K-Means analysis uses **3 clusters**.

The resulting clusters can broadly be interpreted as:

| Cluster | Profile                  |
| ------- | ------------------------ |
| 0       | Premium cars             |
| 1       | Budget, low-mileage cars |
| 2       | Older, high-mileage cars |

The clustering separates the cars mainly according to differences in price and mileage.

## PCA Visualization

Principal Component Analysis (PCA) is used to reduce the standardized features to two dimensions for visualization.

The resulting 2D representation makes it possible to visually inspect the K-Means clusters and their centroids.

PCA provides a linear dimensionality reduction and preserves directions containing the highest overall variance.

## DBSCAN

DBSCAN is applied using:

```python
DBSCAN(eps=0.8, min_samples=5)
```

The results are:

* **2 clusters**
* **35 noise points**

Unlike K-Means, DBSCAN can identify observations that do not belong to sufficiently dense regions and classify them as noise.

## Hierarchical Clustering

Hierarchical clustering is performed using the **Ward linkage** method.

A dendrogram is generated and a cut height of **10** is selected.

Results:

* **Cut height:** 10
* **Number of clusters:** 9

## Clustering Comparison

The three clustering approaches produce different structures:

| Method                  | Number of Clusters | Noise Points |
| ----------------------- | -----------------: | -----------: |
| K-Means                 |                  4 |            0 |
| DBSCAN                  |                  2 |           35 |
| Hierarchical Clustering |                  9 |            0 |

### Interpretation

* **K-Means** assigns every observation to one of the predefined clusters.
* **DBSCAN** identifies dense regions and can classify observations as noise.
* **Hierarchical Clustering** produces a hierarchy of groups that can be explored using a dendrogram.

The different results demonstrate that clustering structure depends strongly on the algorithm and its assumptions.

## t-SNE Visualization

t-SNE is used to reduce the standardized dataset to two dimensions:

```python
TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)
```

The resulting visualization is colored according to the K-Means cluster assignments.

The t-SNE representation shows some relatively distinct groups, although there is still overlap between certain clusters.

### PCA vs. t-SNE

**PCA**

* Linear dimensionality reduction.
* Focuses on directions with the highest variance.
* Provides a more interpretable representation of global structure.

**t-SNE**

* Focuses on preserving local relationships.
* Useful for visualizing local groups and patterns.
* Can reveal separations that are less obvious in a linear PCA projection.

## Anomaly Detection

An **Isolation Forest** is used to identify potentially unusual observations.

The model is trained on the standardized numerical features:

```text
Year
Selling_Price
Present_Price
Kms_Driven
Owner
```

The model uses:

```python
IsolationForest(
    contamination='auto',
    random_state=42
)
```

### Results

The Isolation Forest model flagged:

**38 observations as potential anomalies.**

These observations are not necessarily incorrect. They may simply represent cars with unusual combinations of characteristics.

## Example Anomalies

Two types of potentially unusual observations were examined.

### 1. Unusually Expensive Car

One observation appears far from the main group in the PCA/t-SNE visualizations and is associated with the premium-car group.

A possible explanation is unusually high:

* `Selling_Price`
* `Present_Price`

These extreme price values make the observation different from most cars.

### 2. Older / High-Mileage Car

Another unusual observation appears within the older/high-mileage group.

Possible reasons include:

* An unusually old `Year`
* Very high `Kms_Driven`

These extreme values can cause the observation to be identified as anomalous.

Importantly, an anomaly score does **not** automatically mean that a record is wrong. It may simply represent a legitimate but unusual car.

## Project Workflow

```text
Load car dataset
       ↓
Select numerical features
       ↓
Standardize features
       ↓
      ┌───────────────┐
      │               │
   K-Means         DBSCAN
      │               │
      └───────┬───────┘
              │
    Hierarchical Clustering
              │
              ↓
       Compare clusters
              │
        ┌─────┴─────┐
        ↓           ↓
       PCA         t-SNE
        │           │
        └─────┬─────┘
              ↓
      Isolation Forest
              ↓
     Identify anomalies
```

## How to Run

### 1. Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn scipy
```

### 2. Provide the dataset

Place `car data.csv` in the appropriate location or update the path in the notebook.

The original notebook loads the dataset with:

```python
df = pd.read_csv('/content/car data.csv')
```

### 3. Run the notebook

Open `Day4.ipynb` in:

* Google Colab
* Jupyter Notebook
* JupyterLab

Run the cells from top to bottom.

## Key Takeaways

* Feature scaling is important before applying distance-based unsupervised learning methods.
* K-Means produced meaningful groups of cars based on numerical characteristics.
* The silhouette score favored **3 clusters** over 4 in the notebook's final K-Means selection.
* DBSCAN identified **2 clusters and 35 noise points**.
* Hierarchical clustering produced **9 clusters** at a dendrogram cut height of 10.
* PCA and t-SNE provide complementary ways to visualize the high-dimensional data.
* Isolation Forest identified **38 potential anomalies**.
* Anomalies should be investigated rather than automatically treated as errors.

