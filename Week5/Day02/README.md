# Day 2 — DBSCAN & Hierarchical Clustering

This notebook extends Day 1's K-Means analysis of the `car data.csv` dataset by applying and comparing three clustering algorithms: **K-Means**, **DBSCAN**, and **Hierarchical (Agglomerative) Clustering**.

## Dataset

`car data.csv` — used car listings with the following numeric features:

- `Year`
- `Selling_Price`
- `Present_Price`
- `Kms_Driven`
- `Owner`

## Workflow

### 1. Preprocessing
- Loaded the dataset and selected the 5 numeric columns above.
- Applied `StandardScaler` to normalize all features to mean 0, std 1 — necessary since `Present_Price`/`Selling_Price` (lakhs) and `Kms_Driven` (tens of thousands) are on very different scales and would otherwise dominate distance-based methods.

### 2. K-Means Clustering
- Ran the **Elbow Method** (k = 1–10) on inertia — sharp drop through k=4, then a flatter slope.
- Compared **Silhouette Scores** for k=3 (0.4379) vs. k=4 (0.4122) and selected **k = 3** as the final model, since silhouette score more directly measures cluster quality than the elbow heuristic.
- Visualized clusters in 2D via **PCA**.
- Interpreted the 3 clusters by profile:

| Cluster | Year | Price | Present Price | Kms | Owner | Profile |
|---|---|---|---|---|---|---|
| 0 | 2014.9 | 20.54 | 31.64 | 45,436 | 0.00 | Premium cars |
| 1 | 2014.7 | 4.04 | 5.85 | 27,601 | 0.00 | Budget, low-mileage |
| 2 | 2009.2 | 2.43 | 7.53 | 70,878 | 0.21 | Older, high-mileage |

### 3. DBSCAN Clustering
- Applied `DBSCAN(eps=0.8, min_samples=5)` on the scaled features.
- Result: **2 clusters**, **35 noise points** (points not assigned to any dense region).

### 4. Hierarchical Clustering
- Built a **Ward-linkage dendrogram** on the scaled features.
- Cut the dendrogram at **height = 10**, yielding **9 clusters**.

### 5. Comparison of Methods

| Method | Number of Clusters | Noise Points |
|---|---:|---:|
| K-Means | 4 | 0 |
| DBSCAN | 2 | 35 |
| Hierarchical Clustering | 9 | 0 |

- **K-Means** assigns every point to a cluster and produced a moderate, balanced number of groups.
- **DBSCAN** found only 2 dense clusters and flagged 35 points as noise/outliers — useful for spotting anomalous listings.
- **Hierarchical Clustering** produced the most granular split (9 clusters) at the chosen cut height.
- Results were visualized side-by-side using PCA-reduced 2D scatter plots (one panel per method).

## Tools

- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

## Key Takeaway

K-Means, DBSCAN, and Hierarchical Clustering surface different structures in the same data — the choice of method (and its parameters) materially changes how many clusters are found and how outliers are treated, so method choice should match the analysis goal (balanced segmentation vs. outlier/noise detection vs. fine-grained hierarchy).
