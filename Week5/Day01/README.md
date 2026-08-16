# Day 1 — Unsupervised Learning & K-Means
 
## Project Overview
This notebook applies unsupervised learning (K-Means clustering and PCA) to a used-car pricing dataset to discover natural groupings among vehicles based on their numeric attributes — without using any target/label information.
 
## Dataset
- **Source:** Car Price dataset (`car data.csv`)
- **Size:** 301 records
- **Features used:** `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Owner` (numeric columns only; categorical columns `Car_Name`, `Fuel_Type`, `Seller_Type`, `Transmission` were excluded from clustering)
## Workflow
 
### 1. Scaling
All numeric features were standardized with `StandardScaler` (mean 0, standard deviation 1). This step is essential for distance-based methods like K-Means and PCA — without it, `Kms_Driven` (largest raw magnitude) would dominate the distance calculation regardless of its actual importance relative to price or age.
 
### 2. Choosing k — Elbow Method
K-Means was run for k = 1 to 10, plotting inertia against k. Inertia dropped sharply through k=4, then flattened — suggesting an elbow around **k=4**.
 
### 3. Choosing k — Silhouette Score
To confirm the choice, silhouette scores were computed for the two leading candidates:
 
| k | Silhouette Score |
|---|---|
| 3 | **0.4379** |
| 4 | 0.4122 |
 
**k=3** was selected as the final value — it produced tighter, better-separated clusters than k=4, despite the elbow plot's slightly sharper visual bend at 4. This is a common and expected discrepancy: the elbow method is a visual heuristic based on inertia alone, while silhouette score directly measures cluster quality (cohesion and separation), making it the more reliable criterion when the two disagree.
 
### 4. Final Clustering & Visualization
K-Means was refit with k=3, and PCA was used to reduce the 5 scaled features to 2 components purely for visualization (not for clustering itself).
 
- **PC1: 38.3% variance, PC2: 32.5% variance — 70.8% total captured**, a reasonably balanced 2D representation with no single dominant axis.
- The scatter plot shows one large, dense cluster (budget/low-mileage cars) tightly packed near the origin, while the premium and older/high-mileage clusters are more spread out and overlap more with each other along PC1.
- Several outliers are visible (e.g. one unusually expensive car far right on PC1, several high-mileage/older cars with very negative PC2), pulling their respective centroids away from their main point mass.
- The moderate silhouette score (0.44) is consistent with this visible overlap — clusters are reasonably distinct but not perfectly separated, and ~29% of variance isn't captured in the 2D view.
## Cluster Interpretation
 
| Cluster | Year | Selling Price | Present Price | Kms Driven | Owner | Profile |
|---|---|---|---|---|---|---|
| 0 | 2014.9 | 20.54 | 31.64 | 45,436 | 0.00 | **Premium cars** |
| 1 | 2014.7 | 4.04 | 5.85 | 27,601 | 0.00 | **Budget, low-mileage cars** |
| 2 | 2009.2 | 2.43 | 7.53 | 70,878 | 0.21 | **Older, high-mileage cars** |
 
The clustering separates cars along two largely independent factors:
- **Price tier** — Cluster 0 (premium) vs. Cluster 1 (budget), despite both being similar in age.
- **Age/wear** — Cluster 2 stands apart with the oldest average year, highest mileage, and the only cluster with any prior owners on average.
This aligns with the PCA result: two components of comparable variance (38% vs. 33%) reflect these two distinct underlying dimensions (value/segment and vehicle wear/age) rather than one single dominant factor explaining the data.
 
## Key Findings
1. Silhouette score (a direct cluster-quality metric) was prioritized over the elbow method (a visual heuristic) when the two suggested different values of k, leading to a final choice of k=3.
2. The dataset naturally separates along two independent axes — price/segment and age/wear — rather than a single dominant pattern.
3. Clusters are reasonably well-defined but show meaningful overlap, driven by a handful of outlier vehicles with atypical price or mileage.
## Tools Used
- Python, Jupyter Notebook
- Pandas — data loading and grouping
- Scikit-learn — `StandardScaler`, `KMeans`, `silhouette_score`, `PCA`
- Matplotlib — elbow plot, cluster scatter plot
## Files
- `Day1.ipynb` — full notebook: scaling, elbow method, silhouette validation, final clustering, PCA visualization, and interpretation
- `requirements.txt` — exact library versions for reproducibility
- `Dataset` - car data.csv
