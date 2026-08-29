# Week 5 — Unsupervised Learning

**BinX Tech · AI & Machine Learning Internship Program**
*Clustering, Dimensionality Reduction & Capstone Project Kickoff*

The Phase 2 → 3 transition: learning from data that has no labels. Interns cluster data with K-Means and DBSCAN, reduce high-dimensional data with PCA and t-SNE, detect anomalies — then select their Phase 3 capstone project and run Sprint 1 planning.

`PHASE 2 → 3` · `40 HOURS` · `5 TRAINING DAYS` · `HANDS-ON`

---

## 📋 Week 5 Overview

| | |
|---|---|
| **Week** | Week 5 of 10 — Phase 2 → Phase 3 transition |
| **Total Hours** | 40 hours (full-time track) / 20 hours (part-time track, Weeks 9–10 combined) |
| **Format** | On-site / Remote / Hybrid — all work in Jupyter notebooks committed to GitHub |
| **Focus** | Unsupervised learning: K-Means, DBSCAN, hierarchical clustering, PCA, t-SNE, anomaly detection; Phase 3 capstone project selection and Sprint 1 planning |
| **Prerequisite** | Weeks 1–4 completed: Python/EDA, supervised learning, evaluation, pipelines |
| **Mentor Supervision** | Daily check-in; Day 5 project selection and Sprint 1 plan require mentor sign-off before Phase 3 begins |

## 🎯 Week 5 Learning Objectives

- Explain unsupervised learning and how it differs from supervised learning.
- Cluster data with K-Means and choose k using the elbow method and silhouette score.
- Cluster data with DBSCAN and hierarchical clustering, and explain when each is preferable.
- Reduce dimensionality with PCA and explain explained variance; visualize high-dimensional data with t-SNE.
- Detect anomalies in unlabeled data.
- Select a Phase 3 capstone project and complete Sprint 1 planning with a mentor-approved scope.

## 🗓️ Daily Schedule

| Day | Hours | Topic Focus |
|---|---:|---|
| **Day 1** | 8 hrs | Unsupervised learning concepts; K-Means clustering and choosing k |
| **Day 2** | 8 hrs | DBSCAN and hierarchical clustering; comparing clustering methods |
| **Day 3** | 8 hrs | Dimensionality reduction: PCA and explained variance |
| **Day 4** | 8 hrs | t-SNE for visualization; anomaly detection |
| **Day 5** | 8 hrs | Phase 3 project selection; Sprint 1 planning and backlog |

---

## 📚 Day-by-Day Curriculum

### Day 1 — Unsupervised Learning & K-Means (8 hours)

**Learning Objectives**
- Explain unsupervised learning and how it differs from supervised learning.
- Run K-Means clustering and interpret the resulting clusters and centroids.
- Choose the number of clusters k using the elbow method and silhouette score.

**Key Topics:** Supervised vs. unsupervised learning · What clustering does · K-Means: the centroid-assignment loop · Choosing k (elbow method, silhouette score) · Scaling before clustering

**Hands-On Lab:** Load and scale a dataset with `StandardScaler` → run K-Means for k=1–10 and plot the elbow → compute silhouette scores for top candidates → fit final K-Means and visualize in 2D → interpret each cluster in Markdown.

**Tools:** Scikit-learn (`KMeans`) · `StandardScaler` · Matplotlib · Jupyter Notebook

---

### Day 2 — DBSCAN & Hierarchical Clustering (8 hours)

**Learning Objectives**
- Explain K-Means's limitations and when to prefer another method.
- Run DBSCAN and interpret its clusters and noise points.
- Build and read a hierarchical clustering dendrogram, and choose a method per situation.

**Key Topics:** Limitations of K-Means · DBSCAN (density-based clustering, noise detection, `eps`/`min_samples`) · Hierarchical clustering and dendrograms · Choosing the right clustering method

**Hands-On Lab:** Run DBSCAN on the Day 1 dataset and report clusters/noise points → build a Ward-linkage dendrogram and choose a cut height → compare K-Means, DBSCAN, and hierarchical results → state in Markdown which method best fits the data's shape.

**Tools:** Scikit-learn (`DBSCAN`) · SciPy (`dendrogram`) · Matplotlib · Jupyter Notebook

---

### Day 3 — Dimensionality Reduction with PCA (8 hours)

**Learning Objectives**
- Explain the curse of dimensionality and why reduction helps.
- Apply PCA to reduce a dataset's dimensions.
- Interpret explained variance and choose how many components to keep.

**Key Topics:** The curse of dimensionality · What PCA does (principal components and variance) · Explained variance ratio · Choosing component count (e.g. 95% variance) · When (and when not) to use PCA

**Hands-On Lab:** Scale a high-dimensional dataset → fit PCA and plot cumulative explained variance → choose the component count retaining ~95% variance and justify it → reduce to 2D and plot, colored by group → document what was preserved vs. cost in Markdown.

**Tools:** Scikit-learn (`PCA`) · `StandardScaler` · Matplotlib · Jupyter Notebook

---

### Day 4 — t-SNE & Anomaly Detection (8 hours)

**Learning Objectives**
- Use t-SNE to visualize high-dimensional data and distinguish it from PCA.
- Explain what anomaly detection is and why it is often unsupervised.
- Detect anomalies with Isolation Forest and interpret the flagged points.

**Key Topics:** t-SNE for local-structure visualization · PCA vs. t-SNE · What anomaly detection is · Isolation Forest and the `contamination` parameter · Anomaly detection ↔ clustering overlap

**Hands-On Lab:** Apply t-SNE to reduce and plot data, colored by cluster → compare against the Day 3 PCA plot → run Isolation Forest and report flagged points → inspect two anomalies and hypothesize why, in Markdown.

**Tools:** Scikit-learn (`TSNE`, `IsolationForest`) · Matplotlib · Jupyter Notebook

---

### Day 5 — Phase 3 Project Selection & Sprint 1 Planning (8 hours)

**Learning Objectives**
- Select a Phase 3 capstone project type in consultation with the mentor.
- Restate the project's Definition of Done and keep it visible from the start.
- Complete Sprint 1 planning: backlog, effort estimates, sprint goal, and acceptance criteria.

**Key Topics:** Entering Phase 3 (the four-sprint capstone) · The six capstone project options · The professional baseline / Definition of Done · Sprint 1 planning · Acceptance criteria and the GitHub branch/PR workflow

**Capstone Project Options:**

| Project | In Short |
|---|---|
| Customer Churn Prediction | Binary classification on telecom/SaaS data; EDA, feature importance, threshold tuning |
| House Price Prediction | Regression on real-estate data; pipelines, polynomial features, regularization |
| Sentiment Analysis Tool | NLP classifier for reviews/tweets (positive/neutral/negative); TF-IDF or a transformer |
| Image Classifier | CNN classifying images; data augmentation and transfer learning |
| Recommendation System | Collaborative or content-based filtering on interaction data |
| Fraud Detection Model | Imbalanced classification; SMOTE, precision-recall trade-offs, SHAP explanations |

**Hands-On Lab:** Review the six options with the mentor and select one → write the problem statement and restate the Definition of Done → create the Sprint 1 backlog (dataset selection, EDA, baseline model) with acceptance criteria per task → define the Sprint 1 goal and get mentor sign-off → set up the project GitHub repo with a README skeleton and feature-branch workflow.

**Tools:** Mentor consultation · GitHub (repository + branches) · Sprint backlog · Jupyter Notebook

---

## 📦 Week 5 Deliverables

By the end of Week 5, every intern must submit the following to their mentor and GitHub repository:

- [ ] A K-Means notebook with elbow-method and silhouette analysis and interpreted clusters.
- [ ] A clustering-comparison notebook (K-Means vs. DBSCAN vs. hierarchical) with a method recommendation.
- [ ] A PCA notebook with a cumulative explained-variance plot and a justified component count.
- [ ] A t-SNE and anomaly-detection notebook with a 2D visualization and Isolation Forest results.
- [ ] A signed-off Phase 3 project selection, problem statement, and Sprint 1 plan with backlog and acceptance criteria.
- [ ] All Week 5 notebooks and the initialized project repository committed to GitHub.

## 📊 Week 5 Evaluation Criteria

Scored by the assigned mentor at the end of Week 5 using the criteria below (extracted from the program's 100-point internal rubric).

| Criterion | 50–69 Developing | 70–84 Proficient | 85–100 Excellent |
|---|---|---|---|
| Understanding of unsupervised concepts | Runs algorithms but unclear on how they work | Explains clustering and reduction methods clearly | Deep understanding; justifies method choice per data shape |
| Clustering & reduction correctness | Basic use with guidance | Correct use, sensible k / component choices | Rigorous; validates with silhouette/variance, scales properly |
| Interpretation & insight | Reports output without meaning | Interprets clusters and components clearly | Insight-driven; connects structure to real meaning |
| Project planning quality | Vague scope or missing acceptance criteria | Clear project, backlog, and Sprint 1 goal | Well-scoped plan anticipating Phase 3 risks |
| Notebook quality & Git workflow | Sporadic commits, weak narrative | Regular commits, clear Markdown narrative | Consistent, descriptive, reproducible, well-organized |
| Attendance & punctuality | 3–6 absences | 1–2 absences, on time | Perfect attendance, proactive |

## 🧰 Technical Stack — Week 5

| Area | Tools |
|---|---|
| Clustering | Scikit-learn (`KMeans`, `DBSCAN`), SciPy (hierarchical/dendrogram) |
| Dimensionality Reduction | Scikit-learn (`PCA`, `TSNE`) |
| Anomaly Detection | Scikit-learn (`IsolationForest`) |
| Evaluation | `silhouette_score`, `explained_variance_ratio_` |
| Environment | Python 3.10+, Pandas, Matplotlib, Jupyter Notebook, Git & GitHub |

## 📁 Repository Structure

```text
.
├── Day01.ipynb
├── Day02.ipynb
├── Day03.ipynb
├── Day04.ipynb
└── README.md
```

## 📝 Notes & Best Practices

Unsupervised learning has no answer key, so judgment matters more than ever: scale before clustering, validate k with real metrics, and interpret results rather than trusting them blindly. Day 5 is the hinge into Phase 3 — no capstone work begins until the project scope and Sprint 1 plan are approved by the mentor, exactly as the program requires.
