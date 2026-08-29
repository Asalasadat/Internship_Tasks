# Day 5 — Phase 3 Project Selection & Sprint 1 Planning

This deliverable closes out Phase 2 of the internship program and prepares for **Phase 3**, the four-sprint applied AI/ML capstone (Weeks 6–9). Unlike prior days, the output here is a **planning artifact** rather than a data notebook: a selected capstone project, a restated Definition of Done, and a fully scoped Sprint 1 backlog — all signed off by the mentor before implementation begins.

## 📌 Project Overview

Phase 3 requires each intern to build one complete AI/ML project end-to-end — from raw data to a deployed, usable product — across four one-week sprints. Week 5 exists to select that project and plan its first sprint, so Week 6 can begin with a clear, mentor-approved scope.

## 🎯 Objectives

1. Select a Phase 3 capstone project type in consultation with the mentor.
2. Restate the project's Definition of Done and keep it visible from the start.
3. Complete Sprint 1 planning: backlog, effort estimates, sprint goal, and acceptance criteria.

## 🗂️ Capstone Project Options

One project type is selected from the six below (or a comparable live BinX Tech use case):

| Project | In Short |
|---|---|
| Customer Churn Prediction | Binary classification on telecom/SaaS data; EDA, feature importance, threshold tuning |
| House Price Prediction | Regression on real-estate data; pipelines, polynomial features, regularization |
| Sentiment Analysis Tool | NLP classifier for reviews/tweets (positive/neutral/negative); TF-IDF or a transformer |
| Image Classifier | CNN classifying images; data augmentation and transfer learning |
| Recommendation System | Collaborative or content-based filtering on interaction data |
| Fraud Detection Model | Imbalanced classification; SMOTE, precision-recall trade-offs, SHAP explanations |

## ✅ Professional Baseline (Definition of Done)

Regardless of the project chosen, every capstone must meet the same standard:

- A clean, documented Jupyter Notebook covering the full pipeline: **EDA → preprocessing → modelling → evaluation**
- A trained model with reported metrics
- A working deployment at a public URL (Streamlit or FastAPI)
- A GitHub repo with a clean README, `requirements.txt`, and model artifacts
- A short technical write-up

Keeping this end goal visible from Sprint 1 is what keeps all four sprints focused.

## 🏃 Sprint 1 Planning

### Sprint Goal
> Understand the data and establish a baseline model to beat.

### Backlog & Acceptance Criteria

The natural first tasks — dataset selection, EDA, and a baseline model — are broken down with written acceptance criteria per task:

- [ ] Notebook cells run without errors
- [ ] Code is committed to the correct feature branch with a clear message
- [ ] A pull request is opened for mentor review before merging
- [ ] Results are documented in Markdown
- [ ] Metrics are logged and compared against the baseline

## 🧪 Hands-On Lab

1. **Step 1:** Review the six project options with the mentor and select one based on strengths and interests.
2. **Step 2:** Write the project's problem statement and restate its Definition of Done.
3. **Step 3:** Create the Sprint 1 backlog (dataset selection, EDA, baseline model) with written acceptance criteria per task.
4. **Step 4:** Define the Sprint 1 goal and get mentor sign-off before any Phase 3 work begins.
5. **Step 5:** Set up the project GitHub repository with a README skeleton and a feature-branch workflow.

## 🛠️ Tools Used

- Mentor consultation
- GitHub (repository + branches)
- Sprint backlog
- Jupyter Notebook

## 📦 Week 5 Deliverables

- A K-Means notebook with elbow-method and silhouette analysis and interpreted clusters.
- A clustering-comparison notebook (K-Means vs. DBSCAN vs. hierarchical) with a method recommendation.
- A PCA notebook with a cumulative explained-variance plot and a justified component count.
- A t-SNE and anomaly-detection notebook with a 2D visualization and Isolation Forest results.
- A signed-off Phase 3 project selection, problem statement, and Sprint 1 plan with backlog and acceptance criteria.
- All Week 5 notebooks and the initialized project repository committed to GitHub.

## 📊 Week 5 Evaluation Criteria

Scored by the assigned mentor using the program's 100-point internal rubric:

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

## 📁 Project Structure

```text
.
└── README.md # Problem statement, DoD, backlog, acceptance criteria
```

## 🧠 Key Takeaway

A well-scoped Sprint 1 — clear project choice, explicit Definition of Done, and acceptance criteria written *before* work begins — is what keeps the four-week Phase 3 capstone (Weeks 6–9) focused and prevents scope drift once implementation starts.
