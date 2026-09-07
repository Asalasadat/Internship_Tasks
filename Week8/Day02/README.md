# Day 2 — Text Representation: TF-IDF & Embeddings

## 📌 Project Overview

This project focuses on **text representation and classification** using the WELFake news dataset. It covers text preprocessing, TF-IDF feature extraction, Logistic Regression classification, and pre-trained GloVe word embeddings.

The notebook also compares the TF-IDF baseline with a previously developed LSTM/Transformer model.

## 🎯 Objectives

The main objectives are:

1. Load and inspect the WELFake dataset.
2. Tokenize raw news text.
3. Clean and preprocess the text.
4. Preserve important negation words.
5. Apply TF-IDF vectorization.
6. Train a Logistic Regression text-classification baseline.
7. Evaluate the model using classification metrics.
8. Load pre-trained GloVe word embeddings.
9. Explore semantic relationships using cosine similarity.
10. Compare TF-IDF performance with the Week 7 LSTM/Transformer model.

## 📂 Dataset

The notebook downloads the **WELFake dataset** using KaggleHub:

```python
kagglehub.dataset_download("saurabhshahane/fake-news-classification")
```

The dataset file used is:

```text
WELFake_Dataset.csv
```

The target variable is:

```text
label
```

The text used for classification is stored in the `text` column and is subsequently transformed into `cleaned_text`.

## 🧹 Text Preprocessing

A complete text-cleaning pipeline is applied to the news articles.

The preprocessing steps are:

1. **Lowercasing** — converts text to lowercase.
2. **Tokenization** — splits text into individual tokens using NLTK.
3. **Punctuation removal** — removes punctuation.
4. **Stop-word removal** — removes common English stop words.
5. **Negation preservation** — keeps important words such as `no`, `not`, and `nor`.
6. **Lemmatization** — reduces words to their base form.

### Why Preserve Negations?

Words such as `no`, `not`, and `nor` can change the meaning of a sentence. Therefore, the standard NLTK stop-word list is customized so that these important negation words are not removed.

## 🔢 TF-IDF Representation

The cleaned text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer is configured with:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

This means the model uses up to **5,000 features** and includes both:

- Unigrams — individual words
- Bigrams — pairs of consecutive words

The dataset is split into:

- **57,707 training samples**
- **14,427 testing samples**

## 🤖 Logistic Regression Baseline

A Logistic Regression classifier is trained using the TF-IDF features.

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Results

The baseline achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **95.56%** |
| F1-score — Class 0 | **0.95** |
| F1-score — Class 1 | **0.96** |

The results indicate strong and relatively balanced classification performance across the two classes.

## 🧠 Pre-trained GloVe Embeddings

The notebook also uses pre-trained **GloVe word embeddings**.

The 100-dimensional GloVe vectors are loaded from:

```text
glove.6B.100d.txt
```

The embedding vocabulary contains **400,000 words**, with each word represented by a 100-dimensional vector.

## 🔍 Semantic Similarity

Cosine similarity is used to find words that are close to selected words in the GloVe embedding space.

Nearest-neighbor examples include:

### `government`

- administration — 0.7937
- governments — 0.7701
- officials — 0.7590
- authorities — 0.7442
- opposition — 0.7372

### `president`

- vice — 0.8288
- presidency — 0.7150
- former — 0.7061
- presidents — 0.6962
- chairman — 0.6929

### `war`

- wars — 0.7687
- conflict — 0.7661
- invasion — 0.7430
- military — 0.7365
- occupation — 0.7300

These results demonstrate how word embeddings capture relationships based on semantic meaning and contextual usage.

## ⚖️ TF-IDF vs. Week 7 LSTM/Transformer

The notebook compares the TF-IDF + Logistic Regression baseline with the Week 7 LSTM/Transformer model using test accuracy.

| Model | Test Accuracy |
|---|---:|
| **TF-IDF + Logistic Regression** | **95.56%** |
| Week 7 LSTM/Transformer | 89.44% |

The TF-IDF baseline achieved a **6.12 percentage-point higher accuracy** than the Week 7 LSTM/Transformer model under the reported evaluation setup.

Therefore, for this dataset and evaluation setup, the **TF-IDF + Logistic Regression baseline performed better in test accuracy**.

## 🛠️ Technologies & Libraries

The project uses:

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **NLTK** — tokenization, stop words, and lemmatization
- **Scikit-learn** — TF-IDF, Logistic Regression, and evaluation
- **GloVe** — pre-trained word embeddings
- **KaggleHub** — dataset download

## 🚀 How to Run

### 1. Install dependencies

The notebook generates a `requirements.txt` file using:

```bash
pip freeze > requirements.txt
```

You can install the required packages with:

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

Open:

```text
Day02.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 3. Run the notebook sequentially

Run the cells from top to bottom. The notebook downloads the dataset and required NLTK resources during execution.

The GloVe section also downloads the pre-trained embeddings before performing semantic-similarity analysis.

## 📁 Project Structure

```text
.
├── Day02.ipynb
├── WELFake_Dataset.csv
├── glove.6B.100d.txt
├── requirements.txt
└── README.md
```

## 🧠 Key Takeaways

- Text preprocessing is an important step before applying machine-learning models to news articles.
- Negation words should be preserved because removing them can change sentence meaning.
- TF-IDF provides an effective numerical representation of text for traditional machine-learning models.
- The TF-IDF + Logistic Regression baseline achieved **95.56% test accuracy**.
- GloVe embeddings capture semantic relationships between words through their positions in vector space.
- In the reported comparison, TF-IDF + Logistic Regression outperformed the Week 7 LSTM/Transformer model by **6.12 percentage points** in test accuracy.
