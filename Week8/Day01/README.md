# Day 1 — Sprint 3: NLP Preprocessing for Fake News Detection

## The Problem

Raw news text can't be fed directly into a machine learning classifier — it's full of inconsistent casing, punctuation, and common filler words that add noise rather than signal. But cleaning text isn't just a mechanical pass: an aggressive cleaning pipeline can accidentally strip out words that actually carry meaning, like negations ("not", "no", "nor"), and silently flip the sense of a sentence before the model ever sees it. This notebook is Day 1 of Sprint 3 in a Fake News Detection project, and it exists to solve that problem: build a text preprocessing pipeline that is thorough enough to reduce noise, but careful enough not to destroy meaning.

## Sprint 3 Context

**Sprint Goal:** Integrate the text preprocessing and TF-IDF pipeline with the machine learning classifier, then evaluate model performance.

This notebook covers the first half of that goal — the preprocessing stage. The remaining backlog items (TF-IDF vectorization, train/test split, model training, and evaluation with accuracy/precision/recall/F1/confusion matrix) are planned for subsequent notebooks in the sprint.

## What This Notebook Does

The notebook works through the problem in stages, each one building on the last:

1. **Loads the data.** It pulls the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) via `kagglehub` and reads `WELFake_Dataset.csv` into a pandas DataFrame.

2. **Tokenizes a sample.** Before cleaning anything, it takes one raw text sample and runs it through NLTK's `word_tokenize()` to see what raw tokenization looks like — words, numbers, and punctuation all treated as separate tokens.

3. **Builds a full cleaning pipeline.** The raw text is then run through a sequence of steps: lowercasing, punctuation removal, stop-word removal, and lemmatization (reducing words to their base dictionary form via `WordNetLemmatizer`).

4. **Checks for a specific failure mode.** This is the notebook's key design decision: standard NLTK stop-word lists include negation words like `no`, `not`, and `nor`. If those get removed along with filler words like "the" and "is", a sentence's meaning can flip — which is dangerous for a task like fake news classification, where meaning matters. The notebook explicitly checks which negation words NLTK's stop-word list would remove, then excludes those from the stop-word removal step so they survive the cleaning pass.

5. **Documents the reasoning.** Each cleaning decision (lowercasing, punctuation removal, stop-word removal, negation preservation, lemmatization) is recorded with a short justification, so the choices are auditable later — not just the code, but why the code does what it does.

## Requirements

- Python 3
- `kagglehub`
- `pandas`
- `nltk` (with the `punkt`, `stopwords`, `wordnet`, and `omw-1.4` resources — the notebook downloads these automatically on first run)

A `requirements.txt` is generated at the end of the notebook via `pip freeze`.

## How to Run

1. Install dependencies (or let the first cell's `pip install -q kagglehub` handle that piece).
2. Run the notebook top to bottom — it downloads the dataset automatically, so no manual data setup is needed.
3. NLTK resource downloads happen inline the first time they're needed; subsequent runs will be faster since the resources are cached.

## What's Next

Sprint 3's remaining backlog items — TF-IDF vectorization, splitting the data, training the classifier, and evaluating it — build directly on the cleaned text this notebook produces.
