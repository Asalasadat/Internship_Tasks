# Day 1 — Sprint 3 Planning & NLP Preprocessing

## Project Overview
This notebook kicks off **Sprint 3**, a new phase of the Phase 3 capstone focused on **Natural Language Processing (NLP)**. It opens with Sprint Planning — defining the goal of integrating a trained NLP model into a complete, evaluated pipeline — then walks through the foundational text-preprocessing steps: tokenization, lowercasing, punctuation removal, stop-word removal, and lemmatization.

## Sprint 3 Planning

**Sprint Goal:** Integrate the trained NLP model into a complete pipeline and evaluate it rigorously.

**Backlog:**
- Preprocess and clean raw text.
- Tokenize the text.
- Apply appropriate text normalization.
- Evaluate stop-word removal.
- Apply lemmatization where appropriate.
- Integrate preprocessing with the NLP model.
- Evaluate the complete pipeline using appropriate metrics.
- Document experiments and results.

## Workflow

### 1. Tokenization
Tokenized a raw sample sentence using NLTK's `word_tokenize()`:

- **Original text:** `"Natural Language Processing allows computers to understand human language."`
- **Tokenized output:** `['Natural', 'Language', 'Processing', 'allows', 'computers', 'to', 'understand', 'human', 'language', '.']`

Tokenization is the first step in any NLP pipeline — it converts unstructured text into discrete units (tokens) that downstream models can process.

### 2. Full Text-Cleaning Pipeline
Applied a complete preprocessing pipeline to a second sample sentence (`"Natural Language Processing allows computers to understand human languages. It is very useful!"`), combining five steps in sequence:

1. **Tokenization** — split the text into words/punctuation via `word_tokenize()`.
2. **Lowercasing** — normalized all tokens to lowercase for consistency.
3. **Punctuation Removal** — filtered out tokens found in `string.punctuation`.
4. **Stop-Word Removal** — removed common English stop words (`the`, `is`, `to`, etc.) using NLTK's English stopword list.
5. **Lemmatization** — reduced each remaining word to its base dictionary form using `WordNetLemmatizer`.

**Result:**
- **Cleaned tokens:** `['natural', 'language', 'processing', 'allows', 'computer', 'understand', 'human', 'language', 'useful']`
- **Cleaned text:** `"natural language processing allows computer understand human language useful"`

Note how `computers` → `computer` and `languages` → `language` after lemmatization, and filler words (`to`, `it`, `is`, `very`) and the exclamation mark are dropped entirely.

## Key Findings
1. Tokenization is the essential first step that breaks raw text into processable units before any further NLP analysis.
2. A standard cleaning pipeline — lowercase → remove punctuation → remove stop words → lemmatize — significantly reduces noise and normalizes vocabulary (e.g. collapsing plural/singular forms) while preserving the core meaning of the sentence.
3. This cleaned, normalized token output is now ready to feed into vectorization (e.g. TF-IDF, embeddings) and the NLP model integration planned later in the Sprint 3 backlog.

## Tools Used
- Python, Jupyter Notebook (Google Colab)
- **NLTK** — `word_tokenize`, `stopwords` corpus, `WordNetLemmatizer`; NLTK resource downloads (`punkt_tab`, `stopwords`, `wordnet`)
- Python `string` module — punctuation filtering

## Files
- `Day01.ipynb` — full notebook: Sprint 3 planning, tokenization demo, and the full text-cleaning pipeline (lowercase → remove punctuation → remove stop words → lemmatize)
