\# Day 4 — Attention \& Transformers



\## Overview



Day 4 focuses on \*\*Attention and Transformer architectures\*\* for text classification.



Unlike the previous day, which used RNNs and LSTMs for sequential ECG signal classification, this project applies a Transformer model to a text-based problem: \*\*Fake News Classification\*\*.



The \*\*WELFake dataset\*\* was used, containing news articles labeled as either fake or real. A pretrained \*\*DistilBERT\*\* model was fine-tuned on this dataset and evaluated on a held-out test set.



\---



\## Dataset



The project uses the \*\*WELFake Dataset\*\*, which contains news articles with two classes:



| Label | Class | Samples |

|---|---|---:|

| 0 | Fake News | 35,028 |

| 1 | Real News | 37,067 |



After removing missing values and unnecessary columns, \*\*72,095 articles\*\* remained.



The `title` and `text` fields were combined into a single `content` field to provide the Transformer with the complete news article.



\---



\## Data Preparation



The dataset was divided using \*\*stratified sampling\*\* to preserve the class distribution.



| Dataset | Samples |

|---|---:|

| Train | 46,140 |

| Validation | 11,536 |

| Test | 14,419 |



The text was tokenized using the \*\*DistilBERT tokenizer\*\* with:



\- Maximum sequence length: \*\*256 tokens\*\*

\- Truncation: Enabled

\- Padding: Enabled



The tokenized data was then converted into Hugging Face `Dataset` objects.



\---



\## Transformer Model



A pretrained \*\*DistilBERT (`distilbert-base-uncased`)\*\* model was used for binary classification.



The model was configured with two output classes:



\- `0` → Fake News

\- `1` → Real News



The pretrained DistilBERT weights were loaded, while the classification head was newly initialized for the fake-news classification task.



\---



\## Training



The model was fine-tuned for \*\*3 epochs\*\* using the Hugging Face `Trainer`.



Main training configuration:



| Parameter | Value |

|---|---|

| Learning Rate | `2e-5` |

| Train Batch Size | `16` |

| Evaluation Batch Size | `16` |

| Epochs | `3` |

| Weight Decay | `0.01` |

| Optimizer | AdamW |



\### Training Results



| Epoch | Training Loss | Validation Loss |

|---|---:|---:|

| 1 | 0.039660 | 0.034847 |

| 2 | 0.011350 | \*\*0.030022\*\* |

| 3 | 0.001735 | 0.040547 |



The lowest validation loss was achieved at \*\*Epoch 2\*\*, so the best model was retained using `load\_best\_model\_at\_end=True`.



The training process took approximately \*\*59.15 minutes\*\* on an NVIDIA T4 GPU.



The increase in validation loss during Epoch 3 while training loss continued to decrease suggests signs of \*\*overfitting\*\*.



\---



\## Transformer Evaluation



The final model was evaluated on the held-out test set.



| Metric | Score |

|---|---:|

| Test Loss | \*\*0.02465\*\* |

| Test Accuracy | \*\*99.36%\*\* |

| Precision | \*\*99.50%\*\* |

| Recall | \*\*99.24%\*\* |

| F1-score | \*\*99.37%\*\* |



The Transformer achieved very strong performance on the WELFake test set, reaching \*\*99.36% accuracy\*\* and an \*\*F1-score of 99.37%\*\*.



\---



\## Attention vs. RNN Memory



Attention allows a model to look at all relevant parts of a sequence at the same time and assign different importance to each part, while an RNN processes the sequence step by step, carrying information forward through a hidden state as its memory.



This allows attention-based models to capture relationships between distant parts of a sequence more effectively than traditional step-by-step RNN memory.



\---



\## LSTM vs. Transformer



The Transformer was compared with the LSTM model developed during Day 3.



| Model | Dataset | Test Accuracy |

|---|---|---:|

| LSTM | ECG Heartbeat | 89.44% |

| DistilBERT | WELFake | \*\*99.36%\*\* |



The Transformer achieved \*\*9.92 percentage points higher accuracy\*\* than the Day 3 LSTM.



However, this is a \*\*high-level comparison rather than a controlled model comparison\*\*, because the models were trained on different datasets and solved different tasks. The LSTM classified ECG heartbeat signals, while DistilBERT classified fake and real news articles.



\---



\## Core Model



The \*\*Transformer (DistilBERT)\*\* will serve as the project's core model because it achieved \*\*99.36% test accuracy\*\* and \*\*99.37% F1-score\*\* on the WELFake dataset.



Its attention mechanism also makes it well suited for text classification because it can capture relationships between different parts of a text sequence without relying solely on step-by-step recurrent memory.



\---



\## Technologies



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- PyTorch

\- Hugging Face Transformers

\- Hugging Face Datasets

\- DistilBERT

\- Google Colab

\- NVIDIA T4 GPU



\---



\## Project Structure



```text

Day04/

│

├── Day04.ipynb

├── README.md

└── requirements.txt

