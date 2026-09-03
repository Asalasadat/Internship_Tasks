# Day 3 — RNNs & LSTMs for ECG Heartbeat Classification

## Overview

This project applies **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks to **5-class ECG heartbeat classification**.

The goal is to evaluate how sequence-aware models learn patterns from ECG signals while preserving their original order.

The **LSTM achieved 89.44% test accuracy**, outperforming the Plain RNN by **6.68 percentage points**.

---

## Dataset

**ECG Heartbeat Categorization Dataset**

[Kaggle Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

Used files:

- `mitbih_train.csv`
- `mitbih_test.csv`

Each sample contains **187 ECG signal values** and one target label representing one of five heartbeat classes.

### Dataset Dimensions

| Dataset | Samples | Columns |
|---|---:|---:|
| Training | 87,553 | 188 |
| Test | 21,891 | 188 |

---

## Problem Definition

This is a **multi-class classification problem** with five ECG heartbeat categories.

- **187 columns** → ECG signal sequence
- **1 column** → target label

---

## Data Preprocessing

The training data was divided using an **80/20 stratified split**:

| Split | Samples |
|---|---:|
| Training | 70,042 |
| Validation | 17,511 |
| Test | 21,891 |

The ECG sequences were reshaped into:

```text
(samples, 187, 1)
```

This represents each heartbeat as **187 time steps with one feature per step**.

---

## LSTM Architecture

```text
Input (187, 1)
      ↓
LSTM (64 units)
      ↓
Dropout (30%)
      ↓
Dense (32, ReLU)
      ↓
Dense (5, Softmax)
```

### Configuration

| Component | Configuration |
|---|---|
| LSTM | 64 units |
| Dropout | 0.3 |
| Dense | 32 units, ReLU |
| Output | 5 units, Softmax |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Parameters | 19,141 |

---

## Training

The LSTM was trained for **10 epochs** with a **batch size of 128**.

| Metric | Result |
|---|---:|
| Training Accuracy | 90.12% |
| Validation Accuracy | 89.65% |
| Validation Loss | 0.3771 |
| Training Time | 79.61 sec |

---

## Test Results

| Metric | Result |
|---|---:|
| Test Accuracy | **89.44%** |
| Test Loss | **0.3882** |

Validation and test accuracy differed by only **0.21 percentage points**.

---

## Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Normal | 0.93 | 0.97 | 0.95 |
| Supraventricular | 0.23 | 0.06 | 0.09 |
| Ventricular | 0.66 | 0.43 | 0.52 |
| Fusion | 0.00 | 0.00 | 0.00 |
| Unknown | 0.71 | 0.83 | 0.76 |
| **Macro Avg** | **0.50** | **0.46** | **0.46** |
| **Weighted Avg** | **0.87** | **0.89** | **0.88** |

The model performed substantially better on the **Normal** class, while minority classes were more challenging. The **Fusion** class received no correct predictions in this experiment.

---

## LSTM vs Plain RNN

Both models were trained using the same general experimental setup.

| Model | Test Accuracy | Test Loss | Training Time |
|---|---:|---:|---:|
| Plain RNN | 82.76% | 0.6595 | 83.64 sec |
| **LSTM** | **89.44%** | **0.3882** | **79.61 sec** |

### Improvement

**LSTM accuracy improvement: +6.68 percentage points**

```text
Plain RNN: 82.76%
LSTM:      89.44%
```

---

## Why Sequence Order Matters

ECG signals are sequential, meaning the order of signal values contains information about the shape and structure of a heartbeat.

The LSTM processes the signal in its original order and can retain information from earlier time steps while processing later ones.

The results support the benefit of sequence-aware modeling:

- Plain RNN: **82.76%**
- LSTM: **89.44%**
- Improvement: **6.68 percentage points**

---

## Key Findings

- **89.44%** test accuracy achieved by the LSTM.
- LSTM outperformed the Plain RNN by **6.68 percentage points**.
- LSTM achieved lower test loss.
- Validation and test performance were closely aligned.
- Performance varied significantly across classes.
- The Fusion class was particularly challenging.
- Preserving ECG sequence order was beneficial in this experiment.

---

## Technologies

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `TensorFlow` · `Keras`

---

## Project Structure

```text
Day03/
├── Day03.ipynb
├── README.md
└── requirements.txt
```

> The dataset is not included in the repository due to its size. Download it from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

---

## Reproducibility

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
2. Place `mitbih_train.csv` and `mitbih_test.csv` in the working directory.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open `Day03.ipynb` using **Jupyter Notebook** or **Google Colab**.
5. Run the notebook cells in order.

---

## Conclusion

Under the same experimental setup, the **LSTM achieved 89.44% test accuracy**, compared with **82.76% for the Plain RNN**.

The results indicate that LSTM's ability to model dependencies across ECG sequences was beneficial for this classification task.
