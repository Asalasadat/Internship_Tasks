# Day 2 — Building CNNs & Transfer Learning

## Project Overview
This notebook continues **Sprint 2** (image classification). It recaps the Day 1 CNN architecture plan and edge-detection demo, then moves into the core of Day 2: building the dataset pipeline, training a CNN from scratch, applying data augmentation and EarlyStopping to fight overfitting, and comparing that against transfer learning with a frozen pre-trained MobileNetV2.

## Dataset
- **Source:** [Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images) (Kaggle)
- **Size:** 15,000 images across 3 classes
- **Classes:** `Garbage Bag`, `Paper`, `Plastic`
- **Location (Colab):** `/content/Bag Classes` (copied from Google Drive), one subfolder per class
- **Split:** 70% train (10,500 images) / 30% held out, then split evenly into 15% validation / 15% test

## Workflow

### 1. Recap (Day 1)
- Restated the dataset and the decision to use a CNN over a dense network, since CNNs capture spatial patterns (edges, textures, shapes) far more efficiently than fully-connected layers.
- Re-ran the hand-defined 3×3 vertical edge-detection kernel on a sample image to demonstrate what a `Conv2D` filter does before any learning occurs.
- Re-confirmed the parameter-sharing math: a 3×3 filter on an RGB image needs only 28 parameters (reused everywhere), versus 7,680,064 parameters for a single dense layer with 64 neurons on a 200×200×3 image.

### 2. Data Pipeline
- Loaded images via `tf.keras.utils.image_dataset_from_directory`, resized to 128×128, batch size 32.
- Built the training set (70%) and a combined validation+test pool (30%), then split that pool evenly (via `cardinality`) into a 15% validation set and a 15% test set.
- Confirmed the 3 class names: `Garbage Bag Images`, `Paper Bag Images`, `Plastic Bag Images`.

### 3. CNN Trained From Scratch
Built and trained the planned architecture:

```text
Input(128,128,3)
    ↓
Conv2D(32, 3×3, ReLU, same padding) → MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU, same padding) → MaxPooling2D(2×2)
    ↓
Flatten → Dense(64, ReLU) → Dropout(0.3) → Dense(3, Softmax)
```
- Compiled with Adam optimizer, sparse categorical cross-entropy loss, 30 epochs.

**Results:**
- **Best Validation Accuracy:** 82.05% (epoch 13)
- **Training Time:** 295.77 seconds (~4.93 min)
- Training accuracy reached ~97% while validation stayed at ~82% — a clear sign of **overfitting**.

### 4. Data Augmentation
Rebuilt the same architecture with an augmentation + rescaling stage prepended:

```text
RandomFlip("horizontal") → RandomRotation(0.1) → RandomZoom(0.1) → Rescaling(1./255)
```
- Trained for 30 epochs with the same compile settings.

**Results:**
- **Best Validation Accuracy:** 95.80% (epoch 22, val loss 0.1174) — later re-measured at **96.25%** after adding EarlyStopping (see below)
- Improvement over the non-augmented CNN: **+13.75 to +14.20 percentage points**
- Training and validation accuracy stayed much closer together, indicating reduced overfitting.

### 5. EarlyStopping
Re-trained the augmented model with:
```python
EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
```
- Training stopped automatically once validation loss stopped improving, and `restore_best_weights=True` kept the weights from the best epoch rather than the last one.
- Plotted validation accuracy curves for "With Augmentation" vs. "Without Augmentation" — the augmented run is consistently higher and more stable.

**Final comparison:**

| Model | Best Validation Accuracy |
|---|---:|
| Without Augmentation | 82.05% |
| With Augmentation (+ EarlyStopping) | **96.25%** |
| **Improvement** | **+14.20 pp** |

### 6. Transfer Learning (MobileNetV2)
Built a transfer-learning model using a frozen pre-trained **MobileNetV2** (ImageNet weights, `include_top=False`):

```text
Input(128,128,3) → preprocess_input → MobileNetV2 (frozen) → GlobalAveragePooling2D
    ↓
Dropout(0.3) → Dense(3, Softmax)
```
- Base model weights frozen (`base_model.trainable = False`); only the new head was trained.
- Trained for 30 epochs with the same Adam / sparse categorical cross-entropy setup.

**Results:**
- **Best Validation Accuracy:** 97.28% (epoch 27, val loss 0.0780)
- **Training Time:** 321.75 seconds (~5.36 min)
- Small gap between training and validation accuracy — good generalization, limited overfitting.

## Final Model Comparison

| Model | Best Validation Accuracy | Training Time |
|---|---:|---:|
| CNN from scratch (no augmentation) | 82.05% | 295.77 s (~4.93 min) |
| Transfer Learning (MobileNetV2) | **97.28%** | 321.75 s (~5.36 min) |

Transfer Learning achieved **15.22 percentage points higher** validation accuracy than the CNN trained from scratch, at the cost of only ~26 seconds more training time — making it the clear winner overall.

## Key Findings
1. A CNN trained from scratch overfits noticeably on this dataset (97% train vs. 82% validation accuracy) without additional regularization.
2. **Data augmentation** (flip, rotation, zoom) closed most of that gap, lifting validation accuracy from 82.05% to 96.25% — the single biggest improvement in the notebook.
3. **EarlyStopping** with `restore_best_weights=True` protected against overfitting further by halting training and keeping the best-performing weights rather than the final epoch's.
4. **Transfer learning** with a frozen MobileNetV2 backbone outperformed the best from-scratch CNN (97.28% vs. 96.25%), for a modest extra training-time cost — the strongest option of the three approaches tested.

## Tools Used
- Python, Jupyter Notebook (Google Colab)
- NumPy — manual edge-detection convolution (Day 1 recap)
- Matplotlib — image visualization, accuracy curve comparison plots
- TensorFlow / Keras — `image_dataset_from_directory`, `Sequential`, `Conv2D`, `MaxPooling2D`, `Dropout`, data augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`), `EarlyStopping`, `MobileNetV2` (transfer learning)
- Google Colab `drive` — dataset access from Google Drive

## Files
- `Day02.ipynb` — full notebook: Day 1 recap, dataset pipeline & split, CNN from scratch, data augmentation, EarlyStopping, MobileNetV2 transfer learning, final model comparison
- `requirements.txt` — exact library versions for reproducibility (generated via `pip freeze`)
- Dataset (external, accessed via Google Drive): [Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images)
