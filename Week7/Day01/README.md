# Day 1 — CNN Architecture & Convolution Fundamentals

## Project Overview
This notebook kicks off **Sprint 2**, a new capstone phase focused on image classification. It introduces the dataset, justifies why a Convolutional Neural Network (CNN) is the right architecture for image data (as opposed to the dense network used in Sprint 1's house-price project), plans the CNN architecture, and builds intuition for convolution by hand-implementing a single edge-detection filter before any real training begins.

## Dataset
- **Source:** [Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images) (Kaggle)
- **Classes (3):** `Plastic`, `Paper`, `Garbage Bag`
- **Location (Colab):** `/content/drive/MyDrive/PC/Bag Classes`, one subfolder per class
- Per-class image counts were inspected by listing each class folder and counting its files.

## Workflow

### 1. Dataset & Architecture Decision
- Since the input is image data, a **CNN** was chosen over a standard dense network — CNNs are built to capture spatial patterns (edges, textures, shapes, object features) that a fully-connected network would struggle to learn efficiently.
- Planned CNN architecture for the classifier:

```text
Input Image
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (3, Softmax)
```
- The final `Dense(3, Softmax)` output layer matches the 3-class classification task (Plastic / Paper / Garbage Bag).

### 2. Data Access & Inspection
- Mounted Google Drive in Colab and located the dataset at `PC/Bag Classes`.
- Iterated over each class subfolder and printed the number of images per class to check for class balance before training.
- Loaded a single sample image via `tf.keras.utils.image_dataset_from_directory` (resized to 128×128) to use for the convolution demonstration below.

### 3. Manual Convolution: Edge Detection
- Converted the sample image to grayscale (channel-averaged).
- Defined a hand-crafted **3×3 vertical edge-detection kernel**:

```text
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]
```
- Applied zero-padding, then manually convolved the kernel across the image (nested loop, no framework) to produce a feature map.
- Visualized the original grayscale image alongside the resulting edge-detection feature map — stronger regions in the feature map correspond to areas of sharp pixel-intensity change, i.e. vertical edges and object boundaries.
- This demonstrates the core convolution operation used inside every `Conv2D` layer; in a real CNN, filters like this are **learned automatically** during training rather than hand-defined.

### 4. Why CNNs: Parameter Sharing
- Explained **parameter sharing**, the key efficiency property of CNNs: the same filter (and its weights) is reused at every spatial location in the image.
- **Worked example — a 3×3 filter on an RGB image:**
  `3 × 3 × 3 + 1 = 28` learnable parameters (including bias) — reused everywhere the filter is applied, so parameter count does **not** grow with image size.
- **Contrast — a dense layer on a 200×200×3 image with 64 neurons:**
  `200 × 200 × 3 × 64 + 64 = 7,680,064` parameters in the first layer alone.
- **Conclusion:** CNNs need far fewer parameters than dense layers for image inputs, thanks to parameter sharing and local connectivity — the same learned filter can detect a pattern (e.g. an edge) anywhere in the image.

## Key Findings
1. A CNN (Conv2D → MaxPooling2D ×2 → Flatten → Dense → Dropout → Softmax) is the planned architecture for this 3-class image classification task, chosen for its ability to learn spatial features efficiently.
2. A manually-applied 3×3 kernel successfully highlighted vertical edges in a sample image, illustrating what a `Conv2D` filter does before any learning occurs.
3. Parameter sharing makes CNNs dramatically more parameter-efficient than dense layers on image data (28 vs. ~7.68M parameters in the worked example) — the core reason CNNs scale to real image sizes.

## Tools Used
- Python, Jupyter Notebook (Google Colab)
- NumPy — manual convolution / edge-detection implementation
- Matplotlib — image and feature-map visualization
- TensorFlow / Keras — `image_dataset_from_directory` for loading sample images (full model training to follow in later notebooks)
- Google Colab `drive` — dataset access from Google Drive

## Files
- `Day01.ipynb` — full notebook: dataset/architecture decision, data access & class-count inspection, manual convolution edge-detection demo, parameter-sharing explanation
- Dataset (external, accessed via Google Drive): [Plastic - Paper - Garbage Bag Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images)
