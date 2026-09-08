# Day 3 — Computer Vision Preprocessing with OpenCV 

## Overview

This day focuses on preparing image data for machine learning and deep learning models. Using the **Cats vs Dogs** dataset, an image preprocessing pipeline was built with OpenCV, followed by data augmentation and preprocessing for a pre-trained MobileNetV2 model.

## Dataset

**Cats vs Dogs** dataset from Kaggle.

The dataset contains two classes:

* `cats`
* `dogs`

### Dataset Structure

```text
cat-and-dog/
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/
```

## Tasks Completed

### 1. OpenCV Image Preprocessing

A preprocessing function was created using OpenCV to:

* Read images
* Resize images to `128 × 128`
* Convert images from BGR to RGB
* Normalize pixel values from `0–255` to `0–1`

### 2. Data Augmentation

An augmentation pipeline was created using TensorFlow/Keras.

The pipeline applies:

* Random rotation
* Random zoom
* Horizontal flipping
* Brightness changes

Several augmented versions of one image were visualized to verify the transformations.

### 3. Pre-trained Model Preprocessing

**MobileNetV2** was selected as the pre-trained model.

The MobileNetV2-specific `preprocess_input` function was used to prepare images according to the preprocessing expected by the pre-trained model.

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Matplotlib
* Kaggle Dataset
* MobileNetV2

## Key Learning Outcomes

* Understand why image preprocessing is necessary.
* Standardize image size and color format.
* Normalize image pixel values.
* Apply data augmentation to increase training diversity.
* Understand the importance of model-specific preprocessing.
* Prepare an image pipeline for transfer learning.

## Mentor Review

The integrated notebook was prepared for the mid-sprint Mentor Code & Notebook Review.

Feedback from the mentor was addressed through subsequent updates to the notebook.
