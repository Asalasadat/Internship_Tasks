# Day 5 — Day 5 — Full Evaluation, Explainability & Sprint Review

This notebook completes a cat-versus-dog image-classification workflow using **transfer learning**, full test-set evaluation, inference-pipeline verification, error analysis, and SHAP-based explainability. The classifier uses a frozen **MobileNetV2** backbone with ImageNet weights and a small binary-classification head.

## Project Summary

| Item | Description |
|---|---|
| Task | Binary image classification: cats versus dogs |
| Dataset | Kaggle `tongpython/cat-and-dog` dataset [1] |
| Model | MobileNetV2 with ImageNet weights and a custom classification head |
| Input size | `224 × 224 × 3` |
| Classes | `cats = 0`, `dogs = 1` |
| Training epochs | 5 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Reported test accuracy | 99.0% |
| Saved model | `cat_dog_model.h5` |

## Learning Objectives

The notebook demonstrates how to:

1. Download and inspect an image dataset with `kagglehub`.
2. Read, resize, convert, and normalize images with OpenCV.
3. Generate augmented image examples with Keras `ImageDataGenerator`.
4. Apply the preprocessing expected by MobileNetV2.
5. Train a binary classifier with a frozen pre-trained convolutional base.
6. Build a reusable prediction function that returns a class and confidence score.
7. Verify that inference-time preprocessing matches training-time preprocessing.
8. Evaluate predictions with a confusion matrix and classification report.
9. Inspect misclassified images.
10. Explain model behavior with SHAP image explanations.

## Dataset Structure

The notebook expects the downloaded dataset to contain the following directory structure:

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

The notebook documents the following split:

| Split | Images |
|---|---:|
| Training | 7,205 |
| Validation | 800 |
| Test | 2,023 |

The test set is approximately balanced, with 1,011 cat images and 1,012 dog images. Because the classes are balanced, SMOTE or class weighting was not required.

## Environment and Dependencies

The notebook is designed for Google Colab or a Kaggle-style Python environment with access to TensorFlow and the Kaggle dataset. The main dependencies are:

| Package | Purpose |
|---|---|
| `kagglehub` | Download the dataset |
| `tensorflow` | Build, train, and evaluate the model |
| `opencv-python` | Demonstrate image preprocessing |
| `matplotlib` | Display images and plots |
| `numpy` | Numerical and array operations |
| `scikit-learn` | Classification metrics and confusion matrix |
| `seaborn` | Confusion-matrix visualization |
| `shap` | Model explainability |

If the final notebook cell is executed, it writes the environment packages to `requirements.txt` with:

```python
!pip freeze > requirements.txt
```

A typical local installation is:

```bash
pip install kagglehub tensorflow opencv-python matplotlib numpy scikit-learn seaborn shap jupyter
```

> TensorFlow installation may differ by operating system and hardware. For GPU use, follow the current TensorFlow installation guidance [2].

## Running the Notebook

1. Open `Day05.ipynb` in Google Colab, Kaggle Notebooks, or a local Jupyter environment.
2. Run the dataset-download cell. The notebook uses:

   ```python
   path = kagglehub.dataset_download("tongpython/cat-and-dog")
   ```

3. Confirm that `path` contains `training_set/training_set` and `test_set/test_set`.
4. Run the preprocessing and augmentation cells to inspect the image transformations.
5. Run the MobileNetV2 training cell. This creates the training, validation, and test generators, trains the model for five epochs, evaluates it, and saves `cat_dog_model.h5`.
6. Run the corrected prediction, evaluation, error-analysis, and SHAP cells.
7. Execute the final cell if a reproducible package snapshot is needed.

The notebook contains hard-coded sample paths such as:

```text
/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1.jpg
```

When running outside Kaggle, replace these sample paths with paths under the downloaded `path` variable.

## Image Preprocessing

The notebook demonstrates two preprocessing approaches.

### OpenCV demonstration

The first preprocessing function uses OpenCV to read an image, resize it to `128 × 128`, convert BGR to RGB, and scale pixel values from `[0, 255]` to `[0, 1]`. This section is useful for understanding common image-preprocessing operations, but it is not the final preprocessing pipeline used by the trained MobileNetV2 classifier.

### MobileNetV2 training pipeline

The training and validation generators use MobileNetV2's `preprocess_input` function. Training images additionally receive light augmentation:

| Transformation | Configuration |
|---|---|
| Rotation | Up to 20 degrees |
| Zoom | Up to 15% |
| Horizontal flip | Enabled |
| Brightness | Range `[0.8, 1.2]` |
| Validation split | 10% of the training directory |

The test generator applies only the MobileNetV2 preprocessing function and does not use augmentation.

## Model Architecture

The model uses MobileNetV2 as a frozen feature extractor:

```text
Input: 224 × 224 × 3 image
        ↓
MobileNetV2 convolutional base, ImageNet weights, frozen
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.2)
        ↓
Dense(1, activation="sigmoid")
        ↓
Cat/dog probability
```

Only the classification head is trained. The notebook reports 1,281 trainable parameters out of 2,259,265 total parameters. The sigmoid output is converted to a class using a threshold of `0.5`.

## Training Results

The notebook reports the following results:

| Metric | Result |
|---|---:|
| Final training accuracy | 98.2% |
| Final validation accuracy | 98.5% |
| Test accuracy | **99.0%** |
| Test loss | 0.031 |

The model reached more than 95% accuracy during the first epoch. Validation accuracy remained close to 98–98.5% during training, which suggests that the frozen MobileNetV2 features provided a strong starting point without clear evidence of substantial overfitting in this experiment.

## Inference and Preprocessing Consistency

The notebook initially defines a `predict()` function using OpenCV. That implementation resizes images with OpenCV's default bilinear interpolation, while the training pipeline uses Keras image loading and its default interpolation behavior.

The notebook then corrects this difference by redefining `predict()` with the same Keras utilities used during training:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

CLASS_NAMES = {v: k for k, v in train_generator.class_indices.items()}

def predict(image_path, size=(224, 224)):
    image = load_img(image_path, target_size=size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    probability = model.predict(image_array, verbose=0)[0][0]
    predicted_index = int(probability > 0.5)

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": float(
            probability if predicted_index == 1 else 1 - probability
        ),
    }
```

**Use the corrected Keras-based implementation for inference.** Reusing the same image loader, resize behavior, array conversion, and normalization reduces avoidable differences between evaluation and real-world predictions.

## Evaluation

The test-set evaluation reports balanced performance:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cats | 1.00 | 0.98 | 0.99 | 1,011 |
| Dogs | 0.98 | 1.00 | 0.99 | 1,012 |
| Overall | — | — | **0.99** | 2,023 |

The documented confusion-matrix counts are:

| Outcome | Count | Interpretation |
|---|---:|---|
| True negative | 994 | Cats correctly predicted as cats |
| False positive | 17 | Cats predicted as dogs |
| False negative | 3 | Dogs predicted as cats |
| True positive | 1,009 | Dogs correctly predicted as dogs |

The most common documented error is a false positive, where a cat is classified as a dog. The evaluation cells reproduce the confusion matrix, classification report, and error-type comparison with `scikit-learn`.

## Misclassified Images

The notebook displays three misclassified examples and categorizes the likely causes. The documented causes include visual similarity between classes, poor lighting, partial occlusion, unusual poses, and appearances that are underrepresented in the training data.

These examples should be treated as qualitative diagnoses rather than definitive causal explanations. A stronger analysis would compare more misclassified samples, inspect confidence scores, and use a fixed annotation protocol for error categories.

## SHAP Explainability

The notebook uses `shap.GradientExplainer` to generate image explanations. It produces:

- A batch-level visualization intended to show image regions that influence predictions across sampled test images.
- A single-image explanation intended to show which regions contribute to one prediction.

SHAP explanations support interpretation, but they do not prove that the model uses semantically correct features. Explanations should be reviewed together with the original image, predicted probability, true label, and model performance [3].

## Baseline Comparison Note

The notebook includes a comparison with a reported Week 6 baseline:

| Metric | Week 6 Baseline | Final Model |
|---|---:|---:|
| MAE | 242,663.60 | **120,246.70** |
| RMSE | 410,080.49 | **209,799.39** |
| R² | -0.0009 | **0.7380** |

MAE, RMSE, and R² are regression metrics, while the main model in this notebook is a binary image classifier evaluated with accuracy, precision, recall, F1-score, and a confusion matrix. Therefore, this comparison should be validated before being used as a formal comparison for the cat-versus-dog classifier. The classification metrics are the primary metrics for this notebook.

## Generated Artifacts

Running the notebook can produce the following artifacts:

| Artifact | Description |
|---|---|
| `cat_dog_model.h5` | Trained Keras model in HDF5 format |
| `requirements.txt` | Package snapshot created by the final cell |
| Confusion-matrix plot | Test-set classification visualization |
| Augmentation grid | Examples of transformed training images |
| SHAP plots | Global and individual image explanations |

The `.h5` format is supported but considered a legacy Keras format. For new projects, prefer the native `.keras` format when supported:

```python
model.save("cat_dog_model.keras")
```

## Limitations and Next Steps

The notebook is an evaluation and explainability exercise rather than a production deployment. The next sprint recommendation is to expose the model through **Streamlit** or **FastAPI**, allowing a user to upload an image and receive the predicted class and confidence.

Additional improvements could include fine-tuning selected MobileNetV2 layers, adding reproducible random seeds, saving the class mapping with the model, tracking training curves, validating confidence calibration, and testing on images from outside the original dataset. A production implementation should also replace hard-coded paths with configuration values and add input validation for unsupported files.

## References

[1]: https://www.kaggle.com/datasets/tongpython/cat-and-dog "Cat and Dog Image Dataset on Kaggle"

[2]: https://www.tensorflow.org/install "TensorFlow Installation Guide"

[3]: https://shap.readthedocs.io/en/latest/ "SHAP Documentation"

---

**Notebook:** `Day05.ipynb`  
