# Technical Writeup — Cat vs Dog Classifier

## 1. Problem

The goal of this project is to build an image classification system that identifies whether an uploaded image contains a cat or a dog.

The project demonstrates the complete machine learning pipeline from data preprocessing and model training to evaluation and deployment.

## 2. Dataset

The project uses a Cat vs Dog image dataset.

The dataset contains two classes:

- Cats
- Dogs

The images were preprocessed and resized to 224 × 224 pixels to match the expected input size of MobileNetV2.

## 3. Methodology

The project uses MobileNetV2 as the main deep learning architecture.

The image preprocessing pipeline includes:

1. Convert the image to RGB.
2. Resize the image to 224 × 224.
3. Apply MobileNetV2 `preprocess_input`.
4. Add the batch dimension.
5. Pass the processed image to the trained model.

The model returns prediction probabilities for the two classes.

## 4. Model Evaluation

The model was evaluated using a separate test dataset.

The final evaluation achieved approximately:

| Metric | Result |
|---|---:|
| Accuracy | 0.99 |
| Macro F1-score | 0.99 |
| Weighted F1-score | 0.99 |

The model achieved approximately 99% accuracy on the test set.

## 5. Deployment

The trained model was saved as a Keras model:

`cat_dog_model.keras`

Additional preprocessing artifacts were saved using Joblib:

- `class_names.joblib`
- `preprocessing_config.joblib`

The model was deployed using:

- Gradio
- Hugging Face Spaces
- TensorFlow/Keras

The deployed application allows users to upload an image and receive the predicted class and confidence score.

## 6. Inference Pipeline

The deployment pipeline is:

User Image  
↓  
RGB Conversion  
↓  
Resize to 224 × 224  
↓  
MobileNetV2 Preprocessing  
↓  
Trained Model  
↓  
Prediction  
↓  
Class + Confidence

## 7. Limitations

The model is intended primarily for educational purposes.

Performance may decrease when the uploaded images differ significantly from the training data, such as images with unusual lighting, backgrounds, image quality, or objects that are not clearly cats or dogs.

## 8. Future Improvements

Possible improvements include:

- Adding more diverse training images.
- Applying additional data augmentation.
- Testing the model on a larger independent dataset.
- Improving the user interface.
- Monitoring model performance after deployment.
