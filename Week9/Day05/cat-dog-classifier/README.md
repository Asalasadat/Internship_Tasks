\# Cat vs Dog Classifier



A deep learning image classification application that uses a pre-trained MobileNetV2 architecture to classify images as either a cat or a dog.



\## Demo



The model is deployed using Gradio on Hugging Face Spaces.



\## Model



\- Architecture: MobileNetV2

\- Input Size: 224 × 224

\- Classes:

&#x20; - Cats

&#x20; - Dogs

\- Preprocessing: MobileNetV2 `preprocess\_input`



\## How to Use



1\. Upload a cat or dog image.

2\. The image is resized to 224 × 224.

3\. MobileNetV2 preprocessing is applied.

4\. The model predicts the class.

5\. The application displays the predicted class and confidence score.



\## Files



\- `app.py` — Gradio application and prediction logic.

\- `cat\_dog\_model.keras` — Trained TensorFlow/Keras model.

\- `class\_names.joblib` — Class names.

\- `preprocessing\_config.joblib` — Preprocessing configuration.

\- `requirements.txt` — Required Python dependencies.



\## Technologies



\- Python

\- TensorFlow / Keras

\- MobileNetV2

\- Gradio

\- Hugging Face Spaces

\- NumPy

\- Pillow

\- Joblib



\## Limitations



The model is intended for educational purposes and may not perform reliably on images that differ significantly from the training data.

