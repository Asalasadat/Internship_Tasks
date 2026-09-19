import gradio as gr
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model and preprocessing files
model = tf.keras.models.load_model("cat_dog_model.keras")
class_names = joblib.load("class_names.joblib")
preprocessing_config = joblib.load("preprocessing_config.joblib")


def predict(image):
    # Preprocess image
    image = image.convert("RGB")
    image = image.resize(tuple(preprocessing_config["image_size"]))

    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return {
        predicted_class: confidence
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog image"),
    outputs=gr.Label(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a cat or a dog."
)

demo.launch()