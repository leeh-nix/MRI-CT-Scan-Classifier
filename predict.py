from flask import Flask, request, jsonify

# import tensorflow as tf

import tensorflow.keras.utils as image
from PIL import Image
import numpy as np

from model import model

app = Flask(__name__)
IMAGE_SIZE = (152, 152)


def preprocess_image(image: Image.Image):
    """
    Preprocesses an image for prediction by resizing, converting to a numpy array,
    converting grayscale to RGB if needed, normalizing pixel values,
    and expanding dimensions for prediction.

    Parameters:
    image: Image.Image - The input image to be preprocessed.

    Returns:
    numpy.array - The preprocessed image ready for prediction.
    """
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    if len(image.shape) == 2:  # Grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    """
    A function to predict the class of an image based on the processed image data.

    Parameters:
    None

    Returns:
    JSON object containing the predicted class of the input image.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_class = int(prediction[0][0] > 0.5)

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
