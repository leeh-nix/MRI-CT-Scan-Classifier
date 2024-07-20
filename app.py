from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow.keras.utils as image
from PIL import Image
import os
import logging

from utils import convert_dicom_to_jpg, preprocess_image, classify_image

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    A function that predicts the classification of an image received via POST request.
    It saves the received DICOM file, converts it to JPEG, preprocesses the image, classifies it,
    and returns the classification result. Finally, it cleans up temporary files.

    Parameters:
        None

    Returns:
        JSON: A JSON response containing the classification result.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".dcm"):
        return jsonify({"error": "File is not a DICOM file"}), 400

    # Save the DICOM file to a temporary location
    dicom_path = os.path.join("/tmp", file.filename)
    file.save(dicom_path)

    # Convert the DICOM file to JPEG
    jpg_path = dicom_path.replace(".dcm", ".jpg")
    convert_dicom_to_jpg(dicom_path, jpg_path)

    image = Image.open(jpg_path)

    processed_image = preprocess_image(image)

    result = classify_image(processed_image)

    # Clean up temporary files
    os.remove(dicom_path)
    os.remove(jpg_path)

    return jsonify({"classification": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
