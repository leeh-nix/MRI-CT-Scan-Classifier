from flask import Flask, request, jsonify
import logging
from PIL import Image
import os
import tempfile
from utils import convert_dicom_to_jpg, preprocess_image, classify_image

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            logging.error("No file provided")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if not file.filename.endswith(".dcm"):
            logging.error("File is not a DICOM file")
            return jsonify({"error": "File is not a DICOM file"}), 400

        # Save the DICOM file to a temporary location
        with tempfile.TemporaryDirectory() as tmpdirname:
            dicom_path = os.path.join(tmpdirname, file.filename)
            file.save(dicom_path)
            logging.info(f"File saved to {dicom_path}")

            # Convert the DICOM file to JPEG
            jpg_path = dicom_path.replace(".dcm", ".jpg")
            convert_dicom_to_jpg(dicom_path, jpg_path)
            logging.info(f"DICOM converted to {jpg_path}")

            image = Image.open(jpg_path)

            processed_image = preprocess_image(image)
            result = classify_image(processed_image)

            return jsonify({"classification": result})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
