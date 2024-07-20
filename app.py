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
    """
    A function that predicts the classification of an image file uploaded through a POST request.
    Checks the file type, converts DICOM files to JPEG, preprocesses the image, classifies it, and returns the classification result.
    Handles errors by returning appropriate error responses.
    """
    try:
        if "file" not in request.files:
            logging.error("No file provided")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        # Check if the file is a DICOM file
        if filename.endswith(".dcm"):
            logging.info("File is a DICOM file")
            with tempfile.TemporaryDirectory() as tmpdirname:
                dicom_path = os.path.join(tmpdirname, file.filename)
                file.save(dicom_path)
                logging.info(f"File saved to {dicom_path}")

                # Convert the DICOM file to JPEG
                jpg_path = dicom_path.replace(".dcm", ".jpg")
                convert_dicom_to_jpg(dicom_path, jpg_path)
                logging.info(f"DICOM converted to {jpg_path}")

                image = Image.open(jpg_path)
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            logging.info("File is an image file")
            # Save the image file to a temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(filename)[1]
            ) as tmpfile:
                tmpfile.write(file.read())
                tmpfile_path = tmpfile.name
            logging.info(f"Image file saved to {tmpfile_path}")

            image = Image.open(tmpfile_path)
        else:
            logging.error("Unsupported file type")
            return jsonify({"error": "Unsupported file type"}), 400

        processed_image = preprocess_image(image)
        result = classify_image(processed_image)

        # Clean up temporary files
        if filename.endswith((".jpg", ".jpeg", ".png")):
            os.remove(tmpfile_path)

        return jsonify({"classification": result})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
