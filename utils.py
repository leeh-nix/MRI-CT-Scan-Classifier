import numpy as np
import tensorflow as tf
import pydicom
from PIL import Image
from tensorflow.keras.models import load_model
import logging

model = load_model("model/ct_mri_classifier_5epochs.h5")  # type: ignore

IMAGE_SIZE = (152, 152)


# Function to convert DICOM to JPEG
def convert_dicom_to_jpg(dicom_path, jpg_path):
    """
    Convert a DICOM image to a JPEG image.

    Args:
        dicom_path (str): The path to the DICOM image file.
        jpg_path (str): The path to save the JPEG image.

    Returns:
        None
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array
        # Convert 16-bit grayscale to 8-bit
        img_scaled = np.uint8(img / np.max(img) * 255)  # Scale and convert to 8-bit
        img_mem = Image.fromarray(img_scaled)  # Create image from scaled array
        img_mem.save(jpg_path)
        logging.info(f"Converted DICOM {dicom_path} to JPEG {jpg_path}")
    except Exception as e:
        logging.error(f"Failed to convert DICOM to JPEG: {e}")
        raise


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
    try:
        image = image.resize(IMAGE_SIZE)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        logging.info("Image preprocessing successful")
        return image
    except Exception as e:
        logging.error(f"Failed to preprocess image: {e}")
        raise


# Function to classify image
def classify_image(image):
    """
    Classify an image based on the prediction from the model.

    Parameters:
        image (numpy.ndarray): The image to be classified.

    Returns:
        str: The classification of the image. Either "MRI" or "CT".
    """
    try:
        prediction = model.predict(image)
        logging.info("Image classification successful")
        return "MRI" if prediction[0][0] > 0.5 else "CT"
    except Exception as e:
        logging.error(f"Failed to classify image: {e}")
        raise
