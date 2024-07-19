import tensorflow as tf

model = tf.keras.saving.load_model("model/ct_mri_classifier_5epochs.h5")  # type: ignore
