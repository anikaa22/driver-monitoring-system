import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model (relative to current file)
model_path = os.path.join(os.path.dirname(__file__), 'mobilenetv2_phone_usage_balanced.h5')
model = tf.keras.models.load_model(model_path)

# Class labels in training order
class_labels = ['PhoneUsage', 'Safe']

# Input image size
img_size = (224, 224)

def detect_phone_usage(frame):
    """
    Takes a single video frame, preprocesses it, and returns:
    - predicted label ('PhoneUsage' or 'Safe')
    - confidence (float between 0 and 1)
    """
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[predicted_class])
    label = class_labels[predicted_class]

    return label, confidence, label == "PhoneUsage" and confidence > 0.95

