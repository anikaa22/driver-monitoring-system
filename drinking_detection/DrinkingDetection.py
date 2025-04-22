import tensorflow as tf
import numpy as np
import cv2

import os

model_path = os.path.join(os.path.dirname(__file__), 'drinking_detection_model.h5')
model = tf.keras.models.load_model(model_path)

img_size = (224, 224)
class_labels = ['Drinking', 'Safe']

def detect_drinking(frame):
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    label = class_labels[predicted_class]

    return label == "Drinking" and confidence > 0.95
