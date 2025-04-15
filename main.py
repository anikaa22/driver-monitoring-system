import cv2
import numpy as np

from drowsiness_detection.drowsiness_detection import detect_drowsiness
from phone_usage_detection.phone_detection import detect_phone_usage
from speech_detection.speech_detection import detect_speech
from drinking_detection.drinking_detection import detect_drinking

# Open webcam (0) or replace with video path
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame if needed
    frame = cv2.resize(frame, (640, 480))

    # Call all detection functions
    drowsy = detect_drowsiness(frame)
    phone = detect_phone_usage(frame)
    speech = detect_speech(frame)
    drinking = detect_drinking(frame)

    # Print results (or display them on frame)
    print(f"Drowsy: {drowsy}, Phone: {phone}, Speech: {speech}, Drinking: {drinking}")

    # Display the video
    cv2.imshow("Driver Monitoring", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
