from drowsiness_detection.drowsiness_detection import detect_drowsiness_from_video
from drowsiness_detection.drowsiness_detection import detect_drowsiness_from_webcam
from speech_detection.speech_detection import monitor_speech

# Run speech monitoring for 30 seconds
monitor_speech(duration_seconds=30)
is_drowsy = detect_drowsiness_from_webcam( display=True)
print("Drowsiness Detected" if is_drowsy else "Driver was alert")
