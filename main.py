import cv2
import time
import pyaudio
import sqlite3
from db import init_db, start_ride, log_infraction

# Initialize DB and session
init_db()
driver_id = 1  # Assume logged-in driver ID is known
ride_id = start_ride(driver_id)
from drinking_detection.DrinkingDetection import detect_drinking
from phone_usage_detection.phone_detection import detect_phone_usage
from speech_detection.speech_detection import SpeechDetector
from drowsiness_detection.drowsiness_detection import detect_drowsiness

# Initialize detectors
speech_detector = SpeechDetector()

# Drowsiness detection tracking
frame_count = 0
drowsiness_score = 0

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# PyAudio setup
RATE = 16000
CHUNK = int(RATE * 0.03)  # 30ms chunks
FORMAT = pyaudio.paInt16
CHANNELS = 1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Driver Monitoring System Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ---- Drowsiness detection ----
    drowsiness_score, is_drowsy, processed_frame = detect_drowsiness(frame, frame_count, drowsiness_score, display=True)
    if is_drowsy:
        log_infraction(ride_id, "Drowsiness", "Driver is drowsy")

    # ---- Drinking detection ----
    drinking = detect_drinking(frame)
    if drinking:
        drinking = detect_drinking(frame)
        log_infraction(ride_id, "Drinking", "Drinking detected")
        cv2.putText(processed_frame, "Drinking Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---- Phone usage detection ----
    label, confidence, phone_flag = detect_phone_usage(frame)
    if phone_flag:
        log_infraction(ride_id, "Phone Usage", f"Phone detected with label: {label}, confidence: {confidence:.2f}")
        cv2.putText(processed_frame, "Phone Usage", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ---- Speech detection ----
    try:
        audio_frame = stream.read(CHUNK, exception_on_overflow=False)
        talking, speech_alert = speech_detector.process_audio_frame(audio_frame)
        if speech_alert:
            log_infraction(ride_id, "Speech", "Driver talking excessively")
            cv2.putText(processed_frame, "Talking Too Long!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    except Exception as e:
        print("Audio error:", e)

    # ---- Display output ----
    cv2.imshow("Driver Monitoring System", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()

def show_ride_infractions(ride_id):
    conn = sqlite3.connect('driver_monitoring.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM infractions WHERE ride_id = ?", (ride_id,))
    rows = cursor.fetchall()
    conn.close()
    print("\n--- Infractions Logged ---")
    for row in rows:
        print(row)


show_ride_infractions(ride_id)

