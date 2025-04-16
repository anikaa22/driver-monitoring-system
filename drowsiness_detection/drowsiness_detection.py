import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import warnings

warnings.filterwarnings('ignore')

# EAR and MAR thresholds
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6

# Score threshold to label drowsiness
SCORE_THRESHOLD = 5

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def process_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    eye_flag = mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]

        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth = np.array(landmarks['bottom_lip'])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            eye_flag = True
        if mar > MOUTH_AR_THRESH:
            mouth_flag = True

    return eye_flag, mouth_flag

def detect_drowsiness(frame, count, score, display=False):
    """
    Processes a single frame and updates the score.

    Args:
        frame (ndarray): The video frame to process.
        count (int): Frame count.
        score (int): Current drowsiness score.
        display (bool): Whether to display the frame.

    Returns:
        score (int): Updated score.
        drowsy (bool): Whether the person is considered drowsy.
        frame (ndarray): The annotated frame (only useful if display=True).
    """
    drowsy = False
    frame = cv2.resize(frame, (800, 500))

    if count % 5 == 0:
        eye_flag, mouth_flag = process_image(frame)
        if eye_flag or mouth_flag:
            score += 1
        else:
            score = max(score - 1, 0)

    if display:
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, f"Score: {score}", (10, frame.shape[0] - 10), font, 1, (20,255,0), 2)
        if score >= SCORE_THRESHOLD:
            cv2.putText(frame, "Drowsy", (frame.shape[1] - 130, 40), font, 1, (10,10,255), 2)
            drowsy = True

    return score, drowsy, frame
