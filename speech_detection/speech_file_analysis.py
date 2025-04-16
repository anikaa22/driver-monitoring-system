import webrtcvad
import numpy as np
import wave
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os

# Make sure ffmpeg path is set for pydub
AudioSegment.converter = r"C:\Users\Anika\ffmpeg\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin\ffmpeg.exe"

# Constants
FRAME_DURATION_MS = 30
RATE = 16000
FRAME_SIZE = int(RATE * (FRAME_DURATION_MS / 1000))
ALERT_THRESHOLD = 100
BUFFER_SIZE = 100

def get_audio_energy(frame):
    audio_data = np.frombuffer(frame, dtype=np.int16)
    return np.abs(audio_data).mean()

def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, fps=RATE, verbose=False, logger=None)

def load_audio(path):
    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(RATE).set_sample_width(2)
    return sound.raw_data

def run_speech_detection(audio_data, rate=RATE):
    vad = webrtcvad.Vad(1)

    talk_time = 0
    silence_time = 0
    speech_buffer = []
    alert_triggered = False
    talking = False
    silence_announced = False

    print("🔍 Analyzing audio...")

    for i in range(0, len(audio_data), FRAME_SIZE * 2):
        frame = audio_data[i:i + FRAME_SIZE * 2]
        if len(frame) < FRAME_SIZE * 2:
            break

        is_speech = vad.is_speech(frame, rate)
        energy = get_audio_energy(frame)
        if energy < 500:
            is_speech = False

        speech_buffer.append(is_speech)
        if len(speech_buffer) > BUFFER_SIZE:
            speech_buffer.pop(0)

        if sum(speech_buffer) > BUFFER_SIZE // 3:
            if not talking:
                print("Driver is talking...")
            talking = True
            talk_time += 1
            silence_time = 0
            silence_announced = False
        else:
            if not talking and not silence_announced and silence_time > 100:
                print("Silence detected.")
                silence_announced = True
            talking = False
            silence_time += 1
            if silence_time > BUFFER_SIZE:
                talk_time = 0
                alert_triggered = False

        if talk_time >= ALERT_THRESHOLD and not alert_triggered:
            print("🚨 Alert: Driver has been talking for too long! 🚨")
            alert_triggered = True

    return alert_triggered

def analyze_file(file_path, is_video=True):
    """
    Call this from main.py to analyze audio or video for speech detection.

    Args:
        file_path (str): path to the video or audio file
        is_video (bool): True if video, False if audio

    Returns:
        bool: True if alert was triggered
    """
    temp_audio_path = "temp_speech_audio.wav"

    if is_video:
        extract_audio_from_video(file_path, temp_audio_path)
        audio_data = load_audio(temp_audio_path)
        os.remove(temp_audio_path)
    else:
        audio_data = load_audio(file_path)

    return run_speech_detection(audio_data)
