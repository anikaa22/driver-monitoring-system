#!/usr/bin/env python
# coding: utf-8

# In[1]:


import webrtcvad
import numpy as np
import wave
from pydub import AudioSegment
from pydub.utils import which
from moviepy.editor import VideoFileClip
import os

# Fix: Tell pydub where ffmpeg is
AudioSegment.converter = r"C:\Users\Anika\ffmpeg\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin\ffmpeg.exe"

# Parameters
FRAME_DURATION_MS = 30
RATE = 16000
FRAME_SIZE = int(RATE * (FRAME_DURATION_MS / 1000))
alert_threshold = 100  # ~30 seconds of speech (30ms * 100)
buffer_size = 100

def get_audio_energy(frame):
    audio_data = np.frombuffer(frame, dtype=np.int16)
    return np.abs(audio_data).mean()

def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, fps=RATE)

def load_audio(path):
    sound = AudioSegment.from_file(path)
    sound = sound.set_channels(1).set_frame_rate(RATE).set_sample_width(2)
    return sound.raw_data

def run_speech_detection(audio_data):
    vad = webrtcvad.Vad()
    vad.set_mode(1)

    talk_time = 0
    silence_time = 0
    speech_buffer = []
    alert_triggered = False
    talking = False
    silence_announced = False

    print("Analyzing audio...")

    for i in range(0, len(audio_data), FRAME_SIZE * 2):
        frame = audio_data[i:i + FRAME_SIZE * 2]
        if len(frame) < FRAME_SIZE * 2:
            break

        is_speech = vad.is_speech(frame, RATE)
        energy = get_audio_energy(frame)
        if energy < 500:
            is_speech = False

        speech_buffer.append(is_speech)
        if len(speech_buffer) > buffer_size:
            speech_buffer.pop(0)

        if sum(speech_buffer) > buffer_size // 3:
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
            if silence_time > buffer_size:
                talk_time = 0
                alert_triggered = False

        if talk_time >= alert_threshold and not alert_triggered:
            print("🚨 Alert: Driver has been talking for too long! 🚨")
            alert_triggered = True

# === Choose your file here ===

# AUDIO EXAMPLE
# audio_path = "path_to_audio.wav"
# audio_data = load_audio(audio_path)
# run_speech_detection(audio_data)

# VIDEO EXAMPLE
video_path = "test_audio.mp4"
temp_audio_path = "temp_audio.wav"
extract_audio_from_video(video_path, temp_audio_path)
audio_data = load_audio(temp_audio_path)
run_speech_detection(audio_data)

# Clean up
if os.path.exists(temp_audio_path):
    os.remove(temp_audio_path)


# In[ ]:




