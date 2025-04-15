#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install webrtcvad-wheels')


# In[2]:


get_ipython().system('pip install pyaudio')


# In[3]:


import webrtcvad
import numpy as np
import pyaudio
import time


# In[ ]:


# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # Reduce sensitivity (0 = least aggressive, 3 = most aggressive)

# PyAudio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # WebRTC VAD works best at 16kHz
FRAME_DURATION_MS = 30  # Must be 10, 20, or 30 ms
FRAME_SIZE = int(RATE * (FRAME_DURATION_MS / 1000))  # 480 samples for 30ms

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

talk_time = 0  # Speech counter
silence_time = 0  # Silence counter
speech_buffer = []  # Store last few speech detections
buffer_size = 100  # Number of frames to consider for stable speech detection
alert_triggered = False  # Avoid repeated alerts
alert_threshold = 100  # Alert if driver talks for ~30 seconds
talking = False  # Track if driver is talking
silence_announced = False  # Prevent multiple "Silence detected."

print("Listening for speech...")

def get_audio_energy(frame):
    """Calculate the average energy of an audio frame to filter out background noise."""
    audio_data = np.frombuffer(frame, dtype=np.int16)
    return np.abs(audio_data).mean()

while True:
    frame = stream.read(FRAME_SIZE, exception_on_overflow=False)  # Read audio frame
    
    if len(frame) != FRAME_SIZE * 2:  # Each sample is 2 bytes (16-bit audio)
        continue  # Skip if frame size is incorrect
    
    is_speech = vad.is_speech(frame, RATE)

    # Apply background noise filtering
    energy = get_audio_energy(frame)
    if energy < 500:  # Ignore low-energy noise
        is_speech = False  

    # Maintain speech buffer for smoother detection
    speech_buffer.append(is_speech)
    if len(speech_buffer) > buffer_size:
        speech_buffer.pop(0)  # Keep buffer size fixed
    
    # Only confirm speech if majority of last frames detect speech
    if sum(speech_buffer) > buffer_size // 3:
        if not talking:  # If it's the first time detecting speech
            print("Driver is talking...")  # Only print once
        talking = True
        talk_time += 1
        silence_time = 0
        silence_announced = False  # Reset silence announcement
    else:
        if not talking and not silence_announced and silence_time > 100:  # Only print once when talking stops
            print("Silence detected.")
            silence_announced = True
        talking = False
        silence_time += 1
        if silence_time > buffer_size:
            talk_time = 0  # Reset talk time on long silence
            alert_triggered = False  # Allow new alerts after silence

    # Trigger alert only once after 30 seconds of continuous talking
    if talk_time >= alert_threshold and not alert_triggered:
        print("🚨 Alert: Driver has been talking for too long! 🚨")
        alert_triggered = True  # Prevent repeated alerts


# In[ ]:




