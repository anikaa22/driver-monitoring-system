import webrtcvad
import numpy as np
import pyaudio
import time

def monitor_speech(duration_seconds=30):
    vad = webrtcvad.Vad()
    vad.set_mode(1)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(RATE * (FRAME_DURATION_MS / 1000))

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

    talk_time = 0
    silence_time = 0
    speech_buffer = []
    buffer_size = 100
    alert_triggered = False
    alert_threshold = 100
    talking = False
    silence_announced = False

    print("🎙️ Speech detection started...")

    def get_audio_energy(frame):
        audio_data = np.frombuffer(frame, dtype=np.int16)
        return np.abs(audio_data).mean()

    start_time = time.time()

    try:
        while time.time() - start_time < duration_seconds:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)

            if len(frame) != FRAME_SIZE * 2:
                continue

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

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("🛑 Speech detection ended.")
