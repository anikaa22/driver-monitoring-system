import webrtcvad
import numpy as np

class SpeechDetector:
    def __init__(self, rate=16000, frame_duration_ms=30, buffer_size=100, alert_threshold=100):
        self.vad = webrtcvad.Vad(1)
        self.RATE = rate
        self.FRAME_DURATION_MS = frame_duration_ms
        self.FRAME_SIZE = int(self.RATE * (self.FRAME_DURATION_MS / 1000))
        self.buffer_size = buffer_size
        self.alert_threshold = alert_threshold

        self.speech_buffer = []
        self.talk_time = 0
        self.silence_time = 0
        self.alert_triggered = False
        self.talking = False
        self.silence_announced = False

    def get_audio_energy(self, frame):
        audio_data = np.frombuffer(frame, dtype=np.int16)
        return np.abs(audio_data).mean()

    def process_audio_frame(self, frame):
        """
        Takes one audio frame, updates internal speech state, and returns alert status.

        Returns:
            (talking: bool, alert_triggered: bool)
        """
        if len(frame) != self.FRAME_SIZE * 2:
            return self.talking, self.alert_triggered

        is_speech = self.vad.is_speech(frame, self.RATE)
        energy = self.get_audio_energy(frame)

        if energy < 500:
            is_speech = False

        self.speech_buffer.append(is_speech)
        if len(self.speech_buffer) > self.buffer_size:
            self.speech_buffer.pop(0)

        if sum(self.speech_buffer) > self.buffer_size // 3:
            if not self.talking:
                print("Driver is talking...")
            self.talking = True
            self.talk_time += 1
            self.silence_time = 0
            self.silence_announced = False
        else:
            if not self.talking and not self.silence_announced and self.silence_time > 100:
                print("Silence detected.")
                self.silence_announced = True
            self.talking = False
            self.silence_time += 1
            if self.silence_time > self.buffer_size:
                self.talk_time = 0
                self.alert_triggered = False

        if self.talk_time >= self.alert_threshold and not self.alert_triggered:
            print("🚨 Alert: Driver has been talking for too long! 🚨")
            self.alert_triggered = True

        return self.talking, self.alert_triggered
