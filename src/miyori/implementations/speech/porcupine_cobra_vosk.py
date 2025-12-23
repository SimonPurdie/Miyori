import struct
import pvcobra
import pyaudio
import pvporcupine
import json
from vosk import Model, KaldiRecognizer
from collections import deque
from src.interfaces.speech_input import ISpeechInput
from src.utils.config import Config

class PorcupineCobraVosk(ISpeechInput):
    def __init__(self):
        speech_config = Config.data.get("speech_input", {})
        self.access_key = speech_config.get("porcupine_access_key")
        self.active_listen_timeout = speech_config.get("active_listen_timeout", 30)

        if not self.access_key:
            raise ValueError("Access key not found in config.json")

        # 1. Initialize Engines
        self.porcupine = self._init_porcupine(speech_config)
        self.cobra = pvcobra.create(access_key=self.access_key)
        
        self.sample_rate = self.porcupine.sample_rate 
        self.frame_length = self.porcupine.frame_length 
        
        # 2. Initialize Vosk
        model_path = speech_config.get("vosk_model_path")
        self.vosk_model = Model(model_path)
        self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

        # 3. Setup Audio Stream
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.frame_length
        )

        self.pre_roll_count = 15 
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_count)

    def _init_porcupine(self, config):
        keyword_paths = config.get("keyword_paths", [])
        keywords = config.get("keywords", ["porcupine"])
        if keyword_paths:
            return pvporcupine.create(access_key=self.access_key, keyword_paths=keyword_paths)
        return pvporcupine.create(access_key=self.access_key, keywords=keywords)

    def listen(self, require_wake_word: bool = True) -> str | None:
        try:
            if require_wake_word:
                print("[Status] Waiting for wake word...")
                while True:
                    pcm = self._read_frame()
                    self.pre_roll_buffer.append(pcm)
                    if self.porcupine.process(pcm) >= 0:
                        print("[Status] Wake word detected!")
                        break

            print("[Status] Listening for command...")
            recording_buffer = []

            # Hysteresis & Grace Period settings
            activation_threshold = 0.7  # Need a solid sound to start
            deactivation_threshold = 0.3 # Stay "active" even if quiet            

            is_speaking = False
            silence_frames = 0
            max_silence_frames = int(0.8 * (self.sample_rate / self.frame_length))
            max_recording_time = self.active_listen_timeout * (self.sample_rate / self.frame_length)

            for _ in range(int(max_recording_time)):
                pcm = self._read_frame()
                recording_buffer.append(pcm)
                
                voice_probability = self.cobra.process(pcm)
                
                # HYSTERESIS LOGIC
                if not is_speaking:
                    if voice_probability > activation_threshold:
                        is_speaking = True
                        silence_frames = 0
                        print("[Status] Speech started...")
                else:
                    # Once speaking, we only count silence if below the lower threshold
                    if voice_probability > deactivation_threshold:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                
                # GRACE PERIOD LOGIC
                if is_speaking and silence_frames > max_silence_frames:
                    print("[Status] Definitive end of speech detected.")
                    break
            else:
                print("[Status] Recording timed out.")

            if not is_speaking:
                return None

            return self._transcribe(recording_buffer)

        except Exception as e:
            print(f"Error in listen: {e}")
            return None

    def _read_frame(self):
        data = self.audio_stream.read(self.frame_length, exception_on_overflow=False)
        return struct.unpack_from("h" * self.frame_length, data)

    def _transcribe(self, pcm_frames):
        """Converts PCM frames to text using Vosk."""
        print("[Status] Transcribing with Vosk...")
        
        # Flatten and convert to raw bytes
        flat_pcm = [item for frame in pcm_frames for item in frame]
        audio_bytes = struct.pack("h" * len(flat_pcm), *flat_pcm)
        
        # Vosk processing
        if self.vosk_recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self.vosk_recognizer.Result())
        else:
            result = json.loads(self.vosk_recognizer.FinalResult())
            
        text = result.get("text", "")
        if text:
            print(f"user: {text}")
            return text
        
        print("Vosk could not understand audio.")
        return None

    def __del__(self):
        if hasattr(self, 'porcupine'): self.porcupine.delete()
        if hasattr(self, 'cobra'): self.cobra.delete()
        if hasattr(self, 'audio_stream'): self.audio_stream.close()
        if hasattr(self, 'pa'): self.pa.terminate()