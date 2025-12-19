import struct
import json
import time
import math
from pathlib import Path
import speech_recognition as sr
import pyaudio
import pvporcupine
from src.interfaces.speech_input import ISpeechInput
from src.utils.config import Config

class HorribleSpeechInput(ISpeechInput):
    def __init__(self):
        
        speech_config = Config.data.get("speech_input", {})
        self.access_key = speech_config.get("porcupine_access_key")
        
        if not self.access_key:
            raise ValueError("porcupine_access_key not found in config.json")
            
        self.keyword_paths = speech_config.get("keyword_paths", [])
        self.keywords = speech_config.get("keywords", ["porcupine"])
        
        # Init Porcupine
        # Prioritize keyword_paths if provided, otherwise use keywords
        if self.keyword_paths:
             self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=self.keyword_paths
            )
        else:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=self.keywords
            )
            
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        # For Google ASR fallback
        self.recognizer = sr.Recognizer()
        self.non_speaking_duration = 5.0
        self.pause_threshold = speech_config.get("pause_threshold", 2.0)
        self.energy_threshold = speech_config.get("energy_threshold", 300)
        
        print(f"Porcupine initialized. Keywords: {self.keyword_paths if self.keyword_paths else self.keywords}")

    def listen(self, require_wake_word: bool = True) -> str | None:
        """
        Listens for wake word, then captures command for ASR.
        """
        try:
            # 1. Wait for Wake Word (if required)
            if require_wake_word:
                print("\nWaiting for wake word...")
                while True:
                    pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        print("Wake word detected!")
                        break
            
            # 2. Release PyAudio stream to let SpeechRecognition take over
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
            # 3. Use SpeechRecognition for command capture
            print("Listening for command...")
            try:
                with sr.Microphone() as source:
                    # Quick adjustment for ambient noise can help accuracy
                    # self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    
                print("Processing command...")

                # Janky hack: Append 500ms of silence to avoid it clipping at the end for some reason.
                raw_data = audio.get_raw_data()
                silence_frames = int(audio.sample_rate * 0.5)
                silence_bytes = silence_frames * audio.sample_width
                silence = b'\0' * silence_bytes
                
                audio = sr.AudioData(raw_data + silence, audio.sample_rate, audio.sample_width)

                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                
            except sr.WaitTimeoutError:
                print("No speech detected (timeout)")
                text = None
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                text = None
            except sr.RequestError as e:
                print(f"Request error: {e}")
                text = None
                
            # 4. Restart PyAudio stream for next wake word
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            # Return valid text or None (to continue loop in assistant)
            return text
                
        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"Error in listen: {e}")
            return None


    def __del__(self):
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio_stream'):
            self.audio_stream.close()
        if hasattr(self, 'pa'):
            self.pa.terminate()
