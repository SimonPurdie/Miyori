import struct
import json
import time
import math
from pathlib import Path
import pyaudio
import pvporcupine
import pvcheetah
from src.interfaces.speech_input import ISpeechInput
from src.utils.config import Config

class PorcupineSpeechInput(ISpeechInput):
    def __init__(self):
        speech_config = Config.data.get("speech_input", {})
        self.access_key = speech_config.get("porcupine_access_key")
        
        if not self.access_key:
            raise ValueError("porcupine_access_key not found in config.json")
            
        self.keyword_paths = speech_config.get("keyword_paths", [])
        self.keywords = speech_config.get("keywords", ["porcupine"])
        endpoint_duration = speech_config.get("endpoint_duration_sec", 1.0)
        
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
            
        # Init Cheetah
        self.cheetah = pvcheetah.create(
            access_key=self.access_key,
            endpoint_duration_sec=endpoint_duration
        )
            
        self.pa = pyaudio.PyAudio()
        
        # Determine strict frame length
        # Porcupine and Cheetah usually use 512, but we need to buffer if they differ
        self.frame_length = self.porcupine.frame_length
        if self.cheetah.frame_length != self.porcupine.frame_length:
             raise RuntimeError(f"Frame length mismatch: Porcupine ({self.porcupine.frame_length}) vs Cheetah ({self.cheetah.frame_length})")

        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.frame_length
        )
        
        print(f"Porcupine & Cheetah initialized. Keywords: {self.keyword_paths if self.keyword_paths else self.keywords}")

    def listen(self, require_wake_word: bool = True) -> str | None:
        """
        Listens for wake word, then captures command using Cheetah ASR.
        """
        try:
            # 1. Wait for Wake Word (if required)
            if require_wake_word:
                print("\nWaiting for wake word...")
                while True:
                    pcm = self.audio_stream.read(self.frame_length, exception_on_overflow=False)
                    pcm = struct.unpack_from("h" * self.frame_length, pcm)
                    
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        print("Wake word detected!")
                        break
            
            # 2. Transcribe with Cheetah
            print("Listening for command...")
            text = ""
            is_endpoint = False
            
            while not is_endpoint:
                pcm = self.audio_stream.read(self.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.frame_length, pcm)
                
                partial_transcript, is_endpoint = self.cheetah.process(pcm)
                if partial_transcript:
                    text += partial_transcript
                    # Optional: print partials
                    # print(partial_transcript, end="", flush=True)

            # 3. Final flush
            final_transcript = self.cheetah.flush()
            text += final_transcript
            
            print(f"You said: {text}")
            return text
                
        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"Error in listen: {e}")
            return None


    def __del__(self):
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'cheetah'):
            self.cheetah.delete() 
        if hasattr(self, 'audio_stream'):
            self.audio_stream.close()
        if hasattr(self, 'pa'):
            self.pa.terminate()
