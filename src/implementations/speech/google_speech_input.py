import speech_recognition as sr
import json
from pathlib import Path
from src.interfaces.speech_input import ISpeechInput

class GoogleSpeechInput(ISpeechInput):
    def __init__(self):
        # Get config path: Path(__file__).parent.parent.parent / "config.json"
        # This resolves to e:/_Projects/Miyori/src/implementations/speech/../../../config.json -> e:/_Projects/Miyori/config.json
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        speech_config = config.get("speech_input", {})
        
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = speech_config.get("pause_threshold", 2.0)
        self.recognizer.energy_threshold = speech_config.get("energy_threshold", 300)

    def listen(self, require_wake_word: bool = True) -> str | None:
        try:
            with sr.Microphone() as source:
                print("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Listening...")
                audio = self.recognizer.listen(source)
                print("Processing...")
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
        except Exception as e:
            
            return None
