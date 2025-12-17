import time
import json
from pathlib import Path
from src.interfaces.speech_input import ISpeechInput
from src.interfaces.speech_output import ISpeechOutput
from src.interfaces.llm_backend import ILLMBackend

class VoiceAssistant:
    def __init__(self, 
                 speech_input: ISpeechInput,
                 speech_output: ISpeechOutput,
                 llm: ILLMBackend):
        self.speech_input = speech_input
        self.speech_output = speech_output
        self.llm = llm
        
        # Load config for active listen timeout
        self.active_listen_timeout = 30
        try:
            config_path = Path(__file__).parent.parent.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.active_listen_timeout = config.get("speech_input", {}).get("active_listen_timeout", 30)
            print(f"Active listening timeout set to {self.active_listen_timeout} seconds")
        except Exception as e:
            print(f"Error loading config, using default timeout: {e}")

    def run(self) -> None:
        print("Miyori starting up...")

        last_interaction_time = 0
        
        while True:
            # Determine if we need wake word
            require_wake_word = True
            if last_interaction_time > 0 and (time.time() - last_interaction_time < self.active_listen_timeout):
                require_wake_word = False
                
            text = self.speech_input.listen(require_wake_word=require_wake_word)
            if text is None:
                continue
            
            # Update last interaction time on successful speech detection
            last_interaction_time = time.time()
            
            # Check for exit commands (case-insensitive)
            if any(word in text.lower() for word in ['exit', 'quit', 'stop', 'goodbye']):
                self.speech_output.speak("Goodbye!")
                break
            
            # Define callback for streaming chunks - speak immediately
            def speak_chunk(chunk: str) -> None:
                self.speech_output.speak(chunk)
            
            # Use streaming for real-time TTS
            self.llm.generate_stream(text, speak_chunk)
        
        print("Miyori shutting down...")
