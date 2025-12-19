import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config
Config.load()

from src.interfaces.speech_input import ISpeechInput
from src.implementations.speech.horrible_speech_input import HorribleSpeechInput

def run_speech_test(speech_input: ISpeechInput):
    """
    Test the Speech Input interface.
    This function interacts strictly with the ISpeechInput interface.
    """
    print("\n--- Speech Test ---")
    print("Please say the wake word 'Hey Miyori' to trigger...")
    
    result = speech_input.listen()
    
    print("-" * 20)
    if result:
        print(f"Recognized: {result}")
    else:
        print("No speech detected or error occurred.")
    print("Speech Test Completed.")

def main():
    try:
        speech_input = HorribleSpeechInput()
    except Exception as e:
        print(f"Failed to initialize speech input: {e}")
        return

    run_speech_test(speech_input)

if __name__ == "__main__":
    main()
