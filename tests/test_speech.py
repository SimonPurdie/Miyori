from pydoc import text
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from miyori.utils.config import Config
Config.load()

from miyori.interfaces.speech_input import ISpeechInput
from miyori.implementations.speech.porcupine_cobra_vosk import PorcupineCobraVosk

def run_speech_test(speech_input: ISpeechInput):
    """
    Test the Speech Input interface.
    This function interacts strictly with the ISpeechInput interface.
    """
    print("\n--- Speech Test ---")
    print("Please say the wake word 'Hey Miyori' to trigger...")
    
    wake_word_required = True

    while True:
        result = speech_input.listen(wake_word_required)
        wake_word_required = False  # After first wake word, no longer required
    
        print("-" * 20)
        if result:
            print(f"Recognized: {result}")
            if set(['exit', 'goodbye']).intersection(result.lower().split()):
                print("Goodbye!")
                break
        else:
            print("No speech detected or error occurred.")
            break
    
    print("Speech Test Completed.")

def main():
    try:
        speech_input = PorcupineCobraVosk()
    except Exception as e:
        print(f"Failed to initialize speech input: {e}")
        return

    run_speech_test(speech_input)

if __name__ == "__main__":
    main()
