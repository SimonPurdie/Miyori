import sys
import os
import json
import time

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interfaces.speech_output import ISpeechOutput
from src.implementations.tts.pyttsx_output import PyttsxOutput

def run_tts_test(tts: ISpeechOutput):
    """
    Test the Speech Output interface.
    This function interacts strictly with the ISpeechOutput interface.
    """
    # Load test data
    json_path = os.path.join(os.path.dirname(__file__), 'tts_test_text.json')
    print(f"Loading test data from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return

    print(f"\nStarting stream simulation with {len(test_data)} chunks...")
    
    try:
        for i, item in enumerate(test_data):
            delay_ms = item.get('delayMs', 0)
            text = item.get('data', '')
            
            # Simulate network/generation delay
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            
            print(f"Received Chunk {i+1}: '{text}' (after {delay_ms}ms delay)")
            tts.speak(text)
            
        print("Waiting for speech to finish...")
        print("All chunks sent to TTS. Keeping process alive for 30s to allow playback...")
        print("Press Ctrl+C to exit early.")
        
        try:
            start_wait = time.time()
            while time.time() - start_wait < 30:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExited early by user.")

        print("TTS Test Completed.")
    except Exception as e:
        print(f"Error during TTS: {e}")

def main():
    try:
        tts = PyttsxOutput()
    except Exception as e:
        print(f"Failed to initialize TTS: {e}")
        return

    run_tts_test(tts)

if __name__ == "__main__":
    main()
