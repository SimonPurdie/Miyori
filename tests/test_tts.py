import sys
import os
import json
import time

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.implementations.tts.pyttsx_output import PyttsxOutput

def test_tts():
    print("Initializing PyttsxOutput...")
    try:
        tts = PyttsxOutput()
    except Exception as e:
        print(f"Failed to initialize TTS: {e}")
        return

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
        if hasattr(tts, 'pipeline'):
            tts.pipeline.wait_for_completion()

        print("TTS Test Completed.")
    except Exception as e:
        print(f"Error during TTS: {e}")

if __name__ == "__main__":
    test_tts()
