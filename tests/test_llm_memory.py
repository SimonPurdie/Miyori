import sys
import os
import time

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
Config.load()
from src.implementations.llm.google_ai_backend import GoogleAIBackend

def main():
    backend = GoogleAIBackend()
    
    def on_chunk(text):
        print(text, end="", flush=True)

    print("\nTurn 1: Remembering fruits.")
    backend.generate_stream("remember these fruits: apple, orange, mango.", on_chunk)
    print("\n" + "-"*20)
    print("\nTurn 2: Recalling fruits.")
    backend.generate_stream("What are the fruits?", on_chunk)
    print("\n" + "-"*20)
    time.sleep(3) # Wait a bit to ensure async output is done

if __name__ == "__main__":
    main()
