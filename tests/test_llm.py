import sys
import os
import time

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interfaces.llm_backend import ILLMBackend
from src.implementations.llm.google_ai_backend import GoogleAIBackend

def run_llm_test(backend: ILLMBackend, prompt: str = "tell me a story"):
    """
    Test the LLM backend interface.
    This function interacts strictly with the ILLMBackend interface.
    """
    print(f"\nSending prompt: '{prompt}'")
    print("-" * 20)
    
    # Use a list to be mutable in the inner function
    last_chunk_time = [0.0]

    def on_chunk(text):
        current_time = time.time()
        if last_chunk_time[0] != 0.0:
            diff_ms = (current_time - last_chunk_time[0]) * 1000
            print(f"[{diff_ms:.0f}ms] {text}", end="", flush=True)
        else:
            # First chunk
            print(f"[First chunk] {text}", end="", flush=True)
        last_chunk_time[0] = current_time

    try:
        last_chunk_time[0] = time.time() # Start timing right before the call
        backend.generate_stream(prompt, on_chunk)
        print("\n" + "-" * 20)
        print("LLM Test Completed.")
    except Exception as e:
        print(f"\nError during generation: {e}")

def main():
    try:
        # Instantiate the specific implementation here
        backend = GoogleAIBackend()
    except Exception as e:
        print(f"Failed to initialize backend: {e}")
        return

    # Pass it to the generic test function
    run_llm_test(backend)

if __name__ == "__main__":
    main()
