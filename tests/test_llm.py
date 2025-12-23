import sys
import os
import time
from typing import Dict, Any

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from miyori.utils.config import Config
Config.load()

from miyori.interfaces.llm_backend import ILLMBackend
from miyori.implementations.llm.google_ai_backend import GoogleAIBackend
from miyori.core.tools import Tool, ToolParameter

# Dummy tool for testing when no tools are needed
def dummy_tool_function(**kwargs):
    return "This is a dummy tool response."

dummy_tool = Tool(
    name="dummy_tool",
    description="A dummy tool for testing purposes",
    parameters=[],
    function=dummy_tool_function
)

def run_llm_test(backend: ILLMBackend, prompt: str = "hey Miyori, I'm testing the LLM backend interface. This interaction isnt important so don't worry about remembering it. Just tell me if you're there."):
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

    def on_tool_call(name: str, args: Dict[str, Any]) -> str:
        # For dummy tool, just return a simple response
        if name == "dummy_tool":
            return dummy_tool_function()
        return "Unknown tool called"

    try:
        last_chunk_time[0] = time.time() # Start timing right before the call
        backend.llm_chat(prompt, [dummy_tool], on_chunk, on_tool_call)
        print("\n" + "-" * 20)
        time.sleep(3) # Wait a bit to ensure async output is done
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
