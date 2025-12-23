import sys
import os
import time
from typing import Dict, Any

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from miyori.utils.config import Config
Config.load()
from miyori.implementations.llm.google_ai_backend import GoogleAIBackend
from miyori.core.tools import Tool, ToolParameter

# Dummy tool for testing when no tools are needed
def dummy_tool_function():
    return "This is a dummy tool response."

dummy_tool = Tool(
    name="dummy_tool",
    description="A dummy tool for testing purposes",
    parameters=[],
    function=dummy_tool_function
)

def main():
    backend = GoogleAIBackend()

    def on_chunk(text):
        print(text, end="", flush=True)

    def on_tool_call(name: str, args: Dict[str, Any]) -> str:
        # For dummy tool, just return a simple response
        if name == "dummy_tool":
            return dummy_tool_function()
        return "Unknown tool called"

    print("\nTurn 1: Remembering fruits.")
    backend.llm_chat("remember these fruits: apple, orange, mango.", [dummy_tool], on_chunk, on_tool_call)
    print("\n" + "-"*20)
    print("\nTurn 2: Recalling fruits.")
    backend.llm_chat("What are the fruits?", [dummy_tool], on_chunk, on_tool_call)
    print("\n" + "-"*20)
    time.sleep(3) # Wait a bit to ensure async output is done

if __name__ == "__main__":
    main()
