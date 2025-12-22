"""
Test script for memory logging functionality.
This script follows the same pattern as main.py but uses mock speech components
to provide dummy user input and skip TTS output.
"""

import sys
import os
import time

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.interfaces.speech_input import ISpeechInput
from src.interfaces.speech_output import ISpeechOutput
from src.implementations.llm.google_ai_backend import GoogleAIBackend
from src.core.miyori import MiyoriCore
from src.core.tool_registry import ToolRegistry
from src.tools.web_search import web_search_tool
from src.tools.file_ops import file_ops_tool
from src.tools.memory_search import create_memory_search_tool
from src.utils.logger import setup_logging


class MockSpeechInput(ISpeechInput):
    """Mock speech input that returns predefined text."""

    def __init__(self, dummy_text="Hello, this is a test conversation for memory logging."):
        self.dummy_text = dummy_text
        self.call_count = 0

    def listen(self, require_wake_word: bool = True) -> str | None:
        """Return dummy text on first call, then 'exit' to end the conversation."""
        self.call_count += 1

        if self.call_count == 1:
            print(f"MockSpeechInput: Returning dummy text: '{self.dummy_text}'")
            return self.dummy_text
        else:
            print("MockSpeechInput: Returning 'exit' to end conversation")
            return "exit"


class MockSpeechOutput(ISpeechOutput):
    """Mock speech output that prints instead of speaking."""

    def speak(self, text: str) -> None:
        """Print the text instead of converting to speech."""
        print(f"MockSpeechOutput: Speaking: '{text}'")


def run_memory_logging_test():
    """Run the memory logging test with mock components."""

    print("Starting memory logging test...")

    # Load config and setup logging (same as main.py)
    Config.load()
    setup_logging()

    # Create mock speech components
    speech_input = MockSpeechInput("This is a test of Miyori's memory logging. User input is dummied, and nothing should be saved to memory.")
    speech_output = MockSpeechOutput()

    # Create LLM backend (same memory system initialization as main.py)
    llm_backend = GoogleAIBackend()

    # Setup tools (same as main.py)
    tool_registry = ToolRegistry()
    tools_config = Config.data.get("tools", {})

    # Create memory search tool dependencies
    memory_search_tool = None
    if hasattr(llm_backend, 'memory_retriever') and hasattr(llm_backend, 'embedding_service'):
        memory_search_tool = create_memory_search_tool(
            llm_backend.memory_retriever,
            llm_backend.embedding_service
        )

    if tools_config.get("enabled", False):
        if tools_config.get("web_search", {}).get("enabled", False):
            tool_registry.register(web_search_tool)
        if tools_config.get("file_ops", {}).get("enabled", False):
            tool_registry.register(file_ops_tool)
        if memory_search_tool and tools_config.get("memory_search", {}).get("enabled", True):
            tool_registry.register(memory_search_tool)

    # Create MiyoriCore (same as main.py)
    miyori = MiyoriCore(
        speech_input=speech_input,
        speech_output=speech_output,
        llm=llm_backend,
        tool_registry=tool_registry
    )

    print("Running Miyori with mock components...")
    # Run the main loop (it will handle one conversation turn then exit)
    miyori.run()

    print("Memory logging test completed.")
    print("Check logs/memory.log for new memory events.")


if __name__ == "__main__":
    run_memory_logging_test()
