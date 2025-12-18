"""LLM Backend Interface

This interface defines the contract for Large Language Model implementations.
Any class implementing this interface must provide a generate_stream() method
that takes a prompt and a callback, streaming AI-generated response chunks.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any
from src.core.tools import Tool


class ILLMBackend(ABC):
    """Interface for LLM backend implementations"""
    
    @abstractmethod
    def generate_stream(self, prompt: str, on_chunk: Callable[[str], None]) -> None:
        """Generate an AI response with streaming chunks.
        
        Args:
            prompt: The user's input text to generate a response for
            on_chunk: Callback function called with each text chunk as it arrives
        """
        pass

    @abstractmethod
    def reset_context(self) -> None:
        """Reset the conversation context (history)."""
        pass

    @abstractmethod
    def generate_stream_with_tools(
        self,
        prompt: str,
        tools: List[Tool],
        on_chunk: Callable[[str], None],
        on_tool_call: Callable[[str, Dict[str, Any]], str]
    ) -> None:
        """Generate an AI response with streaming chunks and tool support.
        
        Args:
            prompt: The user's input text
            tools: List of available tools
            on_chunk: Callback for text chunks
            on_tool_call: Callback for tool execution
        """
        pass
