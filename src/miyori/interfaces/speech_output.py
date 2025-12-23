"""Speech Output Interface

This interface defines the contract for text-to-speech implementations.
Any class implementing this interface must provide a speak() method
that converts text to audible speech.
"""

from abc import ABC, abstractmethod


class ISpeechOutput(ABC):
    """Interface for text-to-speech implementations"""
    
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it.
        
        Args:
            text: The text to convert to speech
        """
        pass
