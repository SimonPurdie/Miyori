from google import genai
import json
from pathlib import Path
from typing import Callable
from src.interfaces.llm_backend import ILLMBackend

class GoogleAIBackend(ILLMBackend):
    def __init__(self):
        # e:/_Projects/Miyori/src/implementations/llm/google_ai_backend.py
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        llm_config = config.get("llm", {})
        self.api_key = llm_config.get("api_key")
        self.model_name = llm_config.get("model", "gemini-2.0-flash-exp")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Handle missing API key gracefully or let it fail later
            self.client = None

    def generate_stream(self, prompt: str, on_chunk: Callable[[str], None]) -> None:
        if not self.client:
            print("Error: API Key not configured.")
            return

        print("Thinking...")
        # New SDK usage: client.models.generate_content_stream
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt
        )
        
        for chunk in response:
            if chunk.text:
                on_chunk(chunk.text)
