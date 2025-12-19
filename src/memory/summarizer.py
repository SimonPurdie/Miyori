from google import genai
import json
from pathlib import Path
from src.utils.config import Config

class Summarizer:
    def __init__(self, client: genai.Client = None):
        self.client = client
        llm_config = Config.data.get("llm", {})
        if not self.client:
            api_key = llm_config.get("api_key")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                print("Warning: API Key not found for Summarizer")
        
        # Use a cheap/fast model for summarization
        self.model_name = llm_config.get("summarizer_model")

    async def create_summary(self, user_msg: str, assistant_msg: str) -> str:
        """Create a semantic summary of the exchange using the LLM."""
        if not self.client:
            # Fallback to simple concatenation if LLM unavailable
            return f"User: {user_msg[:100]} | Assistant: {assistant_msg[:100]}"

        prompt = f"""Summarize this conversation exchange in 2-3 sentences.
Preserve: key facts, emotions, decisions, and context.

User: {user_msg}
Assistant: {assistant_msg}

Summary:"""

        try:
            # Running the sync client call in a thread to keep it async-friendly
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            ))
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Turn summary: {user_msg[:50]}..."
