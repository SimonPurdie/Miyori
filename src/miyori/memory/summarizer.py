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

    async def create_summary(self, user_msg: str, miyori_msg: str, recent_context: list[str] = None) -> str:
        """Create a semantic summary of the exchange using the LLM, with optional recent conversation context."""
        if not self.client:
            # Fallback to simple concatenation if LLM unavailable
            return f"User: {user_msg[:100]} | miyori: {miyori_msg[:100]}"

        # Build context from recent turns if available
        context_section = ""
        if recent_context and len(recent_context) > 0:
            context_section = "\n\nRecent conversation context:\n" + "\n\n".join(recent_context) + "\n\n"

        prompt = f"""Write a 1–2 sentence summary of the recent exchange to be stored in Miyori’s long-term memory.
Write the summary in the *first person*, drafting it as if Miyori is recording their own memory.
Use "I", "me", "my" to refer to Miyori.
The user is using voice recognition and their input may contain errors; rely on Miyori's responses to clarify any transcription errors in the user's input.
Focus primarily on the most recent messages, using earlier turns only for supporting context.
Preserve: key facts, emotions, decisions. {context_section} 

Current exchange:
User: {user_msg}
Miyori: {miyori_msg}

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
