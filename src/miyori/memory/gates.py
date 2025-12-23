from google import genai
from typing import Optional
from src.utils.memory_logger import memory_logger
from src.utils.config import Config

class MemoryGate:
    def __init__(self, client: genai.Client):
        self.client = client
        self.model_name = Config.data.get("memory", {}).get("gate_model")

    async def should_remember(self, user_msg: str, miyori_msg: str) -> bool:
        """
        Decide if a conversation turn should be stored using LLM-aided gating.
        """
        # 1. Explicit request bypass (fast)
        keywords = ["remember this", "don't forget", "take a note", "keep this in mind"]
        if any(kw in user_msg.lower() for kw in keywords):
            print("Memory Gate: Explicit request detected.")
            return True

        # 2. LLM Evaluation for complex gates
        if not self.client:
            return True # Fallback to true if no client

        prompt = f"""Evaluate if this conversation exchange should be approved for Miyori's memory.
We should remember this if it contains:
- Important facts about the user
- Useful details about the working environment
- Significant decisions, goals, or commitments
- Information that could causes relational damage if forgotten
Without approval, this exchange will be forgotten.

User: {user_msg}
Miyori: {miyori_msg}

Answer with 'YES' or 'NO' followed by ' --- <one sentence explaining reasoning>':"""

        try:            
            print("Memory Gate: Evaluating with LLM...")
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            ))
            decision = response.text.strip().upper()
            print(f"Memory Gate Decision: {decision}")
            should_rem = "YES" in decision
            memory_logger.log_event("gate_decision", {
                "decision": decision,
                "should_remember": should_rem,
                "user_msg": user_msg[:100],
                "miyori_msg": miyori_msg[:100]
            })
            return should_rem
        except Exception as e:
            memory_logger.log_event("gate_error", {"error": str(e)})
            return True # Conservative fallback: store it anyway
