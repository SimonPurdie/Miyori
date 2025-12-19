from google import genai
import json
from typing import List, Dict, Any
from src.interfaces.memory import IMemoryStore
from src.memory.deep_layers import SemanticExtractor, EmotionalTracker
from src.utils.config import Config

class RelationalManager:
    def __init__(self, client: genai.Client, store: IMemoryStore):
        self.client = client
        self.store = store
        self.interaction_count = 0

    async def analyze_relationship(self, episodes: List[Dict[str, Any]]):
        """Analyze patterns in interaction to update relational norms."""
        if not episodes or not self.client:
            return

        summaries = "\n".join([e['summary'] for e in episodes])
        prompt = f"""Analyze these conversation summaries to update our interaction style and user preferences.
Focus on: tone, communication style, topics of interest, and interaction norms.
Be conservative: only update if patterns are consistent.

Summaries:
{summaries}

Current Relational State: {self.store.get_relational_memories()}

Updated Relational Data (JSON):"""

        try:
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model= Config.data.get("memory", {}).get("relational_model"),
                contents=prompt
            ))
            # Expecting JSON response or similar
            # For Phase 3 simplicity, we assume some structure or just store text
            self.store.update_relational_memory("interaction_style", {"analysis": response.text.strip()}, 0.8)
        except Exception as e:
            print(f"Relational analysis failed: {e}")

class ContradictionDetector:
    def __init__(self, store: IMemoryStore):
        self.store = store

    def detect_conflicts(self, new_fact: str) -> List[Dict[str, Any]]:
        """Check if a new fact contradicts existing semantic memory."""
        # Simple Phase 3 implementation: string matching / keyword conflict
        # A more advanced version would use an LLM
        facts = self.store.get_semantic_facts()
        conflicts = []
        for f in facts:
            # Very simple placeholder for conflict logic
            if "not" in new_fact.lower() and new_fact.lower().replace("not ", "") in f['fact'].lower():
                conflicts.append(f)
        return conflicts

class ConsolidationManager:
    def __init__(self, store, episodic_manager, semantic_extractor, relational_manager):
        self.store = store
        self.episodic_manager = episodic_manager
        self.semantic_extractor = semantic_extractor
        self.relational_manager = relational_manager

    async def perform_consolidation(self):
        """Nightly consolidation task."""
        print("Starting Memory Consolidation...")
        # 1. Get recent episodes that haven't been consolidated
        # (For Phase 3, we'll just take the last 50 active ones)
        episodes = self.store.search_episodes([0.0]*768, limit=50, status='active')
        
        # 2. Extract semantic facts
        await self.semantic_extractor.extract_facts_batched(episodes)
        
        # 3. Analyze relationship
        await self.relational_manager.analyze_relationship(episodes)
        
        # 4. Cleanup/Archive old mundane ones
        # (Already handled by budget, but can add more specific logic here)
        
        print("Memory Consolidation Complete.")
