from google import genai
import json
from typing import List, Dict, Any
from src.interfaces.memory import IMemoryStore
from src.utils.config import Config

class SemanticExtractor:
    def __init__(self, client: genai.Client, store: IMemoryStore):
        self.client = client
        self.store = store
        self.model_name = Config.data.get("memory", {}).get("semantic_model")

    async def extract_facts_from_batch(self, clusters: List[List[Dict[str, Any]]]):
        """Extract semantic facts from multiple clusters in a single API call."""
        if not clusters or not self.client:
            return
        
        # Build prompt with cluster structure
        prompt = "Extract stable semantic facts about the user from these semantically-grouped conversations.\n\n"
        prompt += "Each cluster contains related conversations. Look for:\n"
        prompt += "- Facts that appear multiple times within a cluster\n"
        prompt += "- Facts that appear across different clusters\n"
        prompt += "- Recurring preferences, patterns, and decisions\n\n"
        
        for cluster_idx, cluster in enumerate(clusters):
            prompt += f"<CLUSTER_{cluster_idx}>\n"
            for episode in cluster:
                prompt += f"- {episode['summary']}\n"
            prompt += f"</CLUSTER_{cluster_idx}>\n\n"
        
        prompt += "Extract only objective facts as simple sentences. Format: one fact per line.\n\nFacts:"
        
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            ))
            
            # Parse facts from response
            facts = [line.strip("- ").strip() for line in response.text.split("\n") if line.strip()]
            
            # Store facts with source episode tracking
            all_episode_ids = [ep['id'] for cluster in clusters for ep in cluster]
            
            for fact in facts:
                if len(fact) > 5:
                    self.store.add_semantic_fact({
                        "fact": fact,
                        "confidence": 0.7,
                        "status": "stable",
                        "derived_from": all_episode_ids  # Track which episodes contributed
                    })
                    
            print(f"Extracted {len(facts)} facts from {len(clusters)} clusters ({len(all_episode_ids)} episodes)")
            
        except Exception as e:
            import sys
            sys.stderr.write(f"Semantic Extraction failed: {e}\n")

class EmotionalTracker:
    def __init__(self, store: IMemoryStore):
        self.store = store

    def update_thread(self, user_msg: str, miyori_msg: str):
        """Update the current emotional thread based on the latest exchange."""
        # Simple Phase 3 implementation: check for emotion keywords
        # In a real system, this would be an LLM call or sentiment analysis
        emotions = {
            "happy": ["great", "happy", "love", "good"],
            "sad": ["sad", "unhappy", "bad", "sorry"],
            "angry": ["angry", "mad", "hate", "stop"],
            "stressed": ["stress", "busy", "tired", "hard"]
        }
        
        detected = "neutral"
        for emotion, keywords in emotions.items():
            if any(kw in user_msg.lower() for kw in keywords):
                detected = emotion
                break
                
        current = self.store.get_emotional_thread() or {
            "current_state": "neutral",
            "thread_length": 0
        }
        
        new_state = detected
        thread_length = current['thread_length'] + 1 if new_state == current['current_state'] else 1
        
        self.store.update_emotional_thread({
            "current_state": new_state,
            "thread_length": thread_length,
            "should_acknowledge": thread_length > 2
        })
