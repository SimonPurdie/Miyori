from google import genai
import json
import numpy as np
from typing import List, Dict, Any
from src.interfaces.memory import IMemoryStore
from src.utils.config import Config
from src.utils.embeddings import EmbeddingService

class SemanticExtractor:
    def __init__(self, client: genai.Client, store: IMemoryStore):
        self.client = client
        self.store = store
        self.model_name = Config.data.get("memory", {}).get("semantic_model")
        self.embedding_service = EmbeddingService()

    async def extract_facts_from_batch(self, clusters: List[List[Dict[str, Any]]]):
        """Extract semantic facts from multiple clusters in a single API call."""
        if not clusters or not self.client:
            return
        
        # Build prompt with cluster structure
        prompt = "Extract facts that Miyori has observed from these memories she has.\n\n"
        prompt += "Your extracted fact will be used in her deeper memories, and it should be phrased in first person.\n"
        prompt += "Do not say 'The user asked Miyori to tell a story', but say 'The user asked me to tell a story'.\n\n"
        prompt += "Each cluster contains related conversations. Look for:\n"
        prompt += "- Facts that appear multiple times within a cluster\n"
        prompt += "- Recurring preferences, patterns, and decisions\n\n"
        
        for cluster_idx, cluster in enumerate(clusters):
            prompt += f"<CLUSTER_{cluster_idx}>\n"
            for episode in cluster:
                prompt += f"- {episode['summary']}\n"
            prompt += f"</CLUSTER_{cluster_idx}>\n\n"
        
        prompt += "Extract facts as simple sentences. Format: one fact per line.\n\nFacts:"
        
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            ))
            
            # Parse facts from response
            raw_facts = [line.strip("- ").strip() for line in response.text.split("\n") if line.strip()]
            facts = [fact for fact in raw_facts if len(fact) > 5]

            if facts:
                # Generate embeddings for all facts in batch
                embeddings = self.embedding_service.batchEmbedContents(facts)

                # Store facts with source episode tracking
                all_episode_ids = [ep['id'] for cluster in clusters for ep in cluster]

                for fact, embedding in zip(facts, embeddings):
                    # Convert embedding to bytes for database storage
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

                    self.store.add_semantic_fact({
                        "fact": fact,
                        "confidence": 0.7,
                        "status": "stable",
                        "derived_from": all_episode_ids,  # Track which episodes contributed
                        "embedding": embedding_bytes
                    })

            print(f"Extracted {len(facts)} facts from {len(clusters)} clusters ({len(all_episode_ids)} episodes)")
            
        except Exception as e:
            import sys
            sys.stderr.write(f"Semantic Extraction failed: {e}\n")

