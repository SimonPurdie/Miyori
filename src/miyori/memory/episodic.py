import asyncio
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.interfaces.memory import IMemoryStore
from src.utils.embeddings import EmbeddingService
from src.memory.scoring import ImportanceScorer
from src.memory.budget import MemoryBudget
from src.utils.memory_logger import memory_logger

class EmbeddingQueue:
    def __init__(self, store: IMemoryStore, embedding_service: EmbeddingService):
        self.store = store
        self.embedding_service = embedding_service
        self.queue = asyncio.Queue()
        self.processing = False

    async def add_episode(self, episode_data: Dict[str, Any]) -> str:
        """Store episode immediately, queue embedding generation."""
        # Ensure status is pending_embedding
        episode_data['status'] = 'pending_embedding'
        episode_id = self.store.add_episode(episode_data)
        
        # Queue for embedding
        await self.queue.put((episode_id, episode_data['summary']))
        
        memory_logger.log_event("embedding_queued", {
            "id": episode_id,
            "text_len": len(episode_data['summary'])
        })
        
        # Start processor if not running
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return episode_id

    async def _process_queue(self):
        """Process embeddings in background."""
        self.processing = True
        while not self.queue.empty():
            episode_id, text = await self.queue.get()
            try:
                # Run sync embedding in executor to not block event loop
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(None, self.embedding_service.embed, text)
                
                # Convert list to bytes for BLOB storage
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                
                self.store.update_episode(episode_id, {
                    'embedding': embedding_blob,
                    'status': 'active'
                })
                memory_logger.log_event("embedding_success", {"id": episode_id})
            except Exception as e:
                import sys
                sys.stderr.write(f"Embedding failed for {episode_id}: {e}\n")
                memory_logger.log_event("embedding_failed", {"id": episode_id, "error": str(e)}, level="INFO")
        self.processing = False

class EpisodicMemoryManager:
    def __init__(self, store: IMemoryStore, embedding_service: EmbeddingService):
        self.store = store
        self.embedding_service = embedding_service
        self.queue = EmbeddingQueue(store, embedding_service)
        
        # Load config for budget
        import json
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f).get('memory', {})
        
        self.budget = MemoryBudget(store, config)

    async def add_episode(self, summary: str, full_text: Dict[str, str], importance: float = None):
        if importance is None:
            importance = ImportanceScorer.calculate_importance(full_text.get('user', ''), full_text.get('miyori', ''))
            
        episode_data = {
            'summary': summary,
            'full_text': full_text,
            'importance': importance,
            'timestamp': datetime.now().isoformat()
        }
        episode_id = await self.queue.add_episode(episode_data)
        
        # Trigger budget check
        self.budget.enforce_if_needed()
        
        return episode_id

    def retrieve_relevant(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        # 1. Vector search (over-fetch)
        query_embedding = self.embedding_service.embed(query)
        candidates = self.store.search_episodes(query_embedding, limit=limit*4, status='active')
        
        # 2. Rerank with formula: 
        # relevance = similarity * 0.5 + decayed_importance * 0.3 + recency * 0.2
        reranked = []
        for mem in candidates:
            # Calculate recency (decay over ~30 days)
            timestamp = datetime.fromisoformat(mem['timestamp'])
            age_days = (datetime.now() - timestamp).days
            recency_weight = 1.0 / (1 + age_days / 30)
            
            importance = mem.get('importance', 0.5)
            similarity = mem.get('similarity', 0.0)
            
            # Use decayed importance for ranking
            decayed_importance = ImportanceScorer.get_decayed_score(importance, mem['timestamp'])
            
            relevance = (similarity * 0.5) + (decayed_importance * 0.3) + (recency_weight * 0.2)
            mem['relevance_score'] = relevance
            reranked.append(mem)
            
            memory_logger.log_event("retrieval_ranking_item", {
                "id": mem['id'],
                "similarity": round(similarity, 3),
                "decayed_importance": round(decayed_importance, 3),
                "recency": round(recency_weight, 3),
                "relevance": round(relevance, 3)
            })
            
        # 3. Sort and return top N
        reranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = reranked[:limit]
        
        memory_logger.log_event("retrieval", {
            "query": query[:100],
            "candidate_count": len(candidates),
            "top_relevance": results[0]['relevance_score'] if results else 0
        })
        
        return results
