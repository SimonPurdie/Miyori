import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.memory.memory_retriever import MemoryRetriever
from src.utils.embeddings import EmbeddingService
from src.utils.memory_logger import memory_logger

@dataclass
class MemoryCache:
    """Cache entry for passive memory retrieval."""
    episodic_memories: List[Dict[str, Any]]
    semantic_facts: List[Dict[str, Any]]
    context_embedding: List[float]  # Embedding of the context used to generate this cache
    timestamp: datetime
    context_text: str 

class AsyncMemoryStream:
    """
    Passive streaming memory retrieval that runs asynchronously in background.
    Pre-fetches memories for the next conversation turn based on up to the last 3 turns.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        embedding_service: EmbeddingService
    ):
        self.retriever = retriever
        self.embedding_service = embedding_service

        # Cache for next turn's memories
        self._cache: Optional[MemoryCache] = None

        # Background task management
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Recent conversation context (rolling window)
        self._recent_turns: List[str] = []
        self._max_recent_turns = 3  # Keep up to last 3 turns (earliest to most recent)

        # Track changes for selective refreshing
        self._last_turns_state: List[str] = []

    async def start(self):
        # Mark stream as active
        if self._running:
            return

        self._running = True
        memory_logger.log_event("async_memory_stream_started",{})

    async def stop(self):
        """Stop the background memory streaming task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        memory_logger.log_event("async_memory_stream_stopped")

    def add_turn_context(self, user_message: str, miyori_response: str):
        """
        Add recent conversation turn to context window.
        This triggers ONE background pre-fetch for the next turn.
        """
        turn_text = f"User: {user_message}\nMiyori: {miyori_response}"
        
        self._recent_turns.append(turn_text)
        if len(self._recent_turns) > self._max_recent_turns:
            self._recent_turns.pop(0)
        
        memory_logger.log_event("turn_context_added", {
            "location": "asyncmemorystream"
        })

    def get_cached_memories(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not self._cache:
            memory_logger.log_event("cache_miss",{"reason":"no_cache"})
            return

        memory_logger.log_event("cache_hit", {
            "episodic_count": len(self._cache.episodic_memories),
            "semantic_count": len(self._cache.semantic_facts),
            "cache_age_seconds": (datetime.now() - self._cache.timestamp).total_seconds()
        })
        
        return {
                'episodic': self._cache.episodic_memories,
                'semantic': self._cache.semantic_facts
                }

    async def refresh_cache(self):
        try:
            context_text = " ".join(self._recent_turns)
            if not context_text.strip():
                memory_logger.log_event("cache_refresh_skipped",{"reason":"empty_context"})

            context_changed = True
            if self._cache is not None:
                if self._cache.context_text == context_text:
                    memory_logger.log_event("cache_refresh_skipped", {"reason": "context_unchanged"
                    })
                    return
            
                new_embedding = self.embedding_service.embed(context_text)

                #old_embedding_arr = np.array(self._cache.context_embedding).reshape(1, -1)
                #new_embedding_arr = np.array(new_embedding).reshape(1, -1)

                #from sklearn.metrics.pairwise import cosine_similarity
                #similarity = cosine_similarity(old_embedding_arr, new_embedding_arr)[0, 0]

                #memory_logger.log_event("context_similarity", { 
                #    "similarity": float(similarity), 
                #    "threshold": 0.7
                #})
            else:
                new_embedding = self.embedding_service.embed("context_text")
            
            episodic_memories = self.retriever.search_memories(
                query_embedding=new_embedding,
                search_type='episodic',
                limit=5,
                filters={'status': 'active', 'confidence__gt': 0.5}
            )

            semantic_facts = self.retriever.search_memories(
                query_embedding=new_embedding,
                search_type='semantic',
                limit=5,
                filters={}
            )
            self._cache = MemoryCache(
                episodic_memories=episodic_memories,
                semantic_facts=semantic_facts,
                context_embedding=new_embedding,
                timestamp=datetime.now(),
                context_text = context_text
            )

            memory_logger.log_event("cache_refreshed", {
                "episodic_count": len(self._cache.episodic_memories),
                "semantic_count": len(self._cache.semantic_facts),
                "context_length": len(context_text)
            })

        except Exception as e:
            memory_logger.log_event("cache_refresh_error", {"error": str(e)})
