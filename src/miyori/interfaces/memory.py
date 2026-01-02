from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class IMemoryStore(ABC):
    """Interface for cognitive memory storage backends."""
    
    @abstractmethod
    def add_episode(self, episode_data: Dict[str, Any]) -> str:
        """Store a new episodic memory (conversation turn)."""
        pass

    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific episode by ID."""
        pass

    @abstractmethod
    def update_episode(self, episode_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of an existing episode (e.g., status, embedding)."""
        pass

    @abstractmethod
    def search_episodes(self, query_embedding: List[float], limit: int = 5, status: str = 'active') -> List[Dict[str, Any]]:
        """Search episodes by semantic similarity."""
        pass

    @abstractmethod
    def get_unconsolidated_episodes(self, status: str = 'active', limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get episodes that haven't been consolidated yet."""
        pass

    @abstractmethod
    def mark_episodes_consolidated(self, episode_ids: List[str]) -> bool:
        """Mark episodes as consolidated by setting consolidated_at timestamp."""
        pass

    @abstractmethod
    def add_semantic_fact(self, fact_data: Dict[str, Any]) -> str:
        """Store or update a semantic fact."""
        pass

    @abstractmethod
    def get_semantic_facts(self, status: str = 'stable', limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve established facts about the user."""
        pass

    @abstractmethod
    def update_semantic_fact(self, fact_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of an existing semantic fact (e.g., confidence, status)."""
        pass

    @abstractmethod
    def get_all_active_facts(self, min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all active/stable facts for batch operations like merging."""
        pass

    @abstractmethod
    def archive_merged_facts(self, loser_ids: List[str], winner_id: str) -> bool:
        """Archive facts that have been merged into a canonical fact."""
        pass
