from typing import List, Dict, Any
from src.interfaces.memory import IMemoryStore
from src.memory.scoring import ImportanceScorer
from src.utils.memory_logger import memory_logger

class MemoryBudget:
    def __init__(self, store: IMemoryStore, config: Dict[str, Any]):
        self.store = store
        self.max_active = config.get('max_episodic_active', 1000)
        self.check_frequency = 50
        self.add_count = 0

    def enforce_if_needed(self):
        """Perform budget check and pruning if enough episodes have been added."""
        self.add_count += 1
        if self.add_count >= self.check_frequency:
            self.enforce()
            self.add_count = 0

    def enforce(self):
        """Prune least important/oldest memories if over budget."""
        # Get all active episodes (for Phase 2, we just get them all)
        # In a real large-scale system, we'd use a more efficient count/query
        all_active = self.store.search_episodes([0.0]*768, limit=self.max_active + 100, status='active')
        
        if len(all_active) <= self.max_active:
            return

        print(f"Memory Budget: {len(all_active)} episodes. Pruning back to {self.max_active}...")

        # Rank by: similarity (not applicable here, use 1.0) * 0.5 + decayed_importance * 0.3 + recency * 0.2
        # But for pruning, we just care about decayed_importance and recency.
        
        for mem in all_active:
            importance = mem.get('importance', 0.5)
            decayed_importance = ImportanceScorer.get_decayed_score(importance, mem['timestamp'])
            
            # Recency (using simpler calc for budget)
            from datetime import datetime
            timestamp = datetime.fromisoformat(mem['timestamp'])
            age_days = (datetime.now() - timestamp).days
            recency = 1.0 / (1 + age_days / 30)
            
            # Ranking score: 0.6*importance + 0.4*recency
            mem['budget_rank'] = (decayed_importance * 0.6) + (recency * 0.4)

        # Sort by budget_rank (keep highest)
        all_active.sort(key=lambda x: x['budget_rank'], reverse=True)
        
        to_keep = all_active[:self.max_active]
        to_archive = all_active[self.max_active:]
        
        for episode in to_archive:
            self.store.update_episode(episode['id'], {'status': 'archived'})
        
        memory_logger.log_event("budget_pruning", {
            "initial_count": len(all_active),
            "archived_count": len(to_archive),
            "max_active": self.max_active
        })
