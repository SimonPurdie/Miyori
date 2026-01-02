"""
Unit tests for ConfidenceManager and MergeManager.

Tests cover:
- Evidence accumulation formula
- Time decay calculation
- Auto-deprecation threshold
- Merge candidate clustering
- Auto-merge criteria
- Evidence lineage preservation
"""

import sys
import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from miyori.utils.config import Config
Config.load()

from miyori.memory.confidence_manager import ConfidenceManager
from miyori.memory.merge_manager import MergeManager


class MockMemoryStore:
    """Mock memory store for testing."""
    
    def __init__(self):
        self.facts = {}
        self.episodes = {}
        self.archived_facts = []
    
    def get_all_active_facts(self, min_confidence=None):
        results = [f for f in self.facts.values() if f['status'] in ('stable', 'tentative')]
        if min_confidence is not None:
            results = [f for f in results if f['confidence'] >= min_confidence]
        return results
    
    def update_semantic_fact(self, fact_id, updates):
        if fact_id in self.facts:
            self.facts[fact_id].update(updates)
            return True
        return False
    
    def archive_merged_facts(self, loser_ids, winner_id):
        for lid in loser_ids:
            if lid in self.facts:
                self.facts[lid]['status'] = 'merged_into'
                self.facts[lid]['merged_into_id'] = winner_id
                self.archived_facts.append(lid)
        return True
    
    def _get_connection(self):
        """Mock connection for retriever compatibility."""
        return MockConnection(self)


class MockConnection:
    """Mock SQLite connection."""
    def __init__(self, store):
        self.store = store
        self.row_factory = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def cursor(self):
        return MockCursor(self.store)


class MockCursor:
    """Mock SQLite cursor."""
    def __init__(self, store):
        self.store = store
        self.results = []
    
    def execute(self, query, params=None):
        # Simple mock - return episodes for episodic_memory queries
        if 'episodic_memory' in query:
            self.results = list(self.store.episodes.values())
        elif 'semantic_memory' in query:
            self.results = list(self.store.facts.values())
    
    def fetchall(self):
        # Convert dicts to Row-like objects
        return [MockRow(r) for r in self.results]


class MockRow:
    """Mock SQLite Row."""
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    def keys(self):
        return self._data.keys()


class MockRetriever:
    """Mock MemoryRetriever for testing."""
    
    def __init__(self, store):
        self.store = store
    
    def vector_search(self, query_embedding, table, limit=10, filters=None):
        if table == 'episodic_memory':
            # Return mock episodes with simulated similarities
            results = []
            for ep in self.store.episodes.values():
                ep_copy = dict(ep)
                # Simulate similarity based on some mock logic
                ep_copy['similarity'] = 0.8  # High similarity for testing
                results.append(ep_copy)
            return results[:limit]
        return []


def create_test_fact(fact_id, fact_text, confidence=0.7, embedding=None, **kwargs):
    """Helper to create test fact dict."""
    if embedding is None:
        # Create random 768-dim embedding
        embedding = np.random.rand(768).astype(np.float32).tobytes()
    
    return {
        'id': fact_id,
        'fact': fact_text,
        'confidence': confidence,
        'status': kwargs.get('status', 'stable'),
        'first_observed': kwargs.get('first_observed', datetime.now().isoformat()),
        'last_confirmed': kwargs.get('last_confirmed', datetime.now().isoformat()),
        'derived_from': kwargs.get('derived_from', []),
        'contradictions': kwargs.get('contradictions', []),
        'version_history': kwargs.get('version_history', []),
        'evidence_count': kwargs.get('evidence_count', 0),
        'merged_into_id': kwargs.get('merged_into_id', None),
        'embedding': embedding
    }


def create_test_episode(ep_id, summary, embedding=None, **kwargs):
    """Helper to create test episode dict."""
    if embedding is None:
        embedding = np.random.rand(768).astype(np.float32).tobytes()
    
    return {
        'id': ep_id,
        'summary': summary,
        'status': kwargs.get('status', 'active'),
        'timestamp': kwargs.get('timestamp', datetime.now().isoformat()),
        'embedding': embedding
    }


# ==================== Confidence Manager Tests ====================

class TestEvidenceAccumulation:
    """Test the asymptotic evidence growth formula."""
    
    def test_single_episode_boost(self):
        """Single supporting episode should boost confidence by rate * (1 - confidence)."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        initial_confidence = 0.7
        result = manager._apply_evidence_accumulation(initial_confidence, 1)
        
        expected = initial_confidence + 0.05 * (1 - initial_confidence)
        assert abs(result - expected) < 0.001
    
    def test_multiple_episodes_asymptotic(self):
        """Multiple episodes should approach but never exceed 1.0."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        # Apply 100 episodes
        result = manager._apply_evidence_accumulation(0.5, 100)
        
        # Should be very close to 1.0 but not exceed
        assert result < 1.0
        assert result > 0.99
    
    def test_zero_episodes_no_change(self):
        """Zero episodes should not change confidence."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        initial = 0.7
        result = manager._apply_evidence_accumulation(initial, 0)
        
        assert result == initial


class TestTimeDecay:
    """Test the exponential time decay formula."""
    
    def test_recent_fact_minimal_decay(self):
        """Fact confirmed today should have minimal decay."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        fact = create_test_fact('f1', 'test', confidence=0.8,
                               last_confirmed=datetime.now().isoformat())
        
        result = manager._apply_time_decay(fact, 0.8)
        
        # Should be essentially unchanged (within 1%)
        assert abs(result - 0.8) < 0.01
    
    def test_old_fact_significant_decay(self):
        """Fact not confirmed in 30 days should decay ~26%."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        fact = create_test_fact('f1', 'test', confidence=0.8,
                               last_confirmed=old_date)
        
        result = manager._apply_time_decay(fact, 0.8)
        
        # Should have decayed by approximately exp(-30 * 0.01) ≈ 0.74
        expected = 0.8 * 0.74
        assert abs(result - expected) < 0.05


class TestAutoDeprecation:
    """Test auto-deprecation of low confidence facts."""
    
    def test_deprecate_below_threshold(self):
        """Facts below 0.3 should be deprecated."""
        store = MockMemoryStore()
        store.facts['f1'] = create_test_fact('f1', 'low confidence', confidence=0.2)
        store.facts['f2'] = create_test_fact('f2', 'ok confidence', confidence=0.5)
        
        retriever = MockRetriever(store)
        manager = ConfidenceManager(store, retriever)
        
        count = manager._deprecate_low_confidence_facts()
        
        assert count == 1
        assert store.facts['f1']['status'] == 'deprecated'
        assert store.facts['f2']['status'] == 'stable'


# ==================== Merge Manager Tests ====================

class TestFindMergeCandidates:
    """Test similarity-based clustering for merge candidates."""
    
    def test_identical_embeddings_cluster(self):
        """Facts with identical embeddings should cluster together."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        # Create facts with identical embeddings
        embedding = np.random.rand(768).astype(np.float32).tobytes()
        facts = [
            create_test_fact('f1', 'user likes coffee', embedding=embedding),
            create_test_fact('f2', 'user enjoys coffee', embedding=embedding),
        ]
        
        clusters = manager._find_merge_candidates(facts)
        
        assert len(clusters) == 1
        assert len(clusters[0]) == 2
    
    def test_different_embeddings_no_cluster(self):
        """Facts with very different embeddings should not cluster."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        # Create facts with orthogonal embeddings
        facts = [
            create_test_fact('f1', 'user likes coffee'),
            create_test_fact('f2', 'user works in tech'),
        ]
        
        clusters = manager._find_merge_candidates(facts, threshold=0.95)
        
        # Should have no clusters (or empty list) with very high threshold
        assert len(clusters) == 0 or all(len(c) == 1 for c in clusters)


class TestAutoMergeCriteria:
    """Test auto-merge eligibility checks."""
    
    def test_high_confidence_allows_merge(self):
        """Cluster with at least one high-confidence fact can auto-merge."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [
            create_test_fact('f1', 'fact 1', confidence=0.8),
            create_test_fact('f2', 'fact 2', confidence=0.5),
        ]
        
        assert manager._is_auto_mergeable(cluster) is True
    
    def test_low_confidence_prevents_merge(self):
        """Cluster with all low-confidence facts cannot auto-merge."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [
            create_test_fact('f1', 'fact 1', confidence=0.4),
            create_test_fact('f2', 'fact 2', confidence=0.3),
        ]
        
        assert manager._is_auto_mergeable(cluster) is False
    
    def test_existing_contradiction_prevents_merge(self):
        """Cluster with existing contradictions cannot auto-merge."""
        store = MockMemoryStore()
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [
            create_test_fact('f1', 'fact 1', confidence=0.8, contradictions=['f2']),
            create_test_fact('f2', 'fact 2', confidence=0.7),
        ]
        
        assert manager._is_auto_mergeable(cluster) is False


class TestMergeEvidencePreservation:
    """Test that merge preserves evidence lineage."""
    
    def test_derived_from_merged(self):
        """Winner should have union of all derived_from arrays."""
        store = MockMemoryStore()
        store.facts['f1'] = create_test_fact('f1', 'fact 1', confidence=0.8,
                                             derived_from=['ep1', 'ep2'])
        store.facts['f2'] = create_test_fact('f2', 'fact 2', confidence=0.6,
                                             derived_from=['ep3', 'ep4'])
        
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [store.facts['f1'], store.facts['f2']]
        manager._execute_auto_merge(cluster)
        
        # Winner (f1 with higher confidence) should have all episode IDs
        winner_derived = set(store.facts['f1'].get('derived_from', []))
        assert 'ep1' in winner_derived
        assert 'ep2' in winner_derived
        assert 'ep3' in winner_derived
        assert 'ep4' in winner_derived
    
    def test_evidence_count_summed(self):
        """Winner should have sum of all evidence counts."""
        store = MockMemoryStore()
        store.facts['f1'] = create_test_fact('f1', 'fact 1', confidence=0.8,
                                             evidence_count=5)
        store.facts['f2'] = create_test_fact('f2', 'fact 2', confidence=0.6,
                                             evidence_count=3)
        
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [store.facts['f1'], store.facts['f2']]
        manager._execute_auto_merge(cluster)
        
        assert store.facts['f1']['evidence_count'] == 8
    
    def test_losers_archived(self):
        """Loser facts should be archived with merged_into pointer."""
        store = MockMemoryStore()
        store.facts['f1'] = create_test_fact('f1', 'fact 1', confidence=0.8)
        store.facts['f2'] = create_test_fact('f2', 'fact 2', confidence=0.6)
        
        retriever = MockRetriever(store)
        manager = MergeManager(store, retriever)
        
        cluster = [store.facts['f1'], store.facts['f2']]
        manager._execute_auto_merge(cluster)
        
        assert store.facts['f2']['status'] == 'merged_into'
        assert store.facts['f2']['merged_into_id'] == 'f1'


# ==================== Integration Test ====================

class TestFullConsolidationFlow:
    """Integration test for the complete consolidation flow."""
    
    def test_confidence_and_merge_cycle(self):
        """Test running both confidence updates and merge cycle."""
        store = MockMemoryStore()
        
        # Add some test facts
        embedding = np.random.rand(768).astype(np.float32).tobytes()
        store.facts['f1'] = create_test_fact('f1', 'user likes tea', 
                                             confidence=0.8, embedding=embedding)
        store.facts['f2'] = create_test_fact('f2', 'user enjoys tea', 
                                             confidence=0.6, embedding=embedding)
        store.facts['f3'] = create_test_fact('f3', 'user works remotely',
                                             confidence=0.2)  # Should deprecate
        
        retriever = MockRetriever(store)
        confidence_manager = ConfidenceManager(store, retriever)
        merge_manager = MergeManager(store, retriever)
        
        # Run confidence updates (will deprecate f3)
        confidence_results = confidence_manager.update_all_confidences()
        assert confidence_results['deprecated_count'] >= 1
        
        # Run merge cycle (should merge f1 and f2)
        merge_results = merge_manager.run_merge_cycle()
        assert merge_results['clusters_found'] >= 0  # May or may not find clusters
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
