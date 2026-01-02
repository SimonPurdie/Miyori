"""
Merge Manager for Semantic Fact Deduplication.

Implements three-stage deduplication:
1. Find merge candidates via pairwise similarity
2. Rule-based auto-merge for clear cases
3. LLM validation for ambiguous cases
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity

from miyori.interfaces.memory import IMemoryStore
from miyori.memory.memory_retriever import MemoryRetriever
from miyori.utils.memory_logger import memory_logger


class MergeManager:
    """
    Three-stage deduplication system for semantic facts.
    
    Stage 1: Find merge candidates via embedding similarity clustering
    Stage 2: Rule-based auto-merge for clear duplicates
    Stage 3: LLM validation for ambiguous cases
    """
    
    # Similarity threshold for considering facts as duplicates
    MERGE_SIMILARITY_THRESHOLD = 0.85
    
    # Auto-merge criteria
    MIN_WINNER_CONFIDENCE = 0.6  # At least one fact must have this confidence
    
    def __init__(self, store: IMemoryStore, retriever: MemoryRetriever, llm_client=None):
        self.store = store
        self.retriever = retriever
        self.llm_client = llm_client
        self._llm_queue: List[List[Dict[str, Any]]] = []  # Clusters for LLM review
    
    def run_merge_cycle(self) -> Dict[str, Any]:
        """
        Run complete merge cycle.
        
        Returns:
            Dict with stats: candidates_found, auto_merged, llm_reviewed, etc.
        """
        stats = {
            "total_facts": 0,
            "clusters_found": 0,
            "auto_merged": 0,
            "queued_for_llm": 0,
            "facts_archived": 0
        }
        
        facts = self.store.get_all_active_facts()
        stats["total_facts"] = len(facts)
        
        if len(facts) < 2:
            memory_logger.log_event("merge_cycle", {"message": "Not enough facts to merge"})
            return stats
        
        memory_logger.log_event("merge_cycle_start", {"fact_count": len(facts)})
        
        # Stage 1: Find merge candidates
        clusters = self._find_merge_candidates(facts)
        stats["clusters_found"] = len(clusters)
        
        # Stage 2 & 3: Process each cluster
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            if self._is_auto_mergeable(cluster):
                # Auto-merge
                archived_count = self._execute_auto_merge(cluster)
                stats["auto_merged"] += 1
                stats["facts_archived"] += archived_count
            else:
                # Queue for LLM
                self._queue_for_llm(cluster)
                stats["queued_for_llm"] += 1
        
        # Process LLM queue if we have a client
        if self.llm_client and self._llm_queue:
            llm_results = self._resolve_merges_with_llm()
            stats["facts_archived"] += llm_results.get("archived", 0)
        
        memory_logger.log_event("merge_cycle_complete", stats)
        return stats
    
    def _find_merge_candidates(self, facts: List[Dict[str, Any]], threshold: float = None) -> List[List[Dict[str, Any]]]:
        """
        Stage 1: Find clusters of similar facts using pairwise similarity.
        Uses connected components to group facts that transitively relate.
        """
        if threshold is None:
            threshold = self.MERGE_SIMILARITY_THRESHOLD
        
        # Build embedding matrix
        facts_with_embeddings = []
        embeddings = []
        
        for fact in facts:
            if fact.get('embedding') is not None:
                emb = fact['embedding']
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32)
                elif isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                embeddings.append(emb)
                facts_with_embeddings.append(fact)
        
        if len(embeddings) < 2:
            return []
        
        # Compute pairwise similarities
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find pairs above threshold
        n = len(facts_with_embeddings)
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for start in range(n):
            if start in visited:
                continue
            if not adjacency[start]:  # No connections
                continue
            
            # BFS to find connected component
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(facts_with_embeddings[node])
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(component) > 1:
                clusters.append(component)
        
        memory_logger.log_event("merge_candidates_found", {
            "total_facts": len(facts_with_embeddings),
            "clusters": len(clusters),
            "clustered_facts": sum(len(c) for c in clusters)
        })
        
        return clusters
    
    def _is_auto_mergeable(self, cluster: List[Dict[str, Any]]) -> bool:
        """
        Check if a cluster can be auto-merged without LLM review.
        
        Criteria from spec:
        - Cosine similarity > 0.85 (already ensured by clustering)
        - Same category (if categorized) - not currently implemented
        - No contradiction detected (similar confidence, no opposite meaning)
        - At least one fact has confidence > 0.6
        """
        if not cluster or len(cluster) < 2:
            return False
        
        # Check: at least one fact has high confidence
        max_confidence = max(f['confidence'] for f in cluster)
        if max_confidence < self.MIN_WINNER_CONFIDENCE:
            return False
        
        # Check: no existing contradictions between cluster members
        cluster_ids = {f['id'] for f in cluster}
        for fact in cluster:
            contradictions = set(fact.get('contradictions', []))
            if contradictions & cluster_ids:
                # Contradiction exists within cluster
                return False
        
        # Check: similar language patterns (simple heuristic)
        # If facts are very different in length or structure, might need LLM review
        fact_lengths = [len(f['fact']) for f in cluster]
        if max(fact_lengths) > 3 * min(fact_lengths):
            # Very different lengths suggest potentially different meanings
            return False
        
        return True
    
    def _execute_auto_merge(self, cluster: List[Dict[str, Any]]) -> int:
        """
        Stage 2: Execute rule-based merge.
        
        - Highest confidence fact wins as canonical
        - Merge all derived_from arrays
        - Sum evidence counts
        - Archive losers with merged_into pointer
        
        Returns:
            Number of facts archived
        """
        if len(cluster) < 2:
            return 0
        
        # Find winner (highest confidence)
        winner = max(cluster, key=lambda f: f['confidence'])
        losers = [f for f in cluster if f['id'] != winner['id']]
        
        # Merge evidence lineage
        all_derived_from = set(winner.get('derived_from', []))
        total_evidence_count = winner.get('evidence_count', 0) or 0
        
        for loser in losers:
            all_derived_from.update(loser.get('derived_from', []))
            total_evidence_count += (loser.get('evidence_count', 0) or 0)
        
        # Update winner
        self.store.update_semantic_fact(winner['id'], {
            'derived_from': list(all_derived_from),
            'evidence_count': total_evidence_count,
            'last_confirmed': datetime.now().isoformat()
        })
        
        # Archive losers
        loser_ids = [f['id'] for f in losers]
        self.store.archive_merged_facts(loser_ids, winner['id'])
        
        memory_logger.log_event("auto_merge_executed", {
            "winner_id": winner['id'],
            "winner_fact": winner['fact'][:50],
            "losers_archived": len(loser_ids),
            "merged_evidence_count": total_evidence_count
        })
        
        return len(loser_ids)
    
    def _queue_for_llm(self, cluster: List[Dict[str, Any]]):
        """Queue a cluster for LLM review."""
        self._llm_queue.append(cluster)
    
    def _resolve_merges_with_llm(self) -> Dict[str, Any]:
        """
        Stage 3: LLM validation for ambiguous merge candidates.
        """
        if not self.llm_client or not self._llm_queue:
            return {"archived": 0}
        
        prompt = """You are reviewing potential duplicate facts in a personal knowledge base.
For each cluster of similar facts, decide:
- MERGE: Facts are duplicates; keep the most comprehensive/accurate one
- KEEP_ALL: Facts are distinct and should all be kept
- PARTIAL_MERGE: Some facts should merge, others kept separate

For MERGE decisions, specify which fact should be the "winner" (index 0, 1, etc.)

Respond in JSON format:
{"decisions": [{"cluster_index": 0, "decision": "MERGE", "winner_index": 0, "reason": "..."}]}

Clusters to analyze:
"""
        
        for i, cluster in enumerate(self._llm_queue):
            prompt += f"\nCluster {i}:\n"
            for j, fact in enumerate(cluster):
                prompt += f"  [{j}] \"{fact['fact']}\" (confidence: {fact['confidence']:.2f})\n"
        
        archived_total = 0
        
        try:
            from miyori.utils.config import Config
            model_name = Config.data.get("memory", {}).get("semantic_model", "gemini-2.0-flash")
            
            response = self.llm_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            # Parse response
            import json
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            decisions = json.loads(response_text)
            
            for decision_obj in decisions.get("decisions", []):
                cluster_idx = decision_obj.get("cluster_index", -1)
                decision = decision_obj.get("decision", "KEEP_ALL")
                winner_idx = decision_obj.get("winner_index", 0)
                
                if 0 <= cluster_idx < len(self._llm_queue):
                    cluster = self._llm_queue[cluster_idx]
                    
                    if decision == "MERGE" and len(cluster) > 1:
                        # Validate winner index
                        if 0 <= winner_idx < len(cluster):
                            winner = cluster[winner_idx]
                            losers = [f for j, f in enumerate(cluster) if j != winner_idx]
                            
                            # Merge evidence
                            all_derived_from = set(winner.get('derived_from', []))
                            total_evidence_count = winner.get('evidence_count', 0) or 0
                            
                            for loser in losers:
                                all_derived_from.update(loser.get('derived_from', []))
                                total_evidence_count += (loser.get('evidence_count', 0) or 0)
                            
                            # Update winner
                            self.store.update_semantic_fact(winner['id'], {
                                'derived_from': list(all_derived_from),
                                'evidence_count': total_evidence_count,
                                'last_confirmed': datetime.now().isoformat()
                            })
                            
                            # Archive losers
                            loser_ids = [f['id'] for f in losers]
                            self.store.archive_merged_facts(loser_ids, winner['id'])
                            archived_total += len(loser_ids)
                            
                            memory_logger.log_event("llm_merge_executed", {
                                "cluster_index": cluster_idx,
                                "winner_fact": winner['fact'][:50],
                                "losers_archived": len(loser_ids)
                            })
                    # KEEP_ALL and PARTIAL_MERGE: no automatic action
            
            memory_logger.log_event("llm_merge_resolution", {
                "clusters_processed": len(self._llm_queue),
                "decisions": len(decisions.get("decisions", [])),
                "total_archived": archived_total
            })
            
        except Exception as e:
            memory_logger.log_event("llm_merge_error", {"error": str(e)})
        
        # Clear queue
        self._llm_queue = []
        
        return {"archived": archived_total}
