"""
Confidence Management System for Semantic Facts.

Implements rule-based confidence updates:
- Evidence accumulation (asymptotic growth)
- Time decay (exponential)
- Contradiction detection and handling
"""

import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from miyori.interfaces.memory import IMemoryStore
from miyori.memory.memory_retriever import MemoryRetriever
from miyori.utils.memory_logger import memory_logger


class ConfidenceManager:
    """
    Rule-based confidence update system for semantic facts.
    
    Runs during nightly consolidation to:
    1. Find supporting episodes for each fact (evidence accumulation)
    2. Apply time decay for facts without recent evidence
    3. Detect and handle contradictions between facts
    """
    
    # Confidence thresholds from spec
    DEPRECATION_THRESHOLD = 0.3    # Below this: auto-deprecate
    STREAMING_THRESHOLD = 0.5      # Below this: excluded from passive streaming
    
    # Decay/growth parameters
    EVIDENCE_GROWTH_RATE = 0.05    # Asymptotic growth per supporting episode
    DAILY_DECAY_RATE = 0.01        # ~10% decay per 10 days
    CONTRADICTION_PENALTY = 0.5    # Reduce both facts by this factor
    
    # Similarity thresholds
    CONTRADICTION_SIMILARITY = 0.85  # High similarity suggests potential contradiction
    SUPPORTING_EPISODE_THRESHOLD = 0.75
    
    def __init__(self, store: IMemoryStore, retriever: MemoryRetriever, llm_client=None):
        self.store = store
        self.retriever = retriever
        self.llm_client = llm_client  # For LLM-based contradiction resolution
        self._llm_queue: List[Tuple[str, str, str]] = []  # (fact_id_1, fact_id_2, reason)
    
    def update_all_confidences(self) -> Dict[str, Any]:
        """
        Nightly batch update for all active facts.
        
        Returns:
            Dict with stats: updated_count, deprecated_count, contradictions_found, etc.
        """
        stats = {
            "updated_count": 0,
            "deprecated_count": 0,
            "evidence_boosts": 0,
            "time_decays": 0,
            "clear_contradictions": 0,
            "ambiguous_contradictions": 0
        }
        
        facts = self.store.get_all_active_facts()
        if not facts:
            memory_logger.log_event("confidence_update", {"message": "No active facts to update"})
            return stats
        
        memory_logger.log_event("confidence_update_start", {"fact_count": len(facts)})
        
        # Step 1: Evidence accumulation for each fact
        for fact in facts:
            if fact.get('embedding') is None:
                continue
                
            original_confidence = fact['confidence']
            new_confidence = original_confidence
            
            # Find supporting episodes since last update
            supporting_episodes = self._find_supporting_episodes(fact)
            if supporting_episodes:
                new_confidence = self._apply_evidence_accumulation(
                    new_confidence, len(supporting_episodes)
                )
                stats["evidence_boosts"] += 1
                
                # Update evidence count and derived_from
                new_evidence_count = (fact.get('evidence_count') or 0) + len(supporting_episodes)
                existing_derived = fact.get('derived_from', [])
                new_derived = list(set(existing_derived + supporting_episodes))
                
                self.store.update_semantic_fact(fact['id'], {
                    'evidence_count': new_evidence_count,
                    'derived_from': new_derived,
                    'last_confirmed': datetime.now().isoformat()
                })
            else:
                # Apply time decay if no new evidence
                new_confidence = self._apply_time_decay(fact, new_confidence)
                stats["time_decays"] += 1
            
            # Update confidence if changed
            if abs(new_confidence - original_confidence) > 0.001:
                self.store.update_semantic_fact(fact['id'], {'confidence': new_confidence})
                stats["updated_count"] += 1
        
        # Step 2: Detect contradictions between facts
        contradictions = self._detect_contradictions(facts)
        
        for fact_id_1, fact_id_2, severity in contradictions:
            if severity == "clear":
                self._handle_clear_contradiction(fact_id_1, fact_id_2)
                stats["clear_contradictions"] += 1
            else:
                self._queue_for_llm_resolution(fact_id_1, fact_id_2, "potential_contradiction")
                stats["ambiguous_contradictions"] += 1
        
        # Step 3: Process LLM queue if we have a client
        if self.llm_client and self._llm_queue:
            self._resolve_contradictions_with_llm()
        
        # Step 4: Auto-deprecate facts below threshold
        stats["deprecated_count"] = self._deprecate_low_confidence_facts()
        
        memory_logger.log_event("confidence_update_complete", stats)
        return stats
    
    def _find_supporting_episodes(self, fact: Dict[str, Any]) -> List[str]:
        """
        Vector search for episodes that support a fact.
        Only looks at episodes since the fact's last confirmation.
        """
        embedding = fact.get('embedding')
        if embedding is None:
            return []
        
        # Convert bytes to numpy if needed
        if isinstance(embedding, bytes):
            embedding = np.frombuffer(embedding, dtype=np.float32).tolist()
        
        last_confirmed = fact.get('last_confirmed', fact.get('first_observed'))
        
        # Search for similar episodes
        results = self.retriever.vector_search(
            query_embedding=embedding,
            table='episodic_memory',
            limit=20,
            filters={'status': 'active'}
        )
        
        # Filter by timestamp and similarity threshold
        supporting = []
        for episode in results:
            if episode['similarity'] >= self.SUPPORTING_EPISODE_THRESHOLD:
                # Check if episode is newer than last confirmation
                episode_time = episode.get('timestamp')
                if episode_time and last_confirmed:
                    try:
                        if episode_time > last_confirmed:
                            supporting.append(episode['id'])
                    except (TypeError, ValueError):
                        # If timestamp comparison fails, include anyway
                        supporting.append(episode['id'])
                else:
                    supporting.append(episode['id'])
        
        return supporting
    
    def _apply_evidence_accumulation(self, confidence: float, episode_count: int) -> float:
        """
        Asymptotic confidence growth: confidence += rate * (1 - confidence) per episode
        Prevents runaway confidence from many weak signals.
        """
        for _ in range(episode_count):
            confidence += self.EVIDENCE_GROWTH_RATE * (1 - confidence)
        return min(confidence, 1.0)
    
    def _apply_time_decay(self, fact: Dict[str, Any], confidence: float) -> float:
        """
        Exponential decay: confidence *= exp(-days * decay_rate)
        """
        last_confirmed = fact.get('last_confirmed', fact.get('first_observed'))
        if not last_confirmed:
            return confidence
        
        try:
            last_dt = datetime.fromisoformat(last_confirmed.replace('Z', '+00:00'))
            days_since = (datetime.now() - last_dt.replace(tzinfo=None)).days
            
            if days_since > 0:
                decay_factor = math.exp(-days_since * self.DAILY_DECAY_RATE)
                confidence *= decay_factor
        except (TypeError, ValueError) as e:
            memory_logger.log_event("time_decay_error", {"fact_id": fact['id'], "error": str(e)})
        
        return max(confidence, 0.0)
    
    def _detect_contradictions(self, facts: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Find pairs of facts that may contradict each other.
        Uses high embedding similarity as a signal that facts discuss the same topic.
        """
        contradictions = []
        
        # Build embedding matrix for all facts with embeddings
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
            return contradictions
        
        # Compute pairwise similarities
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find high-similarity pairs
        for i in range(len(facts_with_embeddings)):
            for j in range(i + 1, len(facts_with_embeddings)):
                sim = similarity_matrix[i, j]
                
                if sim >= self.CONTRADICTION_SIMILARITY:
                    fact_1 = facts_with_embeddings[i]
                    fact_2 = facts_with_embeddings[j]
                    
                    # Heuristic: if both facts have similar confidence, it's ambiguous
                    # If one is much higher, the lower confidence one might be outdated
                    conf_diff = abs(fact_1['confidence'] - fact_2['confidence'])
                    
                    # Simple heuristic for "clear" vs "ambiguous"
                    # Clear: very high similarity and both have decent confidence
                    if sim > 0.92 and min(fact_1['confidence'], fact_2['confidence']) > 0.5:
                        severity = "clear"
                    else:
                        severity = "ambiguous"
                    
                    # Check if already recorded in contradictions field
                    existing_contradictions = set(fact_1.get('contradictions', []) + fact_2.get('contradictions', []))
                    if fact_2['id'] not in existing_contradictions and fact_1['id'] not in existing_contradictions:
                        contradictions.append((fact_1['id'], fact_2['id'], severity))
        
        return contradictions
    
    def _handle_clear_contradiction(self, fact_id_1: str, fact_id_2: str):
        """
        Auto-reduce both facts' confidence by 50% for clear contradictions.
        Also records the contradiction in both facts.
        """
        for fact_id, other_id in [(fact_id_1, fact_id_2), (fact_id_2, fact_id_1)]:
            # Get current fact data
            facts = self.store.get_all_active_facts()
            fact = next((f for f in facts if f['id'] == fact_id), None)
            
            if fact:
                new_confidence = fact['confidence'] * self.CONTRADICTION_PENALTY
                existing_contradictions = fact.get('contradictions', [])
                if other_id not in existing_contradictions:
                    existing_contradictions.append(other_id)
                
                self.store.update_semantic_fact(fact_id, {
                    'confidence': new_confidence,
                    'contradictions': existing_contradictions
                })
        
        memory_logger.log_event("clear_contradiction_handled", {
            "fact_1": fact_id_1,
            "fact_2": fact_id_2
        })
    
    def _queue_for_llm_resolution(self, fact_id_1: str, fact_id_2: str, reason: str):
        """Queue ambiguous contradictions for LLM batch resolution."""
        self._llm_queue.append((fact_id_1, fact_id_2, reason))
    
    def _resolve_contradictions_with_llm(self):
        """
        Batch LLM call to resolve queued contradictions.
        """
        if not self.llm_client or not self._llm_queue:
            return
        
        # Get fact texts for all queued contradictions
        all_facts = {f['id']: f for f in self.store.get_all_active_facts()}
        
        prompt = """You are analyzing potential contradictions in a personal knowledge base.
For each pair of facts, decide:
- KEEP_BOTH: Facts are compatible or context-dependent (e.g., preferences changed over time)
- CONTRADICTION: Facts genuinely contradict; reduce confidence in both
- SUPERSEDED: The newer/higher-confidence fact supersedes the older one

Respond in JSON format: {"decisions": [{"pair_index": 0, "decision": "KEEP_BOTH", "reason": "..."}]}

Pairs to analyze:
"""
        
        for i, (fact_id_1, fact_id_2, reason) in enumerate(self._llm_queue):
            fact_1 = all_facts.get(fact_id_1, {})
            fact_2 = all_facts.get(fact_id_2, {})
            prompt += f"\nPair {i}:\n"
            prompt += f"  Fact A: \"{fact_1.get('fact', 'N/A')}\" (confidence: {fact_1.get('confidence', 0):.2f})\n"
            prompt += f"  Fact B: \"{fact_2.get('fact', 'N/A')}\" (confidence: {fact_2.get('confidence', 0):.2f})\n"
        
        try:
            from miyori.utils.config import Config
            model_name = Config.data.get("memory", {}).get("semantic_model", "gemini-2.0-flash")
            
            response = self.llm_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            # Parse response and apply decisions
            import json
            response_text = response.text.strip()
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            decisions = json.loads(response_text)
            
            for decision_obj in decisions.get("decisions", []):
                pair_idx = decision_obj.get("pair_index", -1)
                decision = decision_obj.get("decision", "KEEP_BOTH")
                
                if 0 <= pair_idx < len(self._llm_queue):
                    fact_id_1, fact_id_2, _ = self._llm_queue[pair_idx]
                    
                    if decision == "CONTRADICTION":
                        self._handle_clear_contradiction(fact_id_1, fact_id_2)
                    elif decision == "SUPERSEDED":
                        # Keep higher confidence, lower the other
                        fact_1 = all_facts.get(fact_id_1)
                        fact_2 = all_facts.get(fact_id_2)
                        if fact_1 and fact_2:
                            if fact_1['confidence'] > fact_2['confidence']:
                                self.store.update_semantic_fact(fact_id_2, {
                                    'confidence': fact_2['confidence'] * 0.5
                                })
                            else:
                                self.store.update_semantic_fact(fact_id_1, {
                                    'confidence': fact_1['confidence'] * 0.5
                                })
                    # KEEP_BOTH: no action needed
            
            memory_logger.log_event("llm_contradiction_resolution", {
                "pairs_processed": len(self._llm_queue),
                "decisions": len(decisions.get("decisions", []))
            })
            
        except Exception as e:
            memory_logger.log_event("llm_contradiction_error", {"error": str(e)})
        
        # Clear queue after processing
        self._llm_queue = []
    
    def _deprecate_low_confidence_facts(self) -> int:
        """Auto-deprecate facts below the deprecation threshold."""
        facts = self.store.get_all_active_facts()
        deprecated_count = 0
        
        for fact in facts:
            if fact['confidence'] < self.DEPRECATION_THRESHOLD:
                self.store.update_semantic_fact(fact['id'], {'status': 'deprecated'})
                deprecated_count += 1
                memory_logger.log_event("fact_deprecated", {
                    "fact_id": fact['id'],
                    "fact": fact['fact'][:50],
                    "confidence": fact['confidence']
                })
        
        return deprecated_count
