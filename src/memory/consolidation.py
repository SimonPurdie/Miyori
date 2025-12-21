from google import genai
import json
import numpy as np
from sklearn.cluster import HDBSCAN
#from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple
from src.interfaces.memory import IMemoryStore
from src.memory.deep_layers import SemanticExtractor, EmotionalTracker
from src.utils.config import Config

class EpisodeClustering:
    def __init__(self):
        self.max_cluster_size = Config.data.get("memory", {}).get("max_semantic_extraction_batch_size", 50)
        self.min_cluster_size = Config.data.get("memory", {}).get("min_cluster_size", 3)

    def cluster_episodes(self, episodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster episodes using HDBSCAN based on their embeddings.

        Args:
            episodes: List of episode dictionaries with 'embedding' field
            min_cluster_size: Minimum cluster size for HDBSCAN

        Returns:
            List of clusters (each cluster is a list of episodes)
        """
        if not episodes or len(episodes) < self.min_cluster_size:
            # Return all episodes as singletons if too few for clustering
            return [[episode] for episode in episodes]

        # Extract embeddings
        embeddings = []
        valid_episodes = []

        for episode in episodes:
            if episode.get('embedding') is not None:
                # Handle different embedding formats
                if isinstance(episode['embedding'], bytes):
                    # Convert bytes to numpy array (assuming float32)
                    emb = np.frombuffer(episode['embedding'], dtype=np.float32)
                elif isinstance(episode['embedding'], list):
                    emb = np.array(episode['embedding'], dtype=np.float32)
                elif isinstance(episode['embedding'], np.ndarray):
                    emb = episode['embedding'].astype(np.float32)
                else:
                    continue  # Skip invalid embeddings

                embeddings.append(emb)
                valid_episodes.append(episode)

        if len(valid_episodes) < self.min_cluster_size:
            return [[episode] for episode in valid_episodes]

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='cosine',
            copy=False
        )
        cluster_labels = clusterer.fit_predict(embeddings_array)

        # Group episodes by cluster
        clusters = {}
        for episode, label in zip(valid_episodes, cluster_labels):
            if label == -1:  # Noise points get their own singleton clusters
                clusters[len(clusters)] = [episode]
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(episode)

        return list(clusters.values())

    def create_consolidation_batches(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create batches that pack multiple clusters together while preserving cluster identity.
        
        Returns:
            List of batch dicts: [{'clusters': [cluster1, cluster2, ...], 'total_episodes': N}, ...]
        """
        if not episodes:
            return []
        
        # First, cluster the episodes
        clusters = self.cluster_episodes(episodes)
        
        # Split oversized clusters
        processed_clusters = []
        for cluster in clusters:
            if len(cluster) <= self.max_cluster_size:
                processed_clusters.append(cluster)
            else:
                # Chunk large cluster but mark chunks as related
                for i in range(0, len(cluster), self.max_cluster_size):
                    processed_clusters.append(cluster[i:i + self.max_cluster_size])
        
        # Pack clusters into batches up to max_cluster_size total episodes
        batches = []
        current_batch = []
        current_episode_count = 0
        
        for cluster in processed_clusters:
            cluster_size = len(cluster)
            
            # If adding this cluster exceeds limit, start new batch
            if current_episode_count + cluster_size > self.max_cluster_size and current_batch:
                batches.append({
                    'clusters': current_batch,
                    'total_episodes': current_episode_count
                })
                current_batch = []
                current_episode_count = 0
            
            # Add cluster to current batch
            current_batch.append(cluster)
            current_episode_count += cluster_size
        
        # Add final batch
        if current_batch:
            batches.append({
                'clusters': current_batch,
                'total_episodes': current_episode_count
            })
    
        return batches

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
        self.clustering = EpisodeClustering()

    async def perform_consolidation(self):
        """Nightly consolidation task using clustering for intelligent batching."""
        print("Starting Memory Consolidation...")

        # 1. Get unconsolidated episodes
        episodes = self.store.get_unconsolidated_episodes(status='active')

        if not episodes:
            print("No unconsolidated episodes found.")
            return

        print(f"Found {len(episodes)} unconsolidated episodes.")

        processed_episode_ids = []
        batches = self.clustering.create_consolidation_batches(episodes)
        print(f"Created {len(batches)} batches from {len(episodes)} episodes.")

        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} with {batch['total_episodes']} episodes across {len(batch['clusters'])} clusters...")
        
            # Extract facts with cluster structure preserved
            await self.semantic_extractor.extract_facts_from_batch(batch['clusters'])
            
            # Mark episodes as consolidated
            all_episode_ids = [ep['id'] for cluster in batch['clusters'] for ep in cluster]
            processed_episode_ids.extend(all_episode_ids)

        if processed_episode_ids:
            success = self.store.mark_episodes_consolidated(processed_episode_ids)
            if success:
                print(f"Marked {len(processed_episode_ids)} episodes as consolidated.")
            else:
                print("Warning: Failed to mark some episodes as consolidated.")

        # 5. Analyze relationship patterns (use all episodes for this)
        await self.relational_manager.analyze_relationship(episodes)

        # 6. Cleanup/Archive old mundane ones
        # (Already handled by budget, but can add more specific logic here)

        print("Memory Consolidation Complete.")
