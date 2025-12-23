from google import genai
import json
import numpy as np
from sklearn.cluster import HDBSCAN
#from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple
from miyori.interfaces.memory import IMemoryStore
from miyori.memory.deep_layers import SemanticExtractor
from miyori.utils.config import Config

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

class ContradictionDetector:
    def __init__(self, store: IMemoryStore):
        self.store = store

    def detect_conflicts(self, new_fact: str) -> List[Dict[str, Any]]:
        """Check if a new fact contradicts existing semantic memory."""
        # NONFUNCTIONAL PLACEHOLDER
        facts = self.store.get_semantic_facts()
        conflicts = []
            
        return conflicts

class ConsolidationManager:
    def __init__(self, store, episodic_manager, semantic_extractor):
        self.store = store
        self.episodic_manager = episodic_manager
        self.semantic_extractor = semantic_extractor
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

        # 6. Cleanup/Archive old mundane ones
        # (Already handled by budget, but can add more specific logic here)

        print("Memory Consolidation Complete.")
