from google import genai
import json
import numpy as np
import hdbscan
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
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='cosine'
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

    def split_large_clusters(self, clusters: List[List[Dict[str, Any]]],
                           max_size: int = None) -> List[List[Dict[str, Any]]]:
        """
        Split clusters that are too large by re-clustering them with higher min_cluster_size.

        Args:
            clusters: List of clusters to potentially split
            max_size: Maximum allowed cluster size

        Returns:
            List of clusters, with large ones split into smaller ones
        """

        result_clusters = []

        for cluster in clusters:
            if len(cluster) <= self.max_cluster_size:
                result_clusters.append(cluster)
            else:
                # Split large cluster by increasing min_cluster_size
                # Start with min_cluster_size = max_size // 2 + 1
                min_size = max(self.max_cluster_size // 2 + 1, 2)

                # Try progressively larger min_cluster_size until clusters are small enough
                while min_size <= self.max_cluster_size:
                    subclusters = self.cluster_episodes(cluster)

                    # Check if all subclusters are now small enough
                    if all(len(subcluster) <= self.max_cluster_size for subcluster in subclusters):
                        result_clusters.extend(subclusters)
                        break

                    min_size += 1
                else:
                    # If we couldn't split effectively, just chunk the cluster
                    # This is a fallback for when clustering doesn't work well
                    for i in range(0, len(cluster), self.max_cluster_size):
                        result_clusters.append(cluster[i:i + self.max_cluster_size])

        return result_clusters

    def create_consolidation_batches(self, episodes: List[Dict[str, Any]],
                                   max_cluster_batch_size: int = None) -> List[List[Dict[str, Any]]]:
        """
        Create consolidation batches by clustering episodes and ensuring no batch exceeds max_cluster_batch_size.

        Args:
            episodes: Episodes to batch
            max_cluster_batch_size: Maximum episodes per batch

        Returns:
            List of batches (each batch is a list of episodes)
        """
        if not episodes:
            return []

        # First, cluster the episodes
        clusters = self.cluster_episodes(episodes)

        # Then split any clusters that are too large
        batches = self.split_large_clusters(clusters, self.max_cluster_size)

        # Final safety check: ensure no batch exceeds max_cluster_size
        final_batches = []
        for batch in batches:
            if len(batch) <= self.max_cluster_size:
                final_batches.append(batch)
            else:
                # Emergency chunking
                for i in range(0, len(batch), self.max_cluster_size):
                    final_batches.append(batch[i:i + self.max_cluster_size])

        return final_batches

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

        # 2. Create consolidation batches using clustering
        batches = self.clustering.create_consolidation_batches(episodes)
        print(f"Created {len(batches)} consolidation batches.")

        # 3. Process each batch
        processed_episode_ids = []

        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} episodes...")

            try:
                # Extract semantic facts from this batch
                await self.semantic_extractor.extract_facts_batched(batch)

                # Collect episode IDs for marking as consolidated
                batch_ids = [episode['id'] for episode in batch]
                processed_episode_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                # Continue with other batches even if one fails

        # 4. Mark processed episodes as consolidated
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
