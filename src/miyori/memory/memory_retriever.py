from typing import List, Dict, Any, Optional, Tuple
import json
import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from src.interfaces.memory import IMemoryStore
from src.utils.memory_logger import memory_logger

class MemoryRetriever:
    """
    Shared memory retrieval primitives used by both passive streaming and active tool-based modes.
    Provides vector search, filtering, and diversity sampling.
    """

    def __init__(self, store: IMemoryStore):
        self.store = store

    def vector_search(
        self,
        query_embedding: List[float],
        table: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search on specified table with optional filters.

        Args:
            query_embedding: Query embedding vector
            table: 'episodic_memory' or 'semantic_memory'
            limit: Maximum results to return
            filters: Additional SQL WHERE conditions (e.g., {'status': 'active', 'confidence__gt': 0.5})

        Returns:
            List of matching records with similarity scores
        """
        filters = filters or {}

        # Build SQL query with filters
        where_clauses = []
        params = []

        # Handle status filter
        if 'status' in filters:
            where_clauses.append("status = ?")
            params.append(filters['status'])

        # Handle confidence filter (semantic memory)
        if 'confidence__gt' in filters and table == 'semantic_memory':
            where_clauses.append("confidence > ?")
            params.append(filters['confidence__gt'])

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Execute search
        with self.store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if table == 'episodic_memory':
                cursor.execute(f"""
                    SELECT * FROM episodic_memory
                    WHERE {where_sql} AND embedding IS NOT NULL
                """, params)
            elif table == 'semantic_memory':
                cursor.execute(f"""
                    SELECT * FROM semantic_memory
                    WHERE {where_sql} AND embedding IS NOT NULL
                """, params)
            else:
                raise ValueError(f"Unknown table: {table}")

            rows = cursor.fetchall()

        if not rows:
            return []

        # Build embeddings matrix
        embeddings_matrix = []
        valid_rows = []

        for row in rows:
            embedding_blob = row['embedding']
            if embedding_blob is not None:
                mem_vec = np.frombuffer(embedding_blob, dtype=np.float32)
                embeddings_matrix.append(mem_vec)
                valid_rows.append(row)

        if not embeddings_matrix:
            return []

        # Compute similarities
        embeddings_matrix = np.array(embeddings_matrix)
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

        # Build results
        results = []
        for idx, row in enumerate(valid_rows):
            data = dict(row)
            data['similarity'] = float(similarities[idx])

            # Parse JSON fields
            if table == 'episodic_memory':
                data['full_text'] = json.loads(data['full_text'])
                data['topics'] = json.loads(data['topics'])
                data['entities'] = json.loads(data['entities'])
                data['connections'] = json.loads(data['connections'])
            elif table == 'semantic_memory':
                data['version_history'] = json.loads(data['version_history'])
                data['derived_from'] = json.loads(data['derived_from'])
                data['contradictions'] = json.loads(data['contradictions'])

            results.append(data)

        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def diversity_sample(self, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        Sample diverse results using clustering to avoid redundancy.
        Groups by embedding similarity and takes top result from each cluster.

        Args:
            results: Search results with embeddings
            limit: Maximum results to return

        Returns:
            Diverse subset of results
        """
        if len(results) <= limit:
            return results

        if not results or 'embedding' not in results[0]:
            # Fallback to simple truncation if no embeddings
            return results[:limit]

        # Extract embeddings for clustering
        embeddings = []
        for result in results:
            if result.get('embedding') and isinstance(result['embedding'], bytes):
                vec = np.frombuffer(result['embedding'], dtype=np.float32)
                embeddings.append(vec)
            else:
                # Skip results without embeddings
                continue

        if len(embeddings) < limit:
            return results[:limit]

        embeddings = np.array(embeddings)

        # Determine number of clusters (capped at limit)
        n_clusters = min(limit, len(embeddings))

        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select best result from each cluster
        diverse_results = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_indices:
                # Take the result with highest similarity in this cluster
                best_idx = max(cluster_indices, key=lambda i: results[i]['similarity'])
                diverse_results.append(results[best_idx])

        # Sort by similarity and return
        diverse_results.sort(key=lambda x: x['similarity'], reverse=True)

        memory_logger.log_event("diversity_sample", {
            "original_count": len(results),
            "sampled_count": len(diverse_results),
            "clusters_used": n_clusters
        })

        return diverse_results

    def search_memories(
        self,
        query_embedding: List[float],
        search_type: str = 'episodic',
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search single memory type with diversity sampling.

        Args:
            query_embedding: Query embedding vector
            search_type: 'episodic' or 'semantic'
            limit: Maximum results to return
            filters: Additional filters

        Returns:
            List of matching memories
        """
        if search_type not in ['episodic', 'semantic']:
            raise ValueError(f"search_type must be 'episodic' or 'semantic', got '{search_type}'")

        table = 'episodic_memory' if search_type == 'episodic' else 'semantic_memory'

        raw_results = self.vector_search(
            query_embedding,
            table,
            limit=limit * 3,  # Over-fetch for diversity
            filters=filters
        )

        return self.diversity_sample(raw_results, limit)
