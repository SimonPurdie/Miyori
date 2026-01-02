import sqlite3
import json
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from miyori.utils.config import Config
from miyori.interfaces.memory import IMemoryStore

class SQLiteMemoryStore(IMemoryStore):
    def __init__(self):
        self.db_path = Config.get_project_root() / "memory.db"
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Episodic Memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    summary TEXT,
                    full_text TEXT, -- JSON string
                    timestamp DATETIME,
                    embedding BLOB,
                    importance REAL,
                    topics TEXT,    -- JSON string
                    entities TEXT,  -- JSON string
                    connections TEXT, -- JSON string
                    status TEXT,
                    consolidated_at DATETIME
                )
            """)

            # Semantic Memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id TEXT PRIMARY KEY,
                    fact TEXT,
                    confidence REAL,
                    first_observed DATETIME,
                    last_confirmed DATETIME,
                    version_history TEXT, -- JSON string
                    derived_from TEXT,    -- JSON string
                    contradictions TEXT,  -- JSON string
                    status TEXT,
                    embedding BLOB,
                    evidence_count INTEGER DEFAULT 0,
                    merged_into_id TEXT DEFAULT NULL
                )
            """)

            # Schema Version
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            cursor.execute("SELECT COUNT(*) FROM schema_version")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
            
            # Add new columns if they don't exist (for existing databases)
            self._migrate_semantic_memory_columns(cursor)
            
            conn.commit()

    def _migrate_semantic_memory_columns(self, cursor):
        """Add new columns to semantic_memory if they don't exist."""
        # Check existing columns
        cursor.execute("PRAGMA table_info(semantic_memory)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        if 'evidence_count' not in existing_columns:
            cursor.execute("ALTER TABLE semantic_memory ADD COLUMN evidence_count INTEGER DEFAULT 0")
        
        if 'merged_into_id' not in existing_columns:
            cursor.execute("ALTER TABLE semantic_memory ADD COLUMN merged_into_id TEXT DEFAULT NULL")

    def add_episode(self, episode_data: Dict[str, Any]) -> str:
        episode_id = episode_data.get('id') or str(uuid.uuid4())
        timestamp = episode_data.get('timestamp') or datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memory (
                    id, summary, full_text, timestamp, embedding,
                    importance, topics, entities,
                    connections, status, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                episode_data.get('summary'),
                json.dumps(episode_data.get('full_text', {})),
                timestamp,
                episode_data.get('embedding'), # Should be bytes/BLOB
                episode_data.get('importance', 0.5),
                json.dumps(episode_data.get('topics', [])),
                json.dumps(episode_data.get('entities', [])),
                json.dumps(episode_data.get('connections', [])),
                episode_data.get('status', 'pending_embedding'),
                episode_data.get('consolidated_at')  # NULL for new episodes
            ))
            conn.commit()
        
        from miyori.utils.memory_logger import memory_logger
        memory_logger.log_event("db_add_episode", {
            "id": episode_id,
            "summary": episode_data.get('summary', '')[:50] + "...",
            "status": episode_data.get('status', 'pending_embedding')
        })
        return episode_id

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodic_memory WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['full_text'] = json.loads(data['full_text'])
                data['topics'] = json.loads(data['topics'])
                data['entities'] = json.loads(data['entities'])
                data['connections'] = json.loads(data['connections'])
                return data
        return None

    def update_episode(self, episode_id: str, updates: Dict[str, Any]) -> bool:
        if not updates:
            return False
            
        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ['full_text', 'topics', 'entities', 'connections']:
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)
        
        values.append(episode_id)
        query = f"UPDATE episodic_memory SET {', '.join(set_parts)} WHERE id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(values))
            conn.commit()
            
            from miyori.utils.memory_logger import memory_logger
            memory_logger.log_event("db_update_episode", {
                "id": episode_id,
                "updates": list(updates.keys()),
                "success": cursor.rowcount > 0
            })
            return cursor.rowcount > 0

    def search_episodes(self, query_embedding: List[float], limit: int = 5, status: str = 'active') -> List[Dict[str, Any]]:
        """
        Vector search using cosine similarity.
        Note: Still does full table scan - see scalability notes below.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodic_memory WHERE status = ?", (status,))
            rows = cursor.fetchall()
        
        if not rows:
            return []
        
        # Build matrix of all embeddings at once
        embeddings_matrix = []
        valid_rows = []
        
        for row in rows:
            if row['embedding'] is not None:
                mem_vec = np.frombuffer(row['embedding'], dtype=np.float32)
                embeddings_matrix.append(mem_vec)
                valid_rows.append(row)
        
        if not embeddings_matrix:
            return []
        
        # Convert to numpy array and compute all similarities at once
        embeddings_matrix = np.array(embeddings_matrix)
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Compute cosine similarity for all vectors at once
        similarities = cosine_similarity(query_vec, embeddings_matrix)[0]
        
        # Build results with similarity scores
        results = []
        for idx, row in enumerate(valid_rows):
            data = dict(row)
            data['similarity'] = float(similarities[idx])
            data['full_text'] = json.loads(data['full_text'])
            data['topics'] = json.loads(data['topics'])
            data['entities'] = json.loads(data['entities'])
            data['connections'] = json.loads(data['connections'])
            results.append(data)
        
        # Sort and return top N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        final_results = results[:limit]
        
        from miyori.utils.memory_logger import memory_logger
        memory_logger.log_event("db_search_episodes", {
            "status": status,
            "total_candidates": len(results),
            "returned_count": len(final_results),
            "top_similarity": final_results[0]['similarity'] if final_results else 0
        })
        
        return final_results

    def get_unconsolidated_episodes(self, status: str = 'active', limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get episodes that haven't been consolidated yet (consolidated_at IS NULL)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM episodic_memory WHERE status = ? AND consolidated_at IS NULL"
            params = [status]

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            data = dict(row)
            data['full_text'] = json.loads(data['full_text'])
            data['topics'] = json.loads(data['topics'])
            data['entities'] = json.loads(data['entities'])
            data['connections'] = json.loads(data['connections'])
            results.append(data)

        from miyori.utils.memory_logger import memory_logger
        memory_logger.log_event("db_get_unconsolidated_episodes", {
            "status": status,
            "limit": limit,
            "returned_count": len(results)
        })
        return results

    def mark_episodes_consolidated(self, episode_ids: List[str]) -> bool:
        """Mark episodes as consolidated by setting consolidated_at timestamp."""
        if not episode_ids:
            return True

        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Use a single UPDATE with IN clause for efficiency
            placeholders = ','.join('?' * len(episode_ids))
            cursor.execute(f"""
                UPDATE episodic_memory
                SET consolidated_at = ?
                WHERE id IN ({placeholders})
            """, [now] + episode_ids)
            conn.commit()

            from miyori.utils.memory_logger import memory_logger
            memory_logger.log_event("db_mark_episodes_consolidated", {
                "episode_count": len(episode_ids),
                "success": cursor.rowcount > 0
            })
            return cursor.rowcount > 0

    def add_semantic_fact(self, fact_data: Dict[str, Any]) -> str:
        fact_id = fact_data.get('id') or str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO semantic_memory (
                    id, fact, confidence, first_observed, last_confirmed,
                    version_history, derived_from, contradictions, status, embedding,
                    evidence_count, merged_into_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                fact_data.get('fact'),
                fact_data.get('confidence', 1.0),
                fact_data.get('first_observed') or now,
                fact_data.get('last_confirmed') or now,
                json.dumps(fact_data.get('version_history', [])),
                json.dumps(fact_data.get('derived_from', [])),
                json.dumps(fact_data.get('contradictions', [])),
                fact_data.get('status', 'stable'),
                fact_data.get('embedding'),  # Already in bytes format
                fact_data.get('evidence_count', 0),
                fact_data.get('merged_into_id')
            ))
            conn.commit()
        return fact_id

    def get_semantic_facts(self, status: str = 'stable', limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM semantic_memory WHERE status = ? LIMIT ?", (status, limit))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            data = dict(row)
            data['version_history'] = json.loads(data['version_history'])
            data['derived_from'] = json.loads(data['derived_from'])
            data['contradictions'] = json.loads(data['contradictions'])
            # Convert embedding bytes back to list of floats
            if data['embedding'] is not None:
                data['embedding'] = np.frombuffer(data['embedding'], dtype=np.float32).tolist()
            results.append(data)
        return results

    def update_semantic_fact(self, fact_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of an existing semantic fact."""
        if not updates:
            return False
        
        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ['version_history', 'derived_from', 'contradictions']:
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == 'embedding' and isinstance(value, (list, np.ndarray)):
                set_parts.append(f"{key} = ?")
                values.append(np.array(value, dtype=np.float32).tobytes())
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)
        
        values.append(fact_id)
        query = f"UPDATE semantic_memory SET {', '.join(set_parts)} WHERE id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(values))
            conn.commit()
            
            from miyori.utils.memory_logger import memory_logger
            memory_logger.log_event("db_update_semantic_fact", {
                "id": fact_id,
                "updates": list(updates.keys()),
                "success": cursor.rowcount > 0
            })
            return cursor.rowcount > 0

    def get_all_active_facts(self, min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all active/stable facts for batch operations."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if min_confidence is not None:
                cursor.execute(
                    "SELECT * FROM semantic_memory WHERE status IN ('stable', 'tentative') AND confidence >= ?",
                    (min_confidence,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM semantic_memory WHERE status IN ('stable', 'tentative')"
                )
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            data = dict(row)
            data['version_history'] = json.loads(data['version_history'])
            data['derived_from'] = json.loads(data['derived_from'])
            data['contradictions'] = json.loads(data['contradictions'])
            # Keep embedding as bytes for vector operations
            results.append(data)
        
        from miyori.utils.memory_logger import memory_logger
        memory_logger.log_event("db_get_all_active_facts", {
            "min_confidence": min_confidence,
            "returned_count": len(results)
        })
        return results

    def archive_merged_facts(self, loser_ids: List[str], winner_id: str) -> bool:
        """Archive facts that have been merged into a canonical fact."""
        if not loser_ids:
            return True
        
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(loser_ids))
            cursor.execute(f"""
                UPDATE semantic_memory
                SET status = 'merged_into',
                    merged_into_id = ?,
                    last_confirmed = ?
                WHERE id IN ({placeholders})
            """, [winner_id, now] + loser_ids)
            conn.commit()
            
            from miyori.utils.memory_logger import memory_logger
            memory_logger.log_event("db_archive_merged_facts", {
                "winner_id": winner_id,
                "loser_count": len(loser_ids),
                "success": cursor.rowcount > 0
            })
            return cursor.rowcount > 0
