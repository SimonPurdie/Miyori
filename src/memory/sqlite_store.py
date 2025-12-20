import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.interfaces.memory import IMemoryStore

class SQLiteMemoryStore(IMemoryStore):
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
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
                    emotional_valence REAL,
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
                    status TEXT
                )
            """)

            # Relational Memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relational_memory (
                    category TEXT PRIMARY KEY,
                    data TEXT, -- JSON string
                    confidence REAL,
                    evidence_count INTEGER,
                    last_updated DATETIME
                )
            """)

            # Emotional Thread
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_thread (
                    id TEXT PRIMARY KEY,
                    current_state TEXT,
                    recognized_from TEXT, -- JSON string
                    should_acknowledge INTEGER, -- 0 or 1
                    thread_length INTEGER,
                    last_update DATETIME
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
            
            conn.commit()

    def add_episode(self, episode_data: Dict[str, Any]) -> str:
        episode_id = episode_data.get('id') or str(uuid.uuid4())
        timestamp = episode_data.get('timestamp') or datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memory (
                    id, summary, full_text, timestamp, embedding,
                    importance, emotional_valence, topics, entities,
                    connections, status, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                episode_data.get('summary'),
                json.dumps(episode_data.get('full_text', {})),
                timestamp,
                episode_data.get('embedding'), # Should be bytes/BLOB
                episode_data.get('importance', 0.5),
                episode_data.get('emotional_valence', 0.0),
                json.dumps(episode_data.get('topics', [])),
                json.dumps(episode_data.get('entities', [])),
                json.dumps(episode_data.get('connections', [])),
                episode_data.get('status', 'pending_embedding'),
                episode_data.get('consolidated_at')  # NULL for new episodes
            ))
            conn.commit()
        
        from src.utils.memory_logger import memory_logger
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
            
            from src.utils.memory_logger import memory_logger
            memory_logger.log_event("db_update_episode", {
                "id": episode_id,
                "updates": list(updates.keys()),
                "success": cursor.rowcount > 0
            })
            return cursor.rowcount > 0

    def search_episodes(self, query_embedding: List[float], limit: int = 5, status: str = 'active') -> List[Dict[str, Any]]:
        """
        Note: Basic SQLite doesn't support vector search natively.
        In Phase 1, we fetch active episodes and compute cosine similarity in Python.
        Scalability improvements will come in later phases.
        """
        import numpy as np
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodic_memory WHERE status = ?", (status,))
            rows = cursor.fetchall()
            
        results = []
        query_vec = np.array(query_embedding)
        
        for row in rows:
            if row['embedding'] is None:
                continue
                
            # Assuming embedding is stored as pickled or bytes of numpy array
            # For simplicity in Phase 1, we'll assume it's a blob of floats
            mem_vec = np.frombuffer(row['embedding'], dtype=np.float32)
            
            # Simple cosine similarity
            if np.linalg.norm(mem_vec) == 0 or np.linalg.norm(query_vec) == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec))
            
            data = dict(row)
            data['similarity'] = float(similarity)
            data['full_text'] = json.loads(data['full_text'])
            data['topics'] = json.loads(data['topics'])
            data['entities'] = json.loads(data['entities'])
            data['connections'] = json.loads(data['connections'])
            results.append(data)
            
        # Sort by similarity and return top N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        final_results = results[:limit]

        from src.utils.memory_logger import memory_logger
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

        from src.utils.memory_logger import memory_logger
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

            from src.utils.memory_logger import memory_logger
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
                    version_history, derived_from, contradictions, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                fact_data.get('fact'),
                fact_data.get('confidence', 1.0),
                fact_data.get('first_observed') or now,
                fact_data.get('last_confirmed') or now,
                json.dumps(fact_data.get('version_history', [])),
                json.dumps(fact_data.get('derived_from', [])),
                json.dumps(fact_data.get('contradictions', [])),
                fact_data.get('status', 'stable')
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
            results.append(data)
        return results

    def update_relational_memory(self, category: str, data: Dict[str, Any], confidence: float) -> bool:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO relational_memory (
                    category, data, confidence, evidence_count, last_updated
                ) VALUES (
                    ?, ?, ?, 
                    COALESCE((SELECT evidence_count FROM relational_memory WHERE category = ?) + 1, 1),
                    ?
                )
            """, (category, json.dumps(data), confidence, category, now))
            conn.commit()
            return True

    def get_relational_memories(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT * FROM relational_memory WHERE category = ?", (category,))
            else:
                cursor.execute("SELECT * FROM relational_memory")
            rows = cursor.fetchall()
            
        results = []
        for row in rows:
            data = dict(row)
            data['data'] = json.loads(data['data'])
            results.append(data)
        return results

    def update_emotional_thread(self, thread_data: Dict[str, Any]) -> bool:
        thread_id = thread_data.get('id') or 'current'
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO emotional_thread (
                    id, current_state, recognized_from, should_acknowledge, 
                    thread_length, last_update
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                thread_id,
                thread_data.get('current_state'),
                json.dumps(thread_data.get('recognized_from', [])),
                1 if thread_data.get('should_acknowledge') else 0,
                thread_data.get('thread_length', 1),
                now
            ))
            conn.commit()
            return True

    def get_emotional_thread(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM emotional_thread ORDER BY last_update DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['recognized_from'] = json.loads(data['recognized_from'])
                data['should_acknowledge'] = bool(data['should_acknowledge'])
                return data
        return None
