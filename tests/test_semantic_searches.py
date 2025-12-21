import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import Config
Config.load()
from src.memory.sqlite_store import SQLiteMemoryStore
from src.memory.episodic import EpisodicMemoryManager
from src.utils.embeddings import EmbeddingService

def test_memory_search():
    db_path = "memory.db"

    if not os.path.exists(db_path):
        print("ERROR: Database file 'memory.db' not found.")
        return False

    store = SQLiteMemoryStore(db_path)
    embedding_service = EmbeddingService()
    manager = EpisodicMemoryManager(store, embedding_service)

    with store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

    if active_count == 0:
        print("ERROR: No active episodes found in database.")
        return False

    print(f"Testing with {active_count} active episodes from memory.db")
    print()

    search_episodes_query = "apple, orange, mango"
    retrieve_relevant_episode_query = "apple, orange, mango"

    print("TESTING: search_episodes")
    print(f"QUERY: {search_episodes_query}")

    query_embedding = embedding_service.embed(search_episodes_query)
    results = store.search_episodes(query_embedding, limit=3)

    for i, result in enumerate(results, 1):
        print(f"RESULT {i}: {result['summary'][:100]}...")
    print()

    print("TESTING: retrieve_relevant")
    print(f"QUERY: {retrieve_relevant_episode_query}")

    results = manager.retrieve_relevant(retrieve_relevant_episode_query, limit=3)

    for i, result in enumerate(results, 1):
        age_days = (datetime.now() - datetime.fromisoformat(result['timestamp'])).days
        print(f"RESULT {i} (age: {age_days}d, imp: {result['importance']:.2f}): {result['summary'][:100]}...")

    return True

if __name__ == "__main__":
    success = test_memory_search()
    sys.exit(0 if success else 1)
