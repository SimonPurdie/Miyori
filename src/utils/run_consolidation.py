#!/usr/bin/env python3
"""
Memory consolidation runner - can be used both as an importable function
and as a standalone command-line script.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional

def run_consolidation(db_path: Optional[str] = None, verbose: bool = True) -> bool:
    """
    Run memory consolidation on the specified database.

    Args:
        db_path: Path to memory.db file. If None, uses project_root/memory.db
        verbose: Whether to print progress messages

    Returns:
        bool: True if consolidation completed successfully, False otherwise
    """
    try:
        # Determine project root and add src to path
        if hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            project_root = Path(sys._MEIPASS).parent
        else:
            # Running as script or import
            current_file = Path(__file__).resolve()
            if current_file.name == 'run_consolidation.py':
                # Running as standalone script from src/utils/
                project_root = current_file.parent.parent.parent
            else:
                # Imported from elsewhere
                project_root = current_file.parent.parent.parent

        # Add src to path for imports
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Import required modules
        from google import genai
        from src.memory.consolidation import ConsolidationManager
        from src.memory.deep_layers import SemanticExtractor
        from src.memory.episodic import EpisodicMemoryManager
        from src.memory.sqlite_store import SQLiteMemoryStore
        from src.utils.embeddings import EmbeddingService
        from src.utils.config import Config

        if verbose:
            print("=== Memory Consolidation ===")

        # Load configuration
        Config.load()
        if verbose:
            print("OK: Configuration loaded")

        # Setup Gemini client
        llm_config = Config.data.get("llm", {})
        api_key = llm_config.get("api_key")
        if not api_key:
            if verbose:
                print("ERROR: API key not found in config")
            return False

        client = genai.Client(api_key=api_key)
        if verbose:
            print("OK: Gemini client initialized")

        # Setup memory store
        if db_path is None:
            db_path = project_root / "memory.db"
        else:
            db_path = Path(db_path)

        store = SQLiteMemoryStore(str(db_path))
        if verbose:
            print(f"OK: Memory store initialized (DB: {db_path})")

        # Setup embedding service
        embedding_service = EmbeddingService()
        if verbose:
            print("OK: Embedding service initialized")

        # Setup episodic manager
        episodic_manager = EpisodicMemoryManager(store, embedding_service)
        if verbose:
            print("OK: Episodic memory manager initialized")

        # Setup semantic extractor
        semantic_extractor = SemanticExtractor(client, store)
        if verbose:
            print("OK: Semantic extractor initialized")

        # Setup consolidation manager
        consolidation_manager = ConsolidationManager(
            store=store,
            episodic_manager=episodic_manager,
            semantic_extractor=semantic_extractor
        )
        if verbose:
            print("OK: Consolidation manager initialized")

        # Check current memory state
        if verbose:
            print("\n--- Current Memory State ---")

        # Count episodes
        try:
            with store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE status = 'active'")
                active_episodes = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE status = 'pending_embedding'")
                pending_episodes = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM semantic_memory")
                semantic_facts = cursor.fetchone()[0]

            if verbose:
                print(f"Active episodes: {active_episodes}")
                print(f"Pending episodes: {pending_episodes}")
                print(f"Semantic facts: {semantic_facts}")

        except Exception as e:
            if verbose:
                print(f"Error checking memory state: {e}")
            return False

        # Run consolidation
        if verbose:
            print("\n--- Running Consolidation ---")

        # Create event loop if one doesn't exist
        try:
            loop = asyncio.get_running_loop()
            # We're already in an event loop, create a task
            loop.create_task(consolidation_manager.perform_consolidation())
        except RuntimeError:
            # No event loop, run synchronously
            asyncio.run(consolidation_manager.perform_consolidation())

        # Check memory state after consolidation
        if verbose:
            print("\n--- Memory State After Consolidation ---")
        try:
            with store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE status = 'active'")
                active_episodes = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM semantic_memory")
                semantic_facts = cursor.fetchone()[0]

            if verbose:
                print(f"Active episodes: {active_episodes}")
                print(f"Semantic facts: {semantic_facts}")

                # Show some recent semantic facts
                if semantic_facts > 0:
                    cursor.execute("SELECT fact, confidence FROM semantic_memory ORDER BY last_confirmed DESC LIMIT 5")
                    recent_facts = cursor.fetchall()
                    print("\nRecent semantic facts:")
                    for fact, confidence in recent_facts:
                        print(f"  • {fact} (confidence: {confidence})")

        except Exception as e:
            if verbose:
                print(f"Error checking final memory state: {e}")
            return False

        if verbose:
            print("\n=== Consolidation Complete ===")

        return True

    except Exception as e:
        if verbose:
            print(f"ERROR: Consolidation failed: {e}")
        return False


def main():
    """Command-line interface for running consolidation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run memory consolidation on Miyori's memory database")
    parser.add_argument("--db", help="Path to memory.db file (default: project_root/memory.db)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output messages")

    args = parser.parse_args()

    verbose = not args.quiet
    success = run_consolidation(db_path=args.db, verbose=verbose)

    if not verbose:
        # If quiet mode, still show success/failure
        print("SUCCESS" if success else "FAILED")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
