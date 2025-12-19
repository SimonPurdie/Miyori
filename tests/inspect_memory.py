import sqlite3
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

def format_timestamp(ts):
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts

def inspect_episodic(db_path, limit=5, status=None, search=None):
    print(f"\n=== EPISODIC MEMORIES (Limit: {limit}) ===")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT id, summary, timestamp, importance, status FROM episodic_memory"
        params = []
        
        where_clauses = []
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if search:
            where_clauses.append("(summary LIKE ? OR full_text LIKE ?)")
            params.append(f"%{search}%")
            params.append(f"%{search}%")
            
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        if not rows:
            print("No memories found.")
            return

        for row in rows:
            print(f"[{format_timestamp(row['timestamp'])}] ID: {row['id'][:8]}... | Status: {row['status']:<18} | Imp: {row['importance']:.2f}")
            print(f"  Summary: {row['summary']}")
            print("-" * 40)

def inspect_semantic(db_path, limit=10):
    print(f"\n=== SEMANTIC FACTS (Limit: {limit}) ===")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT fact, confidence, last_confirmed FROM semantic_memory ORDER BY last_confirmed DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        if not rows:
            print("No facts found.")
            return

        for row in rows:
            print(f"[{format_timestamp(row['last_confirmed'])}] Conf: {row['confidence']:.2f} | {row['fact']}")

def inspect_relational(db_path):
    print("\n=== RELATIONAL MEMORY (Personality/Style) ===")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT category, data, confidence FROM relational_memory")
        rows = cursor.fetchall()
        
        if not rows:
            print("No relational data found.")
            return

        for row in rows:
            data = json.loads(row['data'])
            print(f"Category: {row['category']:<15} | Conf: {row['confidence']:.2f}")
            print(f"  Data: {json.dumps(data, indent=4)}")

def inspect_emotional(db_path):
    print("\n=== EMOTIONAL THREAD ===")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT current_state, last_update, should_acknowledge FROM emotional_thread ORDER BY last_update DESC LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            print("No emotional state found.")
            return

        print(f"Current State: {row['current_state']}")
        print(f"Last Update:   {format_timestamp(row['last_update'])}")
        print(f"Acknowledge:  {'Yes' if row['should_acknowledge'] else 'No'}")

def main():
    parser = argparse.ArgumentParser(description="Miyori Memory Inspector")
    parser.add_argument("--db", default="memory.db", help="Path to memory.db")
    parser.add_argument("--recent", type=int, default=5, help="Number of recent episodes to show")
    parser.add_argument("--status", help="Filter episodes by status (active, pending_embedding, etc.)")
    parser.add_argument("--search", help="Search episodes by keyword")
    parser.add_argument("--semantic", action="store_true", help="Show semantic facts")
    parser.add_argument("--relational", action="store_true", help="Show relational memory")
    parser.add_argument("--emotional", action="store_true", help="Show current emotional state")
    parser.add_argument("--all", action="store_true", help="Show everything")
    
    args = parser.parse_args()
    db_path = Path(args.db)
    
    if not db_path.exists():
        print(f"Error: Database file '{db_path}' not found.")
        sys.exit(1)
        
    if args.all or (not args.semantic and not args.relational and not args.emotional):
        inspect_episodic(db_path, limit=args.recent, status=args.status, search=args.search)
        
    if args.all or args.semantic:
        inspect_semantic(db_path)
        
    if args.all or args.relational:
        inspect_relational(db_path)
        
    if args.all or args.emotional:
        inspect_emotional(db_path)

if __name__ == "__main__":
    main()
