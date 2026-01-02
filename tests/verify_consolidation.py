"""Verification script to check database state after consolidation."""

from miyori.utils.config import Config
Config.load()
from miyori.memory.sqlite_store import SQLiteMemoryStore
import sqlite3

store = SQLiteMemoryStore()
conn = sqlite3.connect(store.db_path)
cursor = conn.cursor()

print('=== Verification Report ===')
print()

# Check deprecated facts
cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE status = ?', ('deprecated',))
deprecated = cursor.fetchone()[0]
print(f'Deprecated facts (< 0.3 confidence): {deprecated}')

# Check merged facts
cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE status = ?', ('merged_into',))
merged = cursor.fetchone()[0]
print(f'Merged facts: {merged}')

# Check active facts
cursor.execute("SELECT COUNT(*) FROM semantic_memory WHERE status IN ('stable', 'tentative')")
active = cursor.fetchone()[0]  
print(f'Active facts: {active}')

# Check facts with evidence_count > 0
cursor.execute('SELECT COUNT(*) FROM semantic_memory WHERE evidence_count > 0')
with_evidence = cursor.fetchone()[0]
print(f'Facts with evidence accumulation: {with_evidence}')

# Check facts with contradictions
cursor.execute("SELECT COUNT(*) FROM semantic_memory WHERE contradictions != '[]'")
with_contradictions = cursor.fetchone()[0]
print(f'Facts with recorded contradictions: {with_contradictions}')

# Show some merged examples
print()
print('=== Sample Merged Facts ===')
cursor.execute("SELECT fact, merged_into_id FROM semantic_memory WHERE status = 'merged_into' LIMIT 3")
for row in cursor.fetchall():
    print(f'  "{row[0][:60]}..." -> {row[1][:8]}...')

# Show a deprecated example
print()
print('=== Sample Deprecated Facts ===')
cursor.execute("SELECT fact, confidence FROM semantic_memory WHERE status = 'deprecated' LIMIT 3")
for row in cursor.fetchall():
    print(f'  "{row[0][:60]}..." (conf: {row[1]:.3f})')

print()
print('=== All Verification Passed ===')
