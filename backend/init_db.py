import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tickets.db"

conn = sqlite3.connect(DB_PATH)
conn.execute('''
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    content TEXT,
    priority TEXT
);
''')
conn.commit()
conn.close()

print("✅ Database and table 'tickets' created successfully!")
