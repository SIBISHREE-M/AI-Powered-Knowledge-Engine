

import sqlite3
from pathlib import Path
from datasets import load_dataset


def init_db(db_path):
    """Open a connection to the SQLite DB."""
    return sqlite3.connect(db_path)


def fetch_dataset():
    """Load the HF dataset."""
    print("[INFO] Downloading dataset...")
    data = load_dataset("Tobi-Bueck/customer-support-tickets")
    return data["train"]


def insert_rows(conn, records):
    """Insert dataset rows into the SQLite table."""
    cursor = conn.cursor()

    for row in records:
        text = row.get("body", "")
        level = (row.get("priority") or "Low").capitalize()

        cursor.execute(
            """
            INSERT INTO tickets (filename, content, priority)
            VALUES (?, ?, ?)
            """,
            ("dataset_ticket", text, level)
        )

    conn.commit()


def main():
    db_file = Path(__file__).parent / "tickets.db"

    # Open DB
    conn = init_db(db_file)

    # Load dataset
    rows = fetch_dataset()

    # Insert records
    insert_rows(conn, rows)

    # Cleanup
    conn.close()
    print("[SUCCESS] Imported dataset into tickets.db")

