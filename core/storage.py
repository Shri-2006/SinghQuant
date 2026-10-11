"""
Database storage checks (Phase 2 I-1).

In Docker the database lives at /app/database/trades.db on a DIRECTORY volume
(./database on the host) so that SQLite's write-ahead log (trades.db-wal) and
shared-memory index (trades.db-shm), which are always created next to the
database file, are persistent and shared by every container. The old
single-file mount (./trades.db:/app/trades.db) left those two files inside
each container's private filesystem.

storage_problems() turns the usual misconfigurations into explicit errors
instead of silent data loss.
"""
import os
import sqlite3
from pathlib import Path


def storage_problems(db_path, need_write=True):
    """Returns a list of human-readable problems with the database location (empty = OK)."""
    problems = []
    p = Path(db_path)
    parent = p.parent if str(p.parent) else Path(".")
    if p.is_dir():
        problems.append(f"{p} is a directory, not a database file. Docker creates a directory when a "
                        "bind-mounted host file does not exist; mount the data DIRECTORY instead (./database:/app/database).")
        return problems
    if not parent.is_dir():
        problems.append(f"database directory {parent} does not exist (is the ./database volume mounted?)")
        return problems
    if need_write and not os.access(parent, os.W_OK):
        problems.append(f"database directory {parent} is not writable; SQLite must create "
                        f"{p.name}-wal and {p.name}-shm next to the database")
    if p.exists():
        try:
            conn = sqlite3.connect(Path(p).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            try:
                result = conn.execute("PRAGMA quick_check").fetchone()[0]
            finally:
                conn.close()
            if result != "ok":
                problems.append(f"{p} failed PRAGMA quick_check: {result}")
        except sqlite3.DatabaseError as e:
            problems.append(f"{p} cannot be read as a SQLite database: {e}")
    return problems
