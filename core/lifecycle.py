"""
Process lifecycle state that must survive restarts (Phase 2 I-4).

process_runs records every start and how it ended. A run that never
recorded an end (SIGKILL, OOM, power loss, a crash) is closed as
"unclean" by the next start. Unclean exits are counted over
CRASH_LOOP_WINDOW_SECONDS:

  * SAFE_MODE_AFTER_UNCLEAN_EXITS or more -> persistent SAFE MODE: every
    strategy keeps managing existing positions (stops, momentum exits,
    kill switch) but opens NO new entries. It is stored in the database,
    so further restarts cannot reset it; only a human clears it
    (tools/safe_mode.py clear).
  * MAX_WATCHDOG_RESTARTS or more -> the watchdog stops exiting the
    process when a strategy thread dies, so a persistent fault cannot turn
    into an endless restart loop; it alerts and keeps the rest running.

Kill-switch halts, peaks, streaks, anti-churn state and ledger flags already
live in the database, so a restart never resets them either.
"""
import json
import os
import socket
from datetime import datetime, timedelta, timezone

from core.config import CRASH_LOOP_WINDOW_SECONDS
from core.logger import db_connection, utc_now_iso

CLEAN = "clean"
WATCHDOG_EXIT = "watchdog_exit"
FATAL_STARTUP = "fatal_startup"
UNCLEAN_UNRECORDED = "unclean (no shutdown recorded)"
SAFE_MODE_KEY = "safe_mode"


def _window_start(window_seconds):
    return (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=window_seconds)).isoformat()


def record_start(db_path=None, window_seconds=CRASH_LOOP_WINDOW_SECONDS):
    """
    Closes any run left open by a previous process as unclean, records this
    start, and returns (run_id, unclean_exits_in_window). Atomic.
    """
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE process_runs SET ended_at = ?, end_reason = ? WHERE ended_at IS NULL",
                     (now, UNCLEAN_UNRECORDED))
        unclean = conn.execute("SELECT COUNT(*) FROM process_runs WHERE end_reason != ? AND started_at >= ?",
                               (CLEAN, _window_start(window_seconds))).fetchone()[0]
        cur = conn.execute("INSERT INTO process_runs (started_at, pid, hostname) VALUES (?,?,?)",
                           (now, os.getpid(), socket.gethostname()))
        return cur.lastrowid, int(unclean)


def record_end(run_id, reason, db_path=None):
    with db_connection(db_path) as conn:
        conn.execute("UPDATE process_runs SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                     (utc_now_iso(), reason, run_id))


def unclean_exits(db_path=None, window_seconds=CRASH_LOOP_WINDOW_SECONDS, readonly=None):
    """Unclean exits recorded in the window (the current, still-open run is not counted)."""
    with db_connection(db_path, readonly=readonly) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM process_runs WHERE ended_at IS NOT NULL AND end_reason != ? "
                                "AND started_at >= ?", (CLEAN, _window_start(window_seconds))).fetchone()[0])


def recent_runs(limit=10, db_path=None, readonly=None):
    with db_connection(db_path, readonly=readonly) as conn:
        rows = conn.execute("SELECT id, started_at, ended_at, end_reason, pid, hostname FROM process_runs "
                            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(("id", "started_at", "ended_at", "end_reason", "pid", "hostname"), r)) for r in rows]


def get_safe_mode(db_path=None, readonly=None):
    """{"since", "reason"} while SAFE MODE is active, else None."""
    with db_connection(db_path, readonly=readonly) as conn:
        row = conn.execute("SELECT value FROM system_state WHERE key = ?", (SAFE_MODE_KEY,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except ValueError:
        return {"since": None, "reason": str(row[0])}


def set_safe_mode(reason, db_path=None):
    """Enters SAFE MODE (idempotent: the original reason and time are kept). Returns True if newly entered."""
    value = json.dumps({"since": utc_now_iso(), "reason": reason})
    with db_connection(db_path) as conn:
        cur = conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES (?,?,?) "
                           "ON CONFLICT(key) DO NOTHING", (SAFE_MODE_KEY, value, utc_now_iso()))
        return cur.rowcount > 0


def clear_safe_mode(note, db_path=None):
    """Human-only: leaves SAFE MODE. Returns the state that was cleared (or None)."""
    previous = get_safe_mode(db_path)
    with db_connection(db_path) as conn:
        conn.execute("DELETE FROM system_state WHERE key = ?", (SAFE_MODE_KEY,))
        conn.execute("INSERT INTO system_state (key, value, updated_at) VALUES (?,?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                     ("safe_mode_last_cleared", json.dumps({"previous": previous, "note": note}), utc_now_iso()))
    return previous
