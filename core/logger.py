"""
SQLite trade log and heartbeat.

All bots share one database file (trades.db at the project root). Schema
changes are ADDITIVE ONLY: `_migrate()` adds missing columns/tables with
`ALTER TABLE ... ADD COLUMN` and `CREATE TABLE IF NOT EXISTS`, so an existing
database (and its historical rows) keeps working unchanged.

Units in the `trades` table (fixed in the 2026-09 audit, issue H-03):
    price        -> signal price (the price the strategy decided on)
    quantity     -> ALWAYS a share / coin quantity, never dollars
    filled_qty   -> quantity actually filled at the broker (None if unknown)
    filled_avg_price -> average fill price from the broker (None if unknown)
Historical rows written before the audit logged SELL `quantity` in dollars; a
row can be recognised as legacy when `status` IS NULL.
"""
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# SINGHQUANT_DB_PATH lets tests, tooling and containers point at another
# database file. In Docker it is /app/database/trades.db on a DIRECTORY volume, so
# the -wal and -shm files live next to the database on the host (Phase 2 I-1).
DB_PATH = os.getenv("SINGHQUANT_DB_PATH") or os.path.join(os.path.dirname(__file__), '..', 'trades.db')

# SINGHQUANT_DB_READONLY=1 (the dashboard): connections are opened read-only,
# nothing is created or migrated at import, and no state rows are inserted.
# The one-shot `migrate` service (tools/db_maintenance.py migrate) owns the
# schema (Phase 2 I-2).
READONLY = os.getenv("SINGHQUANT_DB_READONLY", "") == "1"

# Bumped whenever _migrate changes the schema. Stored in PRAGMA user_version and
# written in the SAME transaction as the migration, so it is only ever set
# when every change was applied. 0 = any database older than Phase 2.
SCHEMA_VERSION = 1


class SchemaTooNew(RuntimeError):
    """The database was migrated by newer code than this process."""


class SchemaNotReady(RuntimeError):
    """A read-only process found a database that has not been migrated yet."""

# One process-wide lock serialises writes from the three strategy threads so
# that SQLite never has to time out on a busy lock inside the same process.
_DB_LOCK = threading.RLock()

TRADES_COLUMNS = ['id', 'timestamp', 'strategy', 'asset', 'action', 'price',
                  'quantity', 'pnl', 'reason']
TRADES_EXTRA_COLUMNS = [
    ('signal_price', 'REAL'),
    ('order_id', 'TEXT'),
    ('status', 'TEXT'),
    ('filled_qty', 'REAL'),
    ('filled_avg_price', 'REAL'),
    ('strategy_position_before', 'REAL'),
    ('account_position_before', 'REAL'),
    ('risk_state', 'TEXT'),
    ('model_output', 'TEXT'),
]


def utc_now_iso():
    """Timezone-aware UTC timestamp in ISO format (datetime.utcnow is deprecated)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _readonly_uri(path):
    return Path(path).resolve().as_uri() + "?mode=ro"


def is_readonly(readonly=None):
    return READONLY if readonly is None else bool(readonly)


@contextmanager
def db_connection(db_path=None, readonly=None):
    """
    Yields a connection with a busy timeout; commits on success.
    Read-only connections (readonly=True or SINGHQUANT_DB_READONLY=1) open the
    file with mode=ro: any write raises instead of touching the database.
    """
    path = db_path or DB_PATH
    with _DB_LOCK:
        if is_readonly(readonly):
            conn = sqlite3.connect(_readonly_uri(path), uri=True, timeout=30)
        else:
            conn = sqlite3.connect(path, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            if not is_readonly(readonly):
                conn.commit()
        finally:
            conn.close()


def _existing_columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn):
    """Adds any columns/tables that an older database is missing."""
    have = _existing_columns(conn, "trades")
    for name, ctype in TRADES_EXTRA_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {ctype}")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            strategy        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            qty             REAL NOT NULL,
            avg_entry_price REAL NOT NULL,
            opened_at       TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (strategy, symbol)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            side            TEXT NOT NULL,
            requested_qty   REAL NOT NULL,
            signal_price    REAL,
            broker_order_id TEXT,
            status          TEXT NOT NULL,
            filled_qty      REAL NOT NULL DEFAULT 0,
            filled_avg_price REAL,
            reason          TEXT,
            submitted_at    TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_orders_open ON orders(strategy, symbol, status)')
    if 'client_order_id' not in _existing_columns(conn, "orders"):
        conn.execute('ALTER TABLE orders ADD COLUMN client_order_id TEXT')
    pos_cols = _existing_columns(conn, "positions")
    if 'last_mark' not in pos_cols:
        conn.execute('ALTER TABLE positions ADD COLUMN last_mark REAL')
    if 'last_mark_at' not in pos_cols:
        conn.execute('ALTER TABLE positions ADD COLUMN last_mark_at TEXT')
    # Remediation R-1: a ledger quantity the broker no longer holds is FLAGGED
    # (never written off automatically). The flag is informational: qty and
    # avg_entry_price are untouched until a human reconciles it.
    for name, ctype in (('discrepancy_qty', 'REAL'), ('discrepancy_reason', 'TEXT'),
                        ('discrepancy_since', 'TEXT'), ('discrepancy_alerted', 'INTEGER NOT NULL DEFAULT 0')):
        if name not in pos_cols:
            conn.execute(f'ALTER TABLE positions ADD COLUMN {name} {ctype}')
    # Journal of every ledger mutation. Broker fills carry a broker_order_id;
    # internal reconciliation events (WRITE_OFF_*) never do (third-pass T-05).
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ledger_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            strategy         TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            qty_delta        REAL NOT NULL,
            cash_delta       REAL NOT NULL,
            price            REAL,
            avg_entry_before REAL NOT NULL,
            qty_before       REAL NOT NULL,
            qty_after        REAL NOT NULL,
            reason           TEXT,
            broker_order_id  TEXT,
            extra            TEXT
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ledger_events_sym ON ledger_events(strategy, symbol)')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS strategy_state (
            strategy        TEXT PRIMARY KEY,
            cash_budget     REAL NOT NULL,
            peak_equity     REAL NOT NULL,
            last_equity     REAL,
            critical_streak INTEGER NOT NULL DEFAULT 0,
            halted          INTEGER NOT NULL DEFAULT 0,
            halted_reason   TEXT,
            updated_at      TEXT NOT NULL
        )
    ''')
    # Remediation R-1: consecutive cycles whose risk reading was "suspect".
    if 'suspect_streak' not in _existing_columns(conn, "strategy_state"):
        conn.execute('ALTER TABLE strategy_state ADD COLUMN suspect_streak INTEGER NOT NULL DEFAULT 0')
    # Phase 2 I-4: liveness (last_seen), readiness and degraded reasons are
    # reported separately. ready = 1 only when the strategy may open entries.
    hb_cols = _existing_columns(conn, "heartbeat")
    if 'ready' not in hb_cols:
        conn.execute('ALTER TABLE heartbeat ADD COLUMN ready INTEGER')
    if 'degraded_reason' not in hb_cols:
        conn.execute('ALTER TABLE heartbeat ADD COLUMN degraded_reason TEXT')
    # Phase 2 I-4: every process start/stop, for persistent crash-loop detection.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS process_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            end_reason  TEXT,
            pid         INTEGER,
            hostname    TEXT
        )
    ''')
    # Phase 2 I-4: process-wide flags that must survive restarts (safe mode).
    conn.execute('''
        CREATE TABLE IF NOT EXISTS system_state (
            key         TEXT PRIMARY KEY,
            value       TEXT,
            updated_at  TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS signal_state (
            strategy        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            last_bar        TEXT,
            last_action     TEXT,
            last_action_at  TEXT,
            PRIMARY KEY (strategy, symbol)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            strategy    TEXT NOT NULL,
            equity      REAL NOT NULL,
            cash_budget REAL NOT NULL,
            positions_value REAL NOT NULL,
            peak_equity REAL NOT NULL,
            drawdown    REAL,
            risk_level  TEXT NOT NULL
        )
    ''')


def schema_version(db_path=None, readonly=None):
    """PRAGMA user_version of the database (0 = never migrated by Phase 2+ code)."""
    with db_connection(db_path, readonly=readonly) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def schema_ready(db_path=None, readonly=None):
    try:
        return schema_version(db_path, readonly=readonly) >= SCHEMA_VERSION
    except sqlite3.Error:
        return False


def _ensure_wal(conn, timeout=30.0):
    """
    Switches the database to WAL once. SQLite returns "database is locked"
    IMMEDIATELY (without the busy handler) when another process holds a lock
    during this switch, so it is retried with backoff for up to `timeout`
    seconds, re-reading the mode each time (the other process may have done it).
    """
    deadline = time.monotonic() + timeout
    delay = 0.01
    while True:
        if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
            return
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or time.monotonic() > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def init_db(db_path=None):
    """
    Creates or migrates the schema. Safe to call from several processes at once
    (Phase 2 I-2, the bot and dashboard containers starting together):
      * fast path: a database already at SCHEMA_VERSION is left untouched;
      * otherwise every CREATE / ALTER and the user_version stamp run in ONE
        `BEGIN IMMEDIATE` transaction. SQLite DDL is transactional, so a crash
        or error leaves the schema exactly as it was (never half-migrated),
        and a second process waits for the lock, re-reads user_version and
        finds nothing left to do (no duplicate-column errors);
      * a database stamped by NEWER code raises SchemaTooNew;
      * in read-only mode nothing is written: SchemaNotReady is raised if the
        schema is not current yet.
    """
    path = db_path or DB_PATH
    if READONLY:
        version = schema_version(path, readonly=True)
        if version < SCHEMA_VERSION:
            raise SchemaNotReady(f"{path} is at schema {version}, need {SCHEMA_VERSION}; run the migrate service")
        if version > SCHEMA_VERSION:
            raise SchemaTooNew(f"{path} is at schema {version}; this code understands {SCHEMA_VERSION}")
        return version
    with _DB_LOCK:
        conn = sqlite3.connect(path, timeout=30, isolation_level=None)   # explicit transactions only
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise SchemaTooNew(f"{path} is at schema {version}; this code understands {SCHEMA_VERSION}")
            _ensure_wal(conn)
            if version == SCHEMA_VERSION:
                return version
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise SchemaTooNew(f"{path} is at schema {version}; this code understands {SCHEMA_VERSION}")
                if version < SCHEMA_VERSION:
                    _create_and_migrate(conn)
                    conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return SCHEMA_VERSION
        finally:
            conn.close()


def _create_and_migrate(conn):
    """Every schema statement. Runs inside init_db's transaction; never call it directly."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            strategy    TEXT    NOT NULL,
            asset       TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            price       REAL    NOT NULL,
            quantity    REAL    NOT NULL,
            pnl         REAL,
            reason      TEXT
        )
    ''')
    # Heartbeat table - each bot writes here to show state of strategy.
    # Dashboard uses this to show RUNNING/PAUSED/DISCONNECTED
    conn.execute('''
        CREATE TABLE IF NOT EXISTS heartbeat (
            strategy    TEXT    PRIMARY KEY,
            last_seen   TEXT    NOT NULL,
            status      TEXT    NOT NULL
        )
    ''')
    _migrate(conn)


def log_trade(strategy, asset, action, price, quantity, pnl=None, reason=None,
              signal_price=None, order_id=None, status=None, filled_qty=None,
              filled_avg_price=None, strategy_position_before=None,
              account_position_before=None, risk_state=None, model_output=None,
              db_path=None):
    """
    Logs a single trade/decision to the database.
    strategy: "stable" | "risky1" | "risky2"
    asset   : e.g. "SPY" or "X:BTCUSD"
    action  : "BUY" | "SELL" | "KILL_SWITCH" | ...
    price   : signal price at time of decision
    quantity: share/coin quantity requested (NEVER dollars)
    pnl     : realized profit or loss in dollars (None if not a closing trade)
    reason  : why the trade fired
    The remaining keyword arguments capture the execution outcome so a
    decision can be reconstructed later (audit issue H-03).
    """
    with db_connection(db_path) as conn:
        conn.execute('''
            INSERT INTO trades (timestamp, strategy, asset, action, price, quantity, pnl, reason,
                                signal_price, order_id, status, filled_qty, filled_avg_price,
                                strategy_position_before, account_position_before, risk_state, model_output)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (utc_now_iso(), strategy, asset, action, float(price or 0.0), float(quantity or 0.0), pnl, reason,
              signal_price, order_id, status, filled_qty, filled_avg_price,
              strategy_position_before, account_position_before, risk_state,
              None if model_output is None else str(model_output)))


def log_heartbeat(strategy, status="RUNNING", db_path=None, ready=None, degraded_reason=None):
    """
    Each bot calls this at least once per cycle to signal it is alive.
    Three separate signals (Phase 2 I-4):
      * liveness  : last_seen (stale -> DISCONNECTED / container unhealthy)
      * readiness : ready = 1 when the strategy may open NEW entries
      * degraded  : degraded_reason explains why it is not ready while it
                    keeps managing existing positions (stops, kill switch).
    ready/degraded_reason are only overwritten when given.
    """
    with db_connection(db_path) as conn:
        conn.execute('''
            INSERT INTO heartbeat (strategy, last_seen, status, ready, degraded_reason)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(strategy) DO UPDATE SET
                last_seen = excluded.last_seen,
                status    = excluded.status,
                ready     = COALESCE(excluded.ready, heartbeat.ready),
                degraded_reason = CASE WHEN excluded.ready IS NULL THEN heartbeat.degraded_reason
                                       ELSE excluded.degraded_reason END
        ''', (strategy, utc_now_iso(), status, None if ready is None else int(bool(ready)), degraded_reason))


def get_heartbeat(strategy, db_path=None):
    """Returns (last_seen, status) for a strategy, or None if it has never run."""
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT last_seen, status FROM heartbeat WHERE strategy = ?',
                           (strategy,)).fetchone()
    return row


def get_heartbeat_full(strategy, db_path=None, readonly=None):
    """{last_seen, status, ready, degraded_reason} or None if the strategy never ran."""
    with db_connection(db_path, readonly=readonly) as conn:
        row = conn.execute('SELECT last_seen, status, ready, degraded_reason FROM heartbeat WHERE strategy = ?',
                           (strategy,)).fetchone()
    if row is None:
        return None
    return {"last_seen": row[0], "status": row[1],
            "ready": None if row[2] is None else bool(row[2]), "degraded_reason": row[3]}


def get_trades(strategy=None, db_path=None):
    """
    Retrieves trades from the database as tuples in TRADES_COLUMNS order
    (the original 9 columns, so existing dashboard code keeps working).
    """
    cols = ", ".join(TRADES_COLUMNS)
    with db_connection(db_path) as conn:
        if strategy:
            rows = conn.execute(f'SELECT {cols} FROM trades WHERE strategy = ? ORDER BY id',
                                (strategy,)).fetchall()
        else:
            rows = conn.execute(f'SELECT {cols} FROM trades ORDER BY id').fetchall()
    return rows


def get_trades_full(strategy=None, db_path=None):
    """Retrieves trades with every column as a list of dicts."""
    with db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if strategy:
            rows = conn.execute('SELECT * FROM trades WHERE strategy = ? ORDER BY id',
                                (strategy,)).fetchall()
        else:
            rows = conn.execute('SELECT * FROM trades ORDER BY id').fetchall()
    return [dict(r) for r in rows]


# Initialize (or migrate) the database on import. Read-only processes (the
# dashboard) never write; they rely on the migrate service having run.
# Tools and run.py set SINGHQUANT_NO_AUTO_MIGRATE=1 before importing so they
# can check the storage first and migrate explicitly (or not at all).
if not READONLY and os.getenv("SINGHQUANT_NO_AUTO_MIGRATE", "") != "1":
    init_db()
