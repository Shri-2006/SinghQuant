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
from contextlib import contextmanager
from datetime import datetime, timezone

# SINGHQUANT_DB_PATH lets tests and tooling point at another database file.
DB_PATH = os.getenv("SINGHQUANT_DB_PATH") or os.path.join(os.path.dirname(__file__), '..', 'trades.db')

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


@contextmanager
def db_connection(db_path=None):
    """Yields a connection with WAL and a busy timeout; commits on success."""
    path = db_path or DB_PATH
    with _DB_LOCK:
        conn = sqlite3.connect(path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
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


def init_db(db_path=None):
    """Creates trades and heartbeat tables if they don't exist. Enables WAL mode."""
    with db_connection(db_path) as conn:
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


def log_heartbeat(strategy, status="RUNNING", db_path=None):
    """
    Each bot calls this at least once per cycle to signal it is alive.
    Dashboard checks last_seen; if stale it shows DISCONNECTED.
    """
    with db_connection(db_path) as conn:
        conn.execute('''
            INSERT INTO heartbeat (strategy, last_seen, status)
            VALUES (?, ?, ?)
            ON CONFLICT(strategy) DO UPDATE SET
                last_seen = excluded.last_seen,
                status    = excluded.status
        ''', (strategy, utc_now_iso(), status))


def get_heartbeat(strategy, db_path=None):
    """Returns (last_seen, status) for a strategy, or None if it has never run."""
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT last_seen, status FROM heartbeat WHERE strategy = ?',
                           (strategy,)).fetchone()
    return row


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


# Initialize database on import
init_db()
