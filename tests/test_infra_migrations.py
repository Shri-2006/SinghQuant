"""
Phase 2 I-2: schema migrations are transactional, versioned and race-free,
and the dashboard reads the database without ever writing to it.

The 2026-10-11 review reproduced 34 crashes in 80 process starts when two
processes (the bot and dashboard containers) migrated a legacy database at
the same time ("duplicate column name", "database is locked").
"""
import hashlib
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from core import logger
from core.logger import init_db, SCHEMA_VERSION, SchemaTooNew

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _legacy_main_db(path):
    """The schema `main` deploys today: trades + heartbeat only, rows present."""
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL, asset TEXT NOT NULL, action TEXT NOT NULL, price REAL NOT NULL,
                    quantity REAL NOT NULL, pnl REAL, reason TEXT)''')
    conn.execute('CREATE TABLE heartbeat (strategy TEXT PRIMARY KEY, last_seen TEXT NOT NULL, status TEXT NOT NULL)')
    conn.executemany("INSERT INTO trades (timestamp,strategy,asset,action,price,quantity,pnl,reason) VALUES (?,?,?,?,?,?,?,?)",
                     [(f"2026-08-{d:02d}T15:00:00", "stable", "SPY", "BUY", 500.0, 0.4, None, "legacy") for d in range(1, 29)])
    conn.commit()
    conn.close()


def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _user_version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


_RACE_WORKER = r"""
import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ["SINGHQUANT_DB_PATH"] = sys.argv[3]      # import against a private file ...
from core import logger
start_at = float(sys.argv[2])
while time.time() < start_at:                       # ... then all processes migrate the shared file at once
    pass
try:
    logger.init_db(sys.argv[1])
    print("ok")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
"""


@pytest.mark.parametrize("trial", range(10))
def test_simultaneous_process_starts_never_fail_or_half_migrate(tmp_path, trial):
    """Includes the one-time rollback-journal -> WAL switch, which SQLite does not retry on its own."""
    path = str(tmp_path / "trades.db")
    _legacy_main_db(path)
    start_at = time.time() + 2.0
    env = {k: v for k, v in os.environ.items() if k != "SINGHQUANT_DB_READONLY"}
    procs = [subprocess.Popen([sys.executable, "-c", _RACE_WORKER, path, str(start_at), str(tmp_path / f"own{n}.db")],
                              cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for n in range(4)]
    outs = [p.communicate(timeout=120) for p in procs]
    results = [o[0].strip().splitlines()[-1] if o[0].strip() else o[1][-300:] for o in outs]
    assert results == ["ok"] * 4, results
    assert _user_version(path) == SCHEMA_VERSION
    assert {"status", "filled_qty", "signal_price"} <= _columns(path, "trades")
    assert {"ready", "degraded_reason"} <= _columns(path, "heartbeat")
    assert {"discrepancy_qty", "last_mark"} <= _columns(path, "positions")
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 28          # history untouched
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_failed_migration_rolls_back_completely(tmp_path, monkeypatch):
    path = str(tmp_path / "trades.db")
    _legacy_main_db(path)
    before_cols = _columns(path, "trades")

    real = logger._migrate
    def explode_halfway(conn):
        conn.execute("ALTER TABLE trades ADD COLUMN signal_price REAL")     # the first of many changes ...
        conn.execute("CREATE TABLE positions (x INTEGER)")
        raise RuntimeError("simulated crash in the middle of the migration")
    monkeypatch.setattr(logger, "_migrate", explode_halfway)
    with pytest.raises(RuntimeError):
        init_db(path)
    assert _columns(path, "trades") == before_cols                           # ... none of them stuck
    assert "positions" not in {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master")}
    assert _user_version(path) == 0

    monkeypatch.setattr(logger, "_migrate", real)
    assert init_db(path) == SCHEMA_VERSION                                  # a clean retry succeeds
    assert "signal_price" in _columns(path, "trades")


def test_current_schema_takes_no_write_lock(tmp_path):
    """The fast path must not wait for (or take) the write lock a running bot may hold."""
    path = str(tmp_path / "trades.db")
    init_db(path)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")                       # another process is mid-write
    try:
        t0 = time.monotonic()
        assert init_db(path) == SCHEMA_VERSION
        assert time.monotonic() - t0 < 2.0
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_database_from_newer_code_is_refused_untouched(tmp_path):
    path = str(tmp_path / "trades.db")
    init_db(path)
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit(); conn.close()
    digest = _sha(path)
    with pytest.raises(SchemaTooNew):
        init_db(path)
    assert _sha(path) == digest


def test_migration_is_idempotent_and_preserves_ledger_rows(tmp_path):
    from core import portfolio
    path = str(tmp_path / "trades.db")
    init_db(path)
    portfolio.adopt_position("stable", "SPY", 2.0, 100.0, path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 0")                 # pretend it was never stamped
    conn.commit(); conn.close()
    init_db(path)
    init_db(path)
    assert portfolio.get_position("stable", "SPY", path)[:2] == (2.0, 100.0)
    assert _user_version(path) == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Read-only dashboard mode (needs its own process: the flag is read at import)
# ---------------------------------------------------------------------------

_READONLY_WORKER = r"""
import os, sys, sqlite3
sys.path.insert(0, os.getcwd())
from core import logger, portfolio
assert logger.READONLY
st = portfolio.get_strategy_state("risky1")            # never ran: defaults, no INSERT
print("state", st["cash_budget"], st["halted"])
print("trades", len(logger.get_trades("stable")))
print("hb", logger.get_heartbeat_full("stable"))
print("equity", round(portfolio.strategy_equity("stable", {"SPY": 110.0})[0], 2))
try:
    logger.log_trade("stable", "SPY", "BUY", 1.0, 1.0)
    print("write: ALLOWED")
except sqlite3.OperationalError as e:
    print("write: refused", "readonly" in str(e).lower())
"""


def _run_readonly(path, code=_READONLY_WORKER):
    env = dict(os.environ, SINGHQUANT_DB_PATH=path, SINGHQUANT_DB_READONLY="1")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_readonly_dashboard_never_writes(tmp_path):
    from core import portfolio
    from core.logger import log_heartbeat
    path = str(tmp_path / "trades.db")
    init_db(path)
    portfolio.adopt_position("stable", "SPY", 2.0, 100.0, path)
    log_heartbeat("stable", "RUNNING", path, ready=False, degraded_reason="model missing")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    digest = _sha(path)
    r = _run_readonly(path)
    assert r.returncode == 0, r.stderr[-800:]
    out = r.stdout
    assert "state 200.0 False" in out
    assert "'ready': False" in out and "model missing" in out
    assert "equity 1020.0" in out                         # 800 cash + 2 x 110
    assert "write: refused True" in out
    assert _sha(path) == digest                           # byte-identical: nothing written
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM strategy_state WHERE strategy = 'risky1'").fetchone()[0] == 0
    conn.close()


def test_readonly_reader_reports_an_unmigrated_database_instead_of_migrating_it(tmp_path):
    path = str(tmp_path / "trades.db")
    _legacy_main_db(path)
    digest = _sha(path)
    code = r"""
import os, sys
sys.path.insert(0, os.getcwd())
from core import logger
try:
    logger.init_db()
    print("migrated?!")
except logger.SchemaNotReady as e:
    print("not ready")
print("ready", logger.schema_ready(readonly=True))
"""
    r = _run_readonly(path, code)
    assert r.returncode == 0, r.stderr[-800:]
    assert "not ready" in r.stdout and "ready False" in r.stdout
    assert _sha(path) == digest
    assert _user_version(path) == 0
