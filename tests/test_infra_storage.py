"""
Phase 2 I-1: the database, its write-ahead log and its shared-memory index
must live together on persistent storage, and the cutover from the old
single-file mount must not lose committed writes.

The old compose file mounted only ./trades.db, so trades.db-wal and
trades.db-shm were created inside each container's private filesystem:
the bot and dashboard did not share a WAL index, and committed transactions
that had not been checkpointed vanished when the container was recreated.
"""
import hashlib
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from core.logger import init_db
from core.storage import storage_problems
from tools import db_maintenance as dbm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# Writes rows and dies WITHOUT closing the connection or checkpointing, like
# a container killed by SIGKILL: the rows exist only in trades.db-wal.
_KILLED_WRITER = r"""
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA wal_autocheckpoint=0")
conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, asset TEXT)")
for i in range(int(sys.argv[2])):
    conn.execute("INSERT INTO trades (asset) VALUES (?)", (f"SYM{i}",))
    conn.commit()
os._exit(0)
"""


def _killed_writer(path, n):
    r = subprocess.run([sys.executable, "-c", _KILLED_WRITER, path, str(n)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# Shared directory: WAL and SHM next to the database, visible to other processes
# ---------------------------------------------------------------------------

def test_wal_and_shm_live_next_to_the_database_and_other_processes_see_commits(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    db = str(data / "trades.db")
    _killed_writer(db, 25)
    assert (data / "trades.db-wal").exists()                    # the log is on the shared volume ...
    assert (data / "trades.db-wal").stat().st_size > 0
    reader = r"""
import sqlite3, sys
from pathlib import Path
conn = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + "?mode=ro", uri=True)
print(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
"""
    r = subprocess.run([sys.executable, "-c", reader, db], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "25", r.stderr                    # ... so a reader (the dashboard) sees every commit


def test_the_old_single_file_layout_loses_uncheckpointed_commits(tmp_path):
    """Documents the hazard the directory volume fixes: copy only the .db file and commits are gone."""
    db = str(tmp_path / "trades.db")
    _killed_writer(db, 25)
    lone = tmp_path / "lone"
    lone.mkdir()
    import shutil
    shutil.copy2(db, lone / "trades.db")                          # what survives on the host with the old mount
    conn = sqlite3.connect(str(lone / "trades.db"))
    try:
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] if \
            conn.execute("SELECT name FROM sqlite_master WHERE name='trades'").fetchone() else 0
    finally:
        conn.close()
    assert n < 25


# ---------------------------------------------------------------------------
# Storage preflight: explicit errors instead of silent data loss
# ---------------------------------------------------------------------------

def test_storage_check_explains_a_docker_created_directory(tmp_path):
    db = tmp_path / "trades.db"
    db.mkdir()                                                   # what Docker does for a missing bind-mounted file
    problems = storage_problems(str(db))
    assert len(problems) == 1 and "is a directory" in problems[0] and "./data:/app/data" in problems[0]


def test_storage_check_reports_missing_and_unwritable_directories(tmp_path, monkeypatch):
    assert "does not exist" in storage_problems(str(tmp_path / "nope" / "trades.db"))[0]
    monkeypatch.setattr(os, "access", lambda path, mode: False)   # tests run as root, which bypasses chmod
    problems = storage_problems(str(tmp_path / "trades.db"))
    assert len(problems) == 1 and "not writable" in problems[0] and "-wal" in problems[0]


def test_storage_check_detects_a_corrupt_database(tmp_path):
    db = tmp_path / "trades.db"
    db.write_bytes(b"this is not a sqlite database" * 100)
    assert "cannot be read" in storage_problems(str(db))[0]


def test_storage_check_passes_a_healthy_database(tmp_path):
    db = str(tmp_path / "trades.db")
    init_db(db)
    assert storage_problems(db) == []


# ---------------------------------------------------------------------------
# db_maintenance commands
# ---------------------------------------------------------------------------

def test_migrate_command_refuses_bad_storage_with_an_explicit_error(tmp_path, capsys):
    bad = tmp_path / "trades.db"
    bad.mkdir()
    assert dbm.main(["migrate", "--db", str(bad)]) == 2
    assert "is a directory" in capsys.readouterr().out
    good = str(tmp_path / "data.db")
    assert dbm.main(["migrate", "--db", good]) == 0


def test_check_command_never_writes(tmp_path):
    db = str(tmp_path / "trades.db")
    init_db(db)
    sqlite3.connect(db).execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
    digest = _sha(db)
    env = dict(os.environ, SINGHQUANT_DB_PATH=db)
    r = subprocess.run([sys.executable, "-m", "tools.db_maintenance", "check"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "quick_check   ok" in r.stdout and "schema        1" in r.stdout
    assert _sha(db) == digest


def test_checkpoint_command_folds_the_log_into_the_database(tmp_path, capsys):
    db = str(tmp_path / "trades.db")
    _killed_writer(db, 10)
    assert os.path.getsize(db + "-wal") > 0
    assert dbm.main(["checkpoint", "--db", db]) == 0
    assert not os.path.exists(db + "-wal") or os.path.getsize(db + "-wal") == 0
    lone = tmp_path / "lone"
    lone.mkdir()
    import shutil
    shutil.copy2(db, lone / "trades.db")                     # the main file alone now holds every commit
    assert sqlite3.connect(str(lone / "trades.db")).execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 10


def test_checkpoint_reports_when_a_writer_is_still_attached(tmp_path, capsys):
    db = str(tmp_path / "trades.db")
    init_db(db)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM trades").fetchone()          # an open read transaction pins the log
    other = sqlite3.connect(db)
    other.execute("INSERT INTO trades (timestamp,strategy,asset,action,price,quantity) VALUES ('t','s','A','BUY',1,1)")
    other.commit(); other.close()
    try:
        code = dbm.main(["checkpoint", "--db", db, "--timeout", "0.5"])
        out = capsys.readouterr().out
        assert code == 1 and "Stop every writer" in out
    finally:
        holder.execute("ROLLBACK"); holder.close()


def test_backup_is_consistent_while_a_writer_keeps_writing(tmp_path):
    db = str(tmp_path / "trades.db")
    init_db(db)
    stop = threading.Event()

    def writer():
        conn = sqlite3.connect(db, timeout=30)
        while not stop.is_set():
            conn.execute("INSERT INTO trades (timestamp,strategy,asset,action,price,quantity) VALUES ('t','s','A','BUY',1,1)")
            conn.commit()
        conn.close()

    t = threading.Thread(target=writer)
    t.start()
    try:
        counts = dbm.backup(db, str(tmp_path / "backup" / "trades.db"))
    finally:
        stop.set(); t.join()
    copy = sqlite3.connect(str(tmp_path / "backup" / "trades.db"))
    assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert copy.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == counts["trades"]
    assert copy.execute("PRAGMA journal_mode").fetchone()[0] == "delete"     # self-contained single file
    with pytest.raises(FileExistsError):
        dbm.backup(db, str(tmp_path / "backup" / "trades.db"))               # never overwrites


def test_cutover_import_recovers_the_old_containers_log_from_copies(tmp_path, capsys):
    """The runbook path: host trades.db + the trades.db-wal copied out of the stopped old container."""
    old = tmp_path / "old"
    old.mkdir()
    old_db = str(old / "trades.db")
    _killed_writer(old_db, 40)
    # simulate `docker cp`: the log sits somewhere else, under another name
    copied = tmp_path / "from_container"
    copied.mkdir()
    os.replace(old_db + "-wal", copied / "trades.db-wal")
    if os.path.exists(old_db + "-shm"):
        os.remove(old_db + "-shm")
    before_db, before_wal = _sha(old_db), _sha(str(copied / "trades.db-wal"))
    dest = str(tmp_path / "data" / "trades.db")
    assert dbm.main(["cutover-import", "--db", old_db, "--wal", str(copied / "trades.db-wal"), "--dest", dest]) == 0
    assert "counts match" in capsys.readouterr().out
    conn = sqlite3.connect(dest)
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 40       # nothing lost
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()
    assert _sha(old_db) == before_db and _sha(str(copied / "trades.db-wal")) == before_wal   # originals untouched
    assert dbm.main(["cutover-import", "--db", old_db, "--dest", dest]) == 2                  # refuses to overwrite
    init_db(dest)                                                                # the new stack migrates the import
    assert sqlite3.connect(dest).execute("PRAGMA user_version").fetchone()[0] == 1
