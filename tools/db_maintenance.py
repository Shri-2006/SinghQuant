"""
SQLite maintenance for SinghQuant (Phase 2 I-1 / I-2).

    python -m tools.db_maintenance migrate
        One-shot schema migration (the compose `migrate` service). Checks the
        storage first and exits non-zero with an explicit message on failure.

    python -m tools.db_maintenance check [--db PATH]
        Read-only health report: location, quick_check, journal mode, schema
        version, row counts. Never writes.

    python -m tools.db_maintenance checkpoint [--db PATH]
        Copies the write-ahead log into the main file and truncates it.
        Only complete when NO other process has the database open.

    python -m tools.db_maintenance backup DEST [--db PATH]
        Consistent online copy using SQLite's backup API (safe while readers
        or writers are active), then integrity_check on the copy. Refuses to
        overwrite DEST.

    python -m tools.db_maintenance cutover-import --db OLD_DB [--wal OLD_WAL] --dest NEW_DB
        One-time move from the old single-file Docker mount. Works on COPIES
        of the old database and its write-ahead log (the originals are never
        opened for writing), lets SQLite recover the log, then writes a
        verified backup to NEW_DB. See docs/DEPLOYMENT.md.

All commands default to SINGHQUANT_DB_PATH (or ./trades.db).
"""
import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SINGHQUANT_NO_AUTO_MIGRATE", "1")   # importing core.logger must not touch the database

from core import logger                       # noqa: E402  (import never writes in read-only mode)
from core.storage import storage_problems     # noqa: E402


def _ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def table_counts(conn):
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_migrate(db):
    problems = storage_problems(db, need_write=True)
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 2
    version = logger.init_db(db)
    print(f"migrated {db} to schema {version}")
    return 0


def cmd_check(db):
    problems = storage_problems(db, need_write=False)
    if not Path(db).exists():
        problems.append(f"{db} does not exist")
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 2
    conn = _ro(db)
    try:
        print(f"database      {Path(db).resolve()}")
        print(f"journal_mode  {conn.execute('PRAGMA journal_mode').fetchone()[0]}")
        print(f"schema        {conn.execute('PRAGMA user_version').fetchone()[0]} (code expects {logger.SCHEMA_VERSION})")
        print(f"quick_check   {conn.execute('PRAGMA quick_check').fetchone()[0]}")
        for name, n in table_counts(conn).items():
            print(f"  {name:20s} {n}")
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        side = Path(str(db) + suffix)
        print(f"{side.name:14s}{'present, ' + str(side.stat().st_size) + ' bytes' if side.exists() else 'absent'}")
    return 0


def cmd_checkpoint(db, timeout=30.0):
    conn = sqlite3.connect(db, timeout=timeout)
    try:
        busy, log_frames, done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    print(f"checkpoint: busy={busy} wal_frames={log_frames} checkpointed={done}")
    if busy:
        print("WARNING: another connection is using the database; the checkpoint is incomplete. "
              "Stop every writer (and the dashboard) and run it again.")
        return 1
    return 0


def backup(src, dest):
    """Consistent copy of `src` into the NEW file `dest` via the SQLite backup API. Returns table counts."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    source = _ro(src)
    target = sqlite3.connect(str(tmp))
    try:
        source.backup(target)                       # restarts automatically if a writer changes the source
        result = target.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"backup failed integrity_check: {result}")
        counts = table_counts(target)
        target.execute("PRAGMA journal_mode=DELETE")    # single self-contained file
    finally:
        target.close()
        source.close()
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    return counts


def cmd_backup(db, dest):
    counts = backup(db, dest)
    print(f"backup written to {dest} (integrity_check ok, sha256 {sha256(dest)})")
    for name, n in counts.items():
        print(f"  {name:20s} {n}")
    return 0


def cutover_import(old_db, old_wal, dest):
    """
    Recovers the old database WITH its write-ahead log from copies and writes a
    verified backup to `dest`. Returns (recovered_counts, dest_counts).
    """
    old_db = Path(old_db)
    if not old_db.is_file():
        raise FileNotFoundError(f"{old_db} is not a file")
    if Path(dest).exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite")
    with tempfile.TemporaryDirectory(prefix="singhquant-cutover-") as work:
        staged = Path(work) / "trades.db"
        shutil.copy2(old_db, staged)
        if old_wal:
            if not Path(old_wal).is_file():
                raise FileNotFoundError(f"{old_wal} is not a file")
            shutil.copy2(old_wal, str(staged) + "-wal")   # same name as the database + "-wal": SQLite replays it
        conn = sqlite3.connect(str(staged))             # read-write on the COPY so the log can be recovered
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"recovered database failed integrity_check: {result}")
            recovered = table_counts(conn)
        finally:
            conn.close()
        dest_counts = backup(staged, dest)
    if recovered != dest_counts:
        raise RuntimeError(f"row counts differ after backup: {recovered} vs {dest_counts}")
    return recovered, dest_counts


def cmd_cutover_import(old_db, old_wal, dest):
    recovered, _ = cutover_import(old_db, old_wal, dest)
    print(f"imported {old_db}{' + ' + str(old_wal) if old_wal else ''} -> {dest} (integrity_check ok, counts match)")
    for name, n in recovered.items():
        print(f"  {name:20s} {n}")
    print(f"sha256 {sha256(dest)}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("migrate", "check", "checkpoint"):
        sp = sub.add_parser(name)
        sp.add_argument("--db", default=logger.DB_PATH)
        if name == "checkpoint":
            sp.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for other connections")
    sp = sub.add_parser("backup")
    sp.add_argument("dest")
    sp.add_argument("--db", default=logger.DB_PATH)
    sp = sub.add_parser("cutover-import")
    sp.add_argument("--db", required=True, help="the OLD database file (a copy is used)")
    sp.add_argument("--wal", default=None, help="the OLD container's trades.db-wal, if one exists")
    sp.add_argument("--dest", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "migrate":
            return cmd_migrate(a.db)
        if a.cmd == "check":
            return cmd_check(a.db)
        if a.cmd == "checkpoint":
            return cmd_checkpoint(a.db, a.timeout)
        if a.cmd == "backup":
            return cmd_backup(a.db, a.dest)
        return cmd_cutover_import(a.db, a.wal, a.dest)
    except (sqlite3.Error, OSError, RuntimeError, logger.SchemaTooNew) as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
