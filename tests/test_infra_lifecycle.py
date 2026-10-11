"""
Phase 2 I-4: container startup and recovery.

* SIGTERM (docker stop) stops the bot cleanly. Before, Python as PID 1 had no
  handler, so Docker waited 10 s and SIGKILLed it mid-cycle.
* Long market-closed sleeps keep the heartbeat fresh (they used to leave it
  stale for up to 48 h, which looks exactly like a dead bot).
* Liveness, readiness and degraded status are separate: a degraded strategy
  (no model, suspect data, halted, safe mode) is alive and still managing
  positions, and the watchdog never restarts it. Only a dead or hung thread
  is a process failure.
* Restarts are bounded and cannot reset safeguards: unclean exits are
  counted in the database, a crash loop enters persistent SAFE MODE (no new
  entries) and, past the restart budget, the watchdog stops exiting.
* After any (re)start, no new entry until orders and positions reconcile.
"""
import hashlib
import itertools
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import run
from core import lifecycle, portfolio
from core.execution import submit_and_track
from core.logger import get_heartbeat_full, log_heartbeat, db_connection
from strategies.common import run_cycle, run_forever, adopt_unowned_broker_positions
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca, FakeAPIError
from tests.test_engine import make_ctx, always
from tools import healthcheck, safe_mode as safe_mode_tool

FAST = dict(sleep_fn=lambda s: None)
_clock = itertools.count(0, 30)
EXEC = dict(sleep_fn=lambda s: None, clock_fn=lambda: next(_clock))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alerts():
    sent = []
    return sent, (lambda **kw: sent.append(kw))


def _set_last_seen(db_path, strategy, seconds_ago, status="RUNNING", ready=True):
    log_heartbeat(strategy, status, db_path, ready=ready, degraded_reason=None if ready else "degraded")
    ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).replace(tzinfo=None).isoformat()
    with db_connection(db_path) as conn:
        conn.execute("UPDATE heartbeat SET last_seen = ? WHERE strategy = ?", (ts, strategy))


# ---------------------------------------------------------------------------
# Crash-loop accounting and SAFE MODE
# ---------------------------------------------------------------------------

def test_unrecorded_exits_are_counted_as_unclean_and_clean_ones_are_not(db_path):
    r1, n = lifecycle.record_start(db_path)
    assert n == 0
    r2, n = lifecycle.record_start(db_path)                 # r1 never recorded an end: SIGKILL / crash
    assert n == 1
    lifecycle.record_end(r2, lifecycle.CLEAN, db_path)
    r3, n = lifecycle.record_start(db_path)
    assert n == 1                                           # the clean stop is not counted
    runs = {r["id"]: r["end_reason"] for r in lifecycle.recent_runs(db_path=db_path)}
    assert runs[r1] == lifecycle.UNCLEAN_UNRECORDED and runs[r2] == "clean" and runs[r3] is None


def test_crash_loop_enters_safe_mode_once_and_restarts_cannot_clear_it(db_path):
    sent, alert = _alerts()
    for _ in range(4):                                      # 3 unclean exits, then a 4th start
        run.Supervisor({}, db_path=db_path, alert=alert).begin()
    assert lifecycle.get_safe_mode(db_path)["reason"].startswith("3 unclean exits")
    assert [a["alert_type"] for a in sent] == ["CRASH LOOP: SAFE MODE"]
    sup = run.Supervisor({}, db_path=db_path, alert=alert)  # a CLEAN restart later ...
    sup.begin(); sup.shutdown("clean")
    sup = run.Supervisor({}, db_path=db_path, alert=alert)
    sup.begin(); sup.shutdown("clean")
    assert lifecycle.get_safe_mode(db_path) is not None     # ... does not clear it
    assert len(sent) == 1


def test_safe_mode_blocks_new_entries_but_not_the_stop_loss(db_path):
    lifecycle.set_safe_mode("test crash loop", db_path)
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    api.set_price("SPY", 90.0)
    frames = {"SPY": make_featured_df(price=90.0, atr=1.35), "QQQ": make_featured_df(price=100.0, atr=1.5)}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["SPY"] == "emergency-exit"
    assert out["tickers"]["QQQ"] == "entry-blocked"
    assert out["ready"] is False and "SAFE MODE: test crash loop" in out["degraded_reason"]
    assert not any(o["side"] == "buy" for o in api.submitted[1:])


def test_safe_mode_tool_requires_confirmation(db_path, capsys):
    lifecycle.set_safe_mode("crash loop", db_path)
    assert safe_mode_tool.main(["clear", "--note", "fixed"], db_path=db_path) == 0
    assert lifecycle.get_safe_mode(db_path) is not None and "DRY RUN" in capsys.readouterr().out
    assert safe_mode_tool.main(["clear", "--note", "root cause fixed", "--confirm"], db_path=db_path) == 0
    assert lifecycle.get_safe_mode(db_path) is None


# ---------------------------------------------------------------------------
# Watchdog: process death vs degraded-but-functioning
# ---------------------------------------------------------------------------

def _sup(db_path, targets, **kw):
    sent, alert = _alerts()
    clock = kw.pop("clock", None) or itertools.count(0, 1).__next__
    sup = run.Supervisor(targets, db_path=db_path, alert=alert, clock=clock, **kw)
    return sup, sent


def _alive_until(stop):
    stop.wait(30)


def test_degraded_but_alive_strategy_is_not_a_failure(db_path):
    sup, sent = _sup(db_path, {"stable": _alive_until}, hang_seconds=600, clock=lambda: 10_000.0)
    sup.begin()
    sup._started_at = 0.0
    _set_last_seen(db_path, "stable", 5, ready=False)        # model missing / halted / suspect: fresh, not ready
    try:
        assert sup.check() == ([], [])
    finally:
        sup.shutdown("clean")
    assert sent == []


def test_dead_thread_exits_for_a_restart_while_the_budget_lasts(db_path):
    sup, sent = _sup(db_path, {"stable": lambda stop: None}, interval=0.01)
    sup.begin()
    sup.watch()                                               # returns once the dead thread is seen
    assert sup.exit_code == run.EXIT_WATCHDOG and sup.stop_event.is_set()
    assert [a["alert_type"] for a in sent] == ["STRATEGY THREAD FAILED"]
    sup.shutdown("watchdog_exit")
    assert lifecycle.recent_runs(1, db_path)[0]["end_reason"] == "watchdog_exit"


def test_hung_thread_is_a_failure(db_path):
    t = {"now": 0.0}
    sup, sent = _sup(db_path, {"stable": _alive_until}, hang_seconds=600, clock=lambda: t["now"])
    sup.begin()
    _set_last_seen(db_path, "stable", 10)
    t["now"] = 100.0
    assert sup.check() == ([], [])                            # too early to judge
    t["now"] = 5000.0
    _set_last_seen(db_path, "stable", 1200)                   # alive thread, no heartbeat for 20 min
    try:
        assert sup.check() == ([], ["stable"])
    finally:
        sup.shutdown("clean")


def test_restart_budget_exhausted_keeps_the_process_up_and_alerts_once(db_path):
    for _ in range(5):                                        # 5 earlier unclean exits
        lifecycle.record_start(db_path)
    sup, sent = _sup(db_path, {"stable": lambda stop: None, "risky1": _alive_until})
    sup.begin()
    try:
        dead, hung = sup.check()
        assert dead == ["stable"]
        assert sup.handle_failure(dead, hung) is False        # no exit -> no endless restart loop
        assert sup.handle_failure(dead, hung) is False
        assert sup.exit_code is None and not sup.stop_event.is_set()
        kinds = [a["alert_type"] for a in sent]
        assert kinds.count("STRATEGY THREAD FAILED - NOT RESTARTING") == 1
        assert "CRASH LOOP: SAFE MODE" in kinds and lifecycle.get_safe_mode(db_path) is not None
    finally:
        sup.shutdown("clean")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _strategy_target(db_path, api, name="stable", **ctx_kw):
    def target(stop):
        ctx = make_ctx(name, api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, db_path=db_path,
                       cycle_sleep=3600, **ctx_kw)
        run_forever(ctx, api=api, stop_event=stop)
    return target


def test_stop_request_wakes_sleeping_threads_and_records_a_clean_exit(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    sup, _ = _sup(db_path, {"stable": _strategy_target(db_path, api)}, clock=time.monotonic, grace_seconds=10)
    sup.begin()
    deadline = time.time() + 10
    while time.time() < deadline and not get_heartbeat_full("stable", db_path):
        time.sleep(0.05)
    time.sleep(0.2)                                            # now in the 1-hour cycle sleep
    t0 = time.monotonic()
    sup.shutdown("clean")
    assert time.monotonic() - t0 < 5
    assert not sup.threads["stable"].is_alive()
    assert get_heartbeat_full("stable", db_path)["status"] == "STOPPED"
    assert lifecycle.recent_runs(1, db_path)[0]["end_reason"] == "clean"


_SIGTERM_HARNESS = r"""
import os, sys, threading
sys.path.insert(0, os.getcwd())
import run
from strategies.common import run_forever
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca
from tests.test_engine import make_ctx
db = sys.argv[1]                              # not the env var: tests.conftest repoints it on import
from core.logger import init_db
init_db(db)                                   # run.main() does this in storage_and_schema_preflight()
api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
def target(stop):
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, db_path=db, cycle_sleep=3600)
    run_forever(ctx, api=api, stop_event=stop)
stop = threading.Event()
if os.environ.get("INSTALL_HANDLERS") == "1":
    run.install_signal_handlers(stop)
sup = run.Supervisor({"stable": target}, db_path=db, alert=lambda **kw: None, stop_event=stop, interval=0.2)
sup.begin()
print("READY", flush=True)
sup.watch()
sup.shutdown("clean")
sys.exit(0)
"""


def _start_harness(tmp_path, install):
    db = str(tmp_path / "trades.db")
    env = dict(os.environ, SINGHQUANT_DB_PATH=db, INSTALL_HANDLERS="1" if install else "0")
    proc = subprocess.Popen([sys.executable, "-c", _SIGTERM_HARNESS, db], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(50):                                   # skip the supervisor's own startup lines
        line = proc.stdout.readline()
        if line.strip() == "READY" or not line:
            break
    assert line.strip() == "READY", proc.stderr.read()[-800:]
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if get_heartbeat_full("stable", db):
                break
        except sqlite3.Error:
            pass
        time.sleep(0.1)
    return proc, db


def test_sigterm_stops_a_real_process_cleanly(tmp_path):
    proc, db = _start_harness(tmp_path, install=True)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=20)
    assert proc.returncode == 0, proc.stderr.read()[-800:]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT end_reason FROM process_runs ORDER BY id DESC LIMIT 1").fetchone()[0] == "clean"
    assert conn.execute("SELECT status FROM heartbeat WHERE strategy='stable'").fetchone()[0] == "STOPPED"
    conn.close()


def test_without_the_handler_sigterm_kills_mid_cycle_and_is_counted_unclean(tmp_path):
    """Control: the old behaviour. The next start counts it as an unclean exit."""
    proc, db = _start_harness(tmp_path, install=False)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=20)
    assert proc.returncode == -signal.SIGTERM
    _, unclean = lifecycle.record_start(db)
    assert unclean == 1


# ---------------------------------------------------------------------------
# Heartbeats while paused
# ---------------------------------------------------------------------------

def test_market_closed_sleep_is_chunked_and_keeps_the_heartbeat_fresh(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, market_open=False)   # next open in 15 h -> sleeps 8 h
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0)}, db_path=db_path)
    sleeps, beats = [], []
    def sleep_fn(secs):
        sleeps.append(secs)
        beats.append(get_heartbeat_full("stable", db_path)["last_seen"])
    run_forever(ctx, api=api, sleep_fn=sleep_fn, max_cycles=1)
    assert sum(sleeps) == 8 * 3600 and max(sleeps) <= 300
    assert len(set(beats)) > 1                                    # refreshed between chunks
    assert healthcheck.collect(db_path)["strategies"]["stable"]["alive"] is True


def test_stop_request_ends_a_long_market_closed_sleep(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, market_open=False)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0)}, db_path=db_path)
    stop = threading.Event()
    calls = []
    def sleep_fn(secs):
        calls.append(secs)
        if len(calls) == 3:
            stop.set()
    run_forever(ctx, api=api, sleep_fn=sleep_fn, stop_event=stop)
    assert len(calls) == 3                                         # not 96 chunks
    assert get_heartbeat_full("stable", db_path)["status"] == "STOPPED"


# ---------------------------------------------------------------------------
# Safe startup reconciliation
# ---------------------------------------------------------------------------

def test_no_entries_until_held_positions_are_checked_against_the_broker(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    frames = {"SPY": make_featured_df(price=100.0, atr=1.5), "QQQ": make_featured_df(price=100.0, atr=1.5)}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)    # a fresh process
    api.fail_get_position = FakeAPIError("service unavailable", 503)                 # broker flaky at startup
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["QQQ"] == "entry-blocked"
    assert "startup reconciliation pending" in out["degraded_reason"]
    api.fail_get_position = None
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["QQQ"] == "buy-filled" and out["ready"] is True


def test_restart_reconciles_before_adopting_so_a_fill_is_not_counted_twice(db_path, monkeypatch):
    """Review probe P3: ADOPT=1 + a BUY that filled while the process was down."""
    monkeypatch.setenv("SINGHQUANT_ADOPT_BROKER_POSITIONS", "1")
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, fill_mode="pending")
    res = submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **EXEC)
    api.fill_pending(res.order_id)                                # fills while the bot is down
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, db_path=db_path)
    run_forever(ctx, api=api, sleep_fn=lambda s: None, max_cycles=1)
    assert api.positions["SPY"]["qty"] == pytest.approx(2.0)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(2.0)   # was 4.0


def test_adoption_skips_a_symbol_with_an_open_order(db_path, monkeypatch):
    monkeypatch.setenv("SINGHQUANT_ADOPT_BROKER_POSITIONS", "1")
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, fill_mode="pending")
    submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **EXEC)
    api.seed_position("SPY", 2.0, 100.0)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0)}, db_path=db_path)
    assert adopt_unowned_broker_positions(ctx, api) == []


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def test_liveness_and_readiness_are_reported_separately(db_path):
    _set_last_seen(db_path, "stable", 5, ready=True)
    _set_last_seen(db_path, "risky1", 5, ready=False)             # degraded: e.g. model missing
    assert healthcheck.main(["liveness"], db_path=db_path) == 0    # alive -> container healthy
    assert healthcheck.main(["readiness"], db_path=db_path) == 1   # but not ready for new entries
    state = healthcheck.collect(db_path)
    assert state["strategies"]["risky1"] == {"alive": True, "ready": False, "status": "RUNNING",
                                            "age_seconds": pytest.approx(5, abs=3), "degraded_reason": "degraded"}


@pytest.mark.parametrize("age, status, alive", [(5, "PAUSED", True), (5, "HALTED", True), (5, "SUSPECT", True),
                                                (5, "STOPPED", False), (1200, "RUNNING", False)])
def test_liveness_by_status_and_age(db_path, age, status, alive):
    _set_last_seen(db_path, "stable", age, status=status)
    _set_last_seen(db_path, "risky1", 5)
    assert (healthcheck.main(["liveness"], db_path=db_path) == 0) is alive


def test_missing_heartbeat_is_not_alive_and_safe_mode_is_not_ready(db_path):
    _set_last_seen(db_path, "stable", 5)
    assert healthcheck.main(["liveness"], db_path=db_path) == 1           # risky1 never reported
    _set_last_seen(db_path, "risky1", 5)
    assert healthcheck.main(["readiness"], db_path=db_path) == 0
    lifecycle.set_safe_mode("crash loop", db_path)
    assert healthcheck.main(["readiness"], db_path=db_path) == 1
    assert healthcheck.main(["liveness"], db_path=db_path) == 0


def test_healthcheck_never_writes(db_path):
    _set_last_seen(db_path, "stable", 5)
    _set_last_seen(db_path, "risky1", 5)
    sqlite3.connect(db_path).execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
    digest = hashlib.sha256(open(db_path, "rb").read()).hexdigest()
    env = dict(os.environ, SINGHQUANT_DB_PATH=db_path)
    r = subprocess.run([sys.executable, "-m", "tools.healthcheck", "report"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and '"alive": true' in r.stdout, r.stderr[-500:]
    assert hashlib.sha256(open(db_path, "rb").read()).hexdigest() == digest


# ---------------------------------------------------------------------------
# Startup preflight
# ---------------------------------------------------------------------------

def test_bad_storage_is_a_fatal_explicit_startup_error(tmp_path):
    bad = tmp_path / "trades.db"
    bad.mkdir()
    problems = run.storage_and_schema_preflight(str(bad))
    assert problems and "is a directory" in problems[0]
    sent, alert = _alerts()
    stop = threading.Event(); stop.set()                          # don't actually wait 5 minutes
    assert run._fatal(problems, stop, alert=alert) == run.EXIT_FATAL_STARTUP
    assert sent[0]["alert_type"] == "STARTUP FAILED" and "is a directory" in sent[0]["message"]
    assert run.storage_and_schema_preflight(str(tmp_path / "ok.db")) == []
