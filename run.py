"""
SinghQuant process entry point and supervisor.

Phase 2 I-4 (container startup and recovery):
  * preflight: live-endpoint guard, database storage check, schema
    migration (backstop; the compose `migrate` service runs first);
    a fatal error is alerted and exits non-zero after a delay;
  * crash-loop accounting in the database (core/lifecycle.py): repeated
    unclean exits put the system in persistent SAFE MODE (no new entries,
    risk management continues), and stop the watchdog from exiting the
    process once the restart budget is spent (no endless restart loop);
  * SIGTERM/SIGINT (docker stop) -> every strategy thread finishes its
    current step, the scheduler stops, the run is recorded as clean, exit 0;
  * watchdog: a strategy thread that DIED or HUNG (alive, no heartbeat for
    WATCHDOG_HANG_SECONDS) is process failure -> alert, clean stop of the
    other threads, exit 3 so `restart: unless-stopped` restarts the
    container. A DEGRADED strategy (model missing, suspect data, halted,
    safe mode) keeps its heartbeat fresh and is NOT restarted: it is still
    managing its positions.
"""
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone

# Check the storage and migrate explicitly instead of as an import side effect.
os.environ.setdefault("SINGHQUANT_NO_AUTO_MIGRATE", "1")

from core.config import (FEATURE_FLAGS, PAPER_MODE, ALLOW_LIVE_TRADING, WATCHDOG_INTERVAL_SECONDS,  # noqa: E402
                         WATCHDOG_HANG_SECONDS, SHUTDOWN_GRACE_SECONDS, SAFE_MODE_AFTER_UNCLEAN_EXITS,
                         MAX_WATCHDOG_RESTARTS, FATAL_STARTUP_EXIT_DELAY_SECONDS)

EXIT_CLEAN = 0
EXIT_FATAL_STARTUP = 2
EXIT_WATCHDOG = 3


def preflight():
    """Refuses to start if any strategy would reach a live endpoint without opt-in."""
    from paper_trading.alpaca_paper import resolve_base_url
    for s in ("stable", "risky1", "risky2"):
        url = resolve_base_url(s)  # raises LiveTradingBlocked if misconfigured
        mode = "PAPER" if PAPER_MODE[s] else "LIVE (explicit opt-in)"
        print(f"  {s:7s} -> {mode}: {url}")
    if ALLOW_LIVE_TRADING:
        print("  WARNING: live-trading opt-in is set in the environment")


def enabled_strategies():
    return ["stable", "risky1"] + (["risky2"] if FEATURE_FLAGS["risky2_enabled"] else [])


def _default_alert(**kw):
    try:
        from data.discord_notifier import send_alert
        send_alert(**kw)
    except Exception as e:
        print(f"[run] alert failed: {e}")


def storage_and_schema_preflight(db_path=None):
    """Explicit storage + schema check. Returns a list of fatal problems (empty = OK)."""
    from core import logger
    from core.storage import storage_problems
    db_path = db_path or logger.DB_PATH
    problems = storage_problems(db_path, need_write=True)
    if problems:
        return problems
    try:
        logger.init_db(db_path)
    except Exception as e:
        return [f"database schema could not be prepared: {type(e).__name__}: {e}"]
    return []


class Supervisor:
    """Runs the strategy threads and watches them. Everything external is injectable for tests."""

    def __init__(self, targets, db_path=None, alert=_default_alert, stop_event=None,
                 interval=WATCHDOG_INTERVAL_SECONDS, hang_seconds=WATCHDOG_HANG_SECONDS,
                 grace_seconds=SHUTDOWN_GRACE_SECONDS, safe_after=SAFE_MODE_AFTER_UNCLEAN_EXITS,
                 max_restarts=MAX_WATCHDOG_RESTARTS, clock=time.monotonic, on_stop=None):
        self.targets = dict(targets)          # name -> callable(stop_event)
        self.db_path = db_path
        self.alert = alert
        self.stop_event = stop_event or threading.Event()
        self.interval = interval
        self.hang_seconds = hang_seconds
        self.grace_seconds = grace_seconds
        self.safe_after = safe_after
        self.max_restarts = max_restarts
        self.clock = clock
        self.on_stop = on_stop or (lambda: None)
        self.threads = {}
        self.run_id = None
        self.unclean_at_start = 0
        self.reported = set()
        self.exit_code = None

    # -- startup -----------------------------------------------------------
    def begin(self):
        from core import lifecycle
        self.run_id, self.unclean_at_start = lifecycle.record_start(self.db_path)
        print(f"[run] process run {self.run_id}: {self.unclean_at_start} unclean exit(s) in the crash-loop window")
        if self.unclean_at_start >= self.safe_after:
            reason = (f"{self.unclean_at_start} unclean exits within the crash-loop window "
                      f"(threshold {self.safe_after})")
            if lifecycle.set_safe_mode(reason, self.db_path):
                self.alert(bot_name="system", alert_type="CRASH LOOP: SAFE MODE",
                           message=f"{reason}. NEW ENTRIES ARE BLOCKED for every strategy; existing positions "
                                   "are still risk-managed. Investigate, then run `python -m tools.safe_mode "
                                   "clear --note ... --confirm`.", portfolio_value=0.0)
        safe = lifecycle.get_safe_mode(self.db_path)
        if safe:
            print(f"[run] SAFE MODE active since {safe.get('since')}: {safe.get('reason')} (no new entries)")
        self._started_at = self.clock()
        for name, target in self.targets.items():
            t = threading.Thread(target=target, args=(self.stop_event,), daemon=True, name=name)
            self.threads[name] = t
            t.start()
            print(f"[run] {name} strategy thread started")

    # -- watchdog ------------------------------------------------------------
    def _heartbeat_age(self, name):
        from core.logger import get_heartbeat_full
        try:
            hb = get_heartbeat_full(name, self.db_path)
        except Exception:
            return None
        if not hb:
            return None
        seen = datetime.fromisoformat(hb["last_seen"]).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - seen).total_seconds()

    def check(self):
        """
        One watchdog pass. Returns (dead, hung) strategy names. Degraded but
        alive strategies (fresh heartbeat) are never reported here.
        """
        dead = [n for n, t in self.threads.items() if not t.is_alive()]
        hung = []
        up_for = self.clock() - self._started_at
        for n, t in self.threads.items():
            if n in dead or up_for < self.hang_seconds:
                continue
            age = self._heartbeat_age(n)
            if age is None or age > self.hang_seconds:
                hung.append(n)
        return dead, hung

    def handle_failure(self, dead, hung):
        """Exit for a restart while the restart budget lasts; afterwards stay up and keep alerting once."""
        from core import lifecycle
        failed = sorted(set(dead) | set(hung))
        unclean = lifecycle.unclean_exits(self.db_path)
        what = ", ".join(f"{n} ({'died' if n in dead else 'hung'})" for n in failed)
        if unclean < self.max_restarts:
            self.alert(bot_name="system", alert_type="STRATEGY THREAD FAILED",
                       message=f"{what}. Its positions are NOT being managed. Restarting the process "
                               f"({unclean + 1}/{self.max_restarts} unclean exits in the window); state is "
                               "reconciled at startup and entries stay blocked until it is.", portfolio_value=0.0)
            self.exit_code = EXIT_WATCHDOG
            self.stop_event.set()
            return True
        new = [n for n in failed if n not in self.reported]
        if new:
            self.reported.update(new)
            self.alert(bot_name="system", alert_type="STRATEGY THREAD FAILED - NOT RESTARTING",
                       message=f"{what}. Restart budget exhausted ({unclean} unclean exits in the window): the "
                               "process stays up so the healthy strategies keep running, but the failed "
                               "strategy's positions are UNMANAGED. Manual intervention required.",
                       portfolio_value=0.0)
        return False

    def watch(self):
        """Blocks until a stop is requested or a failure requires a restart."""
        while not self.stop_event.wait(self.interval):
            dead, hung = self.check()
            if (dead or hung) and self.handle_failure(dead, hung):
                break

    # -- shutdown -------------------------------------------------------------
    def shutdown(self, reason):
        from core import lifecycle
        self.stop_event.set()
        deadline = self.clock() + self.grace_seconds
        for t in self.threads.values():
            t.join(timeout=max(0.0, deadline - self.clock()))
        still = [n for n, t in self.threads.items() if t.is_alive()]
        if still:
            print(f"[run] threads still running after {self.grace_seconds}s grace: {still}")
        try:
            self.on_stop()
        except Exception as e:
            print(f"[run] on_stop failed: {e}")
        if self.run_id is not None:
            lifecycle.record_end(self.run_id, reason, self.db_path)
        print(f"[run] shutdown complete ({reason})")


def install_signal_handlers(stop_event):
    def handler(signum, frame):
        print(f"[run] received signal {signum}; stopping gracefully")
        stop_event.set()
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _fatal(problems, stop_event, alert=_default_alert):
    for p in problems:
        print(f"[run] FATAL: {p}")
    alert(bot_name="system", alert_type="STARTUP FAILED",
          message="SinghQuant cannot start:\n" + "\n".join(problems), portfolio_value=0.0)
    # Wait before exiting so a restart policy cannot spin; SIGTERM still stops it at once.
    stop_event.wait(FATAL_STARTUP_EXIT_DELAY_SECONDS)
    return EXIT_FATAL_STARTUP


def main():
    print("=" * 50)
    print(" SinghQuant Trading System Starting....")
    print("=" * 50)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    preflight()
    problems = storage_and_schema_preflight()
    if problems:
        return _fatal(problems, stop_event)

    from models.retrain import start_scheduler
    from strategies.common import run_forever
    contexts = {}
    from strategies.stable import build_context as stable_ctx
    from strategies.risky1 import build_context as risky1_ctx
    contexts["stable"] = stable_ctx
    contexts["risky1"] = risky1_ctx
    if FEATURE_FLAGS["risky2_enabled"]:
        from strategies.risky2 import build_context as risky2_ctx
        contexts["risky2"] = risky2_ctx
    else:
        print("risky2 RL bot is disabled in config.py (FEATURE_FLAGS); enable it once models are trained")

    targets = {name: (lambda stop, build=build: run_forever(build(), stop_event=stop))
               for name, build in contexts.items()}
    scheduler = start_scheduler()
    sup = Supervisor(targets, stop_event=stop_event, on_stop=lambda: scheduler.shutdown(wait=False))
    sup.begin()
    sup.watch()
    if sup.exit_code == EXIT_WATCHDOG:
        sup.shutdown("watchdog_exit")
        return EXIT_WATCHDOG
    sup.shutdown("clean")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
