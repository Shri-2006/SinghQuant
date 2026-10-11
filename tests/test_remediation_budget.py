"""
Phase 1 remediation R-3: strict per-strategy budget enforcement.

Audit finding H-1: sizing never consulted the strategy's cash budget, so
stable could deploy $1,600 of a $1,000 budget (8 symbols x $200), and the
account-cash guard main had for risky2 (commit 523f7f6) had been dropped.

    available = cash_budget - notional of this strategy's open buy orders

The engine sizes inside `available`; execution re-checks it and records the
reserving order row in one BEGIN IMMEDIATE transaction (thread- and
process-safe). Risk thresholds and MAX_POSITION_SIZE are unchanged.
"""
import itertools
import os
import subprocess
import sys
import threading

import pytest

from core import portfolio
from core.config import CAPITAL, MAX_POSITION_SIZE
from core.execution import submit_and_track, reconcile_open_orders
from strategies.common import run_cycle
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca
from tests.test_engine import make_ctx, always

_clock = itertools.count(0, 30)    # fake monotonic clock: a pending order times out after one poll
FAST = dict(sleep_fn=lambda s: None)                                   # run_cycle
EXEC = dict(sleep_fn=lambda s: None, clock_fn=lambda: next(_clock))  # direct execution calls
STABLE8 = ["SPY", "QQQ", "AAPL", "MSFT", "DIA", "IWM", "SAP", "AMZN"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _invested(strategy, db_path):
    return sum(q * a for q, a, _ in portfolio.list_positions(strategy, db_path).values())


# ---------------------------------------------------------------------------
# Engine sizing
# ---------------------------------------------------------------------------

def test_stable_cannot_deploy_more_than_its_budget(db_path):
    """Audit probe P1: 8 BUY signals x $200 against a $1,000 budget (was: cash_budget -600)."""
    api = FakeAlpaca(cash=100000, prices={s: 100.0 for s in STABLE8})
    frames = {s: make_featured_df(price=100.0, atr=1.5) for s in STABLE8}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)["tickers"]
    assert list(out.values()) == ["buy-filled"] * 5 + ["budget-exhausted"] * 3
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == pytest.approx(0.0, abs=1e-6)
    assert _invested("stable", db_path) == pytest.approx(CAPITAL["stable"])
    # next cycle: still nothing more is bought
    out = run_cycle(ctx, api, **FAST)["tickers"]
    assert not any(v.startswith("buy") for v in out.values())
    assert len(api.submitted) == 5


def test_risky2_cannot_deploy_more_than_its_budget(db_path):
    syms = ["X:BTCUSD", "X:ETHUSD", "X:SOLUSD", "X:AVAXUSD", "X:LINKUSD", "X:ADAUSD", "X:XRPUSD", "X:DOGEUSD"]
    api = FakeAlpaca(cash=100000, prices={s[2:]: 10.0 for s in syms})
    frames = {s: make_featured_df(price=10.0, atr=0.3) for s in syms}
    ctx = make_ctx("risky2", api, frames, decide=always("BUY"), db_path=db_path, check_account_cash=True)
    out = run_cycle(ctx, api, **FAST)["tickers"]
    assert sum(v == "buy-filled" for v in out.values()) == int(CAPITAL["risky2"] // MAX_POSITION_SIZE["risky2"])
    assert portfolio.get_strategy_state("risky2", db_path)["cash_budget"] >= -1e-6


def test_partial_budget_sizes_the_last_order_down_instead_of_skipping(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    portfolio.update_strategy_state("stable", db_path, cash_budget=150.0, peak_equity=150.0, last_equity=150.0)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-filled"
    assert api.submitted[-1]["qty"] == pytest.approx(1.5)          # $150, not the $200 target


def test_rounding_never_exceeds_the_budget(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 6.0})
    portfolio.update_strategy_state("stable", db_path, cash_budget=100.0, peak_equity=100.0, last_equity=100.0)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=6.0, atr=0.09)}, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-filled"     # round(100/6, 4) * 6 = 100.0002
    assert api.submitted[-1]["qty"] * 6.0 <= 100.0
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] >= 0.0


# ---------------------------------------------------------------------------
# Reservations (open orders count against the budget)
# ---------------------------------------------------------------------------

def test_open_buy_order_reserves_budget_until_it_fills_or_ends(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0}, fill_mode="pending")
    res = submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "new"
    assert portfolio.available_budget("stable", db_path) == pytest.approx(800.0)
    refused = submit_and_track(api, "stable", "QQQ", "buy", 8.5, 100.0, "x", db_path=db_path, **EXEC)
    assert refused.status == "refused" and "budget" in refused.message
    assert [o["symbol"] for o in api.submitted] == ["SPY"]                     # never reached the broker
    api.fill_pending(res.order_id, fraction=0.5)                               # partial fill: cash moves, reservation shrinks
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == pytest.approx(900.0)
    assert portfolio.available_budget("stable", db_path) == pytest.approx(800.0)
    api.cancel_pending(res.order_id)                                           # rest cancelled: reservation released
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.available_budget("stable", db_path) == pytest.approx(900.0)


def test_over_budget_submit_writes_nothing(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    res = submit_and_track(api, "stable", "SPY", "buy", 10.5, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "refused" and api.submitted == []
    assert portfolio.open_orders("stable", db_path=db_path) == []
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == pytest.approx(1000.0)


def test_budgets_are_per_strategy(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "NVDA": 100.0})
    assert submit_and_track(api, "stable", "SPY", "buy", 10.0, 100.0, "x", db_path=db_path, **EXEC).status == "filled"
    assert submit_and_track(api, "risky1", "NVDA", "buy", 2.0, 100.0, "x", db_path=db_path, **EXEC).status == "filled"
    assert submit_and_track(api, "risky1", "NVDA", "buy", 0.1, 100.0, "x", db_path=db_path, **EXEC).status == "refused"


# ---------------------------------------------------------------------------
# Concurrency: the check and the reservation are one atomic step
# ---------------------------------------------------------------------------

def test_concurrent_reservations_from_threads_never_overcommit(db_path):
    portfolio.ensure_strategy_state("stable", db_path)
    barrier = threading.Barrier(8)
    results = []

    def worker(i):
        barrier.wait()
        row, _ = portfolio.reserve_buy_order("stable", f"S{i}", 2.0, 100.0, client_order_id=f"t{i}", db_path=db_path)
        results.append(row)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r is not None for r in results) == 5                    # 5 x $200 = $1,000
    assert portfolio.available_budget("stable", db_path) == pytest.approx(0.0)


def test_concurrent_reservations_from_separate_processes_never_overcommit(db_path):
    """Two bot processes on one database (e.g. a second container started by mistake)."""
    portfolio.ensure_strategy_state("stable", db_path)
    code = r"""
import os, sys
sys.path.insert(0, os.getcwd())
from core import portfolio
db, tag = sys.argv[1], sys.argv[2]
ok = 0
for i in range(5):
    row, _ = portfolio.reserve_buy_order("stable", "SPY", 0.6, 100.0, client_order_id=f"{tag}-{i}", db_path=db)
    ok += row is not None
print(ok)
"""
    env = dict(os.environ, SINGHQUANT_DB_PATH=db_path)
    procs = [subprocess.Popen([sys.executable, "-c", code, db_path, f"p{n}"], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for n in range(4)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [o[1][-500:] for o in outs]
    successes = sum(int(o[0].strip().splitlines()[-1]) for o in outs)
    assert successes == 16                                             # floor($1,000 / $60); demand was $1,200
    assert portfolio.available_budget("stable", db_path) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Slippage and the restored account-cash guard
# ---------------------------------------------------------------------------

def test_slippage_overdraft_alerts_once_and_blocks_further_buys(db_path):
    api = FakeAlpaca(cash=100000, prices={s: 100.0 for s in STABLE8}, slippage=0.01)
    frames = {s: make_featured_df(price=100.0, atr=1.5) for s in STABLE8}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    sent = []
    ctx.alert = lambda **kw: sent.append(kw)
    run_cycle(ctx, api, **FAST)                                        # fills 1% above the signal price
    cash = portfolio.get_strategy_state("stable", db_path)["cash_budget"]
    assert -11.0 < cash < 0.0                                          # bounded by slippage x deployed
    out = run_cycle(ctx, api, **FAST)["tickers"]
    assert not any(v.startswith("buy") for v in out.values())
    run_cycle(ctx, api, **FAST)
    assert [a["alert_type"] for a in sent].count("BUDGET OVERDRAWN") == 1


def test_risky2_account_cash_guard_is_restored_and_fails_closed(db_path):
    api = FakeAlpaca(cash=20.0, prices={"BTCUSD": 60000.0})
    frames = {"X:BTCUSD": make_featured_df(price=60000.0, atr=1800.0)}
    ctx = make_ctx("risky2", api, frames, decide=always("BUY"), db_path=db_path, check_account_cash=True)
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "insufficient-account-cash"
    def broken():
        raise ConnectionError("account endpoint down")
    api.get_account = broken
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "account-cash-unknown"
    assert api.submitted == []


def test_risky2_context_enables_the_account_cash_guard():
    from strategies.risky2 import build_context
    assert build_context().check_account_cash is True
