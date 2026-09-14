"""
Regression tests for core/execution.py and core/portfolio.py.

Invariants covered (numbering follows the audit brief):
  1  a strategy cannot sell more than it owns
  2  one strategy cannot liquidate another strategy's position
  3  quantity units are shares on BUY and SELL rows alike
  4  repeated identical signals cannot multiply exposure (see test_engine.py)
  5  pending orders prevent duplicate submissions
  9  strategy-level and account-level accounting are not substituted
 10  restart does not duplicate orders
 11  partial fills reconcile correctly
 12  rejected orders do not create false local positions
"""
import itertools
import math

import pytest

from core import portfolio
from core.execution import (submit_and_track, close_strategy_position, close_all_strategy_positions,
                            reconcile_open_orders, has_open_order)
from core.logger import get_trades_full
from tests.fakes import FakeAlpaca

FAST = dict(sleep_fn=lambda s: None)


def fast_clock():
    c = itertools.count(0, 30)
    return lambda: next(c)


def test_buy_fill_updates_ledger_and_logs_shares(db_path):
    api = FakeAlpaca(prices={"TSLA": 354.25})
    res = submit_and_track(api, "risky1", "TSLA", "buy", 0.1446, 345.82, "ML BUY", db_path=db_path, **FAST)
    assert res.status == "filled" and res.filled_qty == pytest.approx(0.1446)
    qty, avg, opened = portfolio.get_position("risky1", "TSLA", db_path)
    assert qty == pytest.approx(0.1446) and avg == pytest.approx(354.25) and opened
    state = portfolio.get_strategy_state("risky1", db_path)
    assert state["cash_budget"] == pytest.approx(200 - 0.1446 * 354.25)
    row = get_trades_full("risky1", db_path)[-1]
    assert row["action"] == "BUY" and row["quantity"] == pytest.approx(0.1446)
    assert row["price"] == pytest.approx(345.82)            # signal price
    assert row["filled_avg_price"] == pytest.approx(354.25)  # real fill price, distinct from signal price
    assert row["status"] == "filled" and row["order_id"]


def test_sell_logs_shares_not_dollars(db_path):
    """Investigation 2: the SELL row must log 0.1446 shares, not $51.2 of market value."""
    api = FakeAlpaca(prices={"TSLA": 354.25})
    submit_and_track(api, "risky1", "TSLA", "buy", 0.1446, 345.82, "ML BUY", db_path=db_path, **FAST)
    res = close_strategy_position(api, "risky1", "TSLA", 345.82, "Momentum turned negative", db_path=db_path, **FAST)
    assert res.status == "filled"
    rows = get_trades_full("risky1", db_path)
    buy, sell = rows[-2], rows[-1]
    assert buy["quantity"] == pytest.approx(sell["quantity"]) == pytest.approx(0.1446)
    assert sell["pnl"] is not None  # momentum exit now records realized pnl
    assert portfolio.get_position_qty("risky1", "TSLA", db_path) == 0.0


def test_strategy_cannot_sell_more_than_it_owns(db_path):
    api = FakeAlpaca(prices={"AAPL": 200.0})
    api.seed_position("AAPL", 50.0, 150.0)  # account holds 50 (someone else's)
    res = submit_and_track(api, "risky1", "AAPL", "sell", 50.0, 200.0, "x", db_path=db_path, **FAST)
    assert res.status == "refused" and not api.submitted
    submit_and_track(api, "risky1", "AAPL", "buy", 0.25, 200.0, "x", db_path=db_path, **FAST)
    res = submit_and_track(api, "risky1", "AAPL", "sell", 50.0, 200.0, "x", db_path=db_path, **FAST)
    assert res.requested_qty == pytest.approx(0.25)          # clamped to owned qty
    assert api.submitted[-1]["qty"] == pytest.approx(0.25)
    assert api.positions["AAPL"]["qty"] == pytest.approx(50.0)  # the other 50 shares untouched
    with pytest.raises(ValueError):
        portfolio.apply_fill("risky1", "AAPL", "sell", 1.0, 200.0, db_path)


def test_one_strategy_cannot_liquidate_anothers_position(db_path):
    """Investigation 3 scenario: stable owns ~50 TSLA, risky1 owns 0.1446, risky1 sells."""
    api = FakeAlpaca(prices={"TSLA": 345.82})
    submit_and_track(api, "stable", "TSLA", "buy", 50.0, 345.82, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "TSLA", "buy", 0.1446, 345.82, "x", db_path=db_path, **FAST)
    assert api.positions["TSLA"]["qty"] == pytest.approx(50.1446)
    res = close_strategy_position(api, "risky1", "TSLA", 345.82, "Momentum turned negative", db_path=db_path, **FAST)
    assert res.filled_qty == pytest.approx(0.1446)
    assert api.positions["TSLA"]["qty"] == pytest.approx(50.0)
    assert portfolio.get_position_qty("stable", "TSLA", db_path) == pytest.approx(50.0)
    assert portfolio.get_position_qty("risky1", "TSLA", db_path) == 0.0
    assert api.closed_symbols == []  # account-wide close_position never used


def test_close_all_only_touches_own_positions(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0, "NVDA": 100.0, "BTCUSD": 60000.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "NVDA", "buy", 0.5, 100.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky2", "X:BTCUSD", "buy", 0.0008, 60000.0, "x", time_in_force="gtc", db_path=db_path, **FAST)
    results = close_all_strategy_positions(api, "risky1", {"NVDA": 100.0}, "KILL_SWITCH", db_path=db_path, **FAST)
    assert [r.symbol for r in results] == ["NVDA"]
    assert api.close_all_calls == 0
    assert api.positions["SPY"]["qty"] == pytest.approx(0.4)
    assert api.positions["BTCUSD"]["qty"] == pytest.approx(0.0008)
    assert "NVDA" not in api.positions


def test_rejected_order_creates_no_position(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="reject")
    res = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    assert res.status == "rejected"
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == 1000.0
    row = get_trades_full("stable", db_path)[-1]
    assert row["status"] == "rejected" and row["filled_qty"] == 0.0


def test_partial_fill_applies_only_filled_quantity_then_reconciles(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="partial", partial_fraction=0.5)
    res = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                           sleep_fn=lambda s: None, clock_fn=fast_clock())
    assert res.status == "partially_filled" and res.filled_qty == pytest.approx(0.2)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.2)
    assert has_open_order(api, "stable", "SPY", db_path)
    api.fill_pending(res.order_id)  # the rest fills later
    still_open = reconcile_open_orders(api, "stable", db_path)
    assert still_open == []
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    # reconciling again must not double-apply
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)


def test_pending_order_blocks_duplicate_submission(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    first = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                             sleep_fn=lambda s: None, clock_fn=fast_clock())
    assert first.status == "new"
    second = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                              sleep_fn=lambda s: None, clock_fn=fast_clock())
    assert second.status == "refused" and "open" in second.message
    assert len(api.submitted) == 1


def test_restart_does_not_duplicate_and_late_fill_is_applied_once(db_path):
    """Invariant 10. Simulates: submit, process dies, fill happens, process restarts."""
    from strategies.common import StrategyContext, Decision, run_cycle
    from tests.conftest import make_featured_df
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    first = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                             sleep_fn=lambda s: None, clock_fn=fast_clock())
    assert first.status == "new" and portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    api.fill_pending(first.order_id)             # fills while the process is "down"
    api.fill_mode = "fill"
    # "restart": a brand-new engine context with the same database and the same BUY signal
    ctx = StrategyContext(name="stable", assets=["SPY"], decide=lambda *a: Decision("BUY"),
                          fetch_frame=lambda t: make_featured_df(price=500.0), uses_market_hours=False,
                          per_ticker_sleep=0, cycle_sleep=0, regime_ok=lambda df, s, v: True,
                          db_path=db_path, exec_kw=dict(sleep_fn=lambda s: None))
    out = run_cycle(ctx, api, sleep_fn=lambda s: None)
    assert out["tickers"]["SPY"] == "at-max"      # reconciled fill applied once, so already at max: no new order
    assert len(api.submitted) == 1
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    assert portfolio.open_orders("stable", db_path=db_path) == []


def test_invalid_quantities_are_refused(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    for bad in (0, -1, float("nan"), float("inf"), "abc"):
        res = submit_and_track(api, "stable", "SPY", "buy", bad, 500.0, "x", db_path=db_path, **FAST)
        assert res.status == "refused"
    assert not api.submitted


def test_strategy_equity_is_independent_of_account_equity(db_path):
    """Invariant 9: another strategy's losses (or a broken account number) do not change this strategy's equity."""
    api = FakeAlpaca(prices={"SPY": 500.0, "NVDA": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "NVDA", "buy", 0.5, 100.0, "x", db_path=db_path, **FAST)
    api.set_price("NVDA", 1.0)                      # risky1 blows up
    api.account_overrides["portfolio_value"] = 5.0  # and the account number is garbage
    eq, val, missing = portfolio.strategy_equity("stable", {"SPY": 500.0}, db_path)
    assert eq == pytest.approx(1000.0) and missing == []
    eq_r1, _, _ = portfolio.strategy_equity("risky1", {"NVDA": 1.0}, db_path)
    assert eq_r1 == pytest.approx(200 - 50 + 0.5)


def test_missing_mark_uses_entry_price_not_zero(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    eq, val, missing = portfolio.strategy_equity("stable", {}, db_path)
    assert missing == ["SPY"] and eq == pytest.approx(1000.0)


def test_ledger_drift_written_off_when_broker_has_nothing(db_path):
    """A legacy close_all_positions() sold our shares outside the ledger."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    api.positions.clear()  # something else liquidated the account
    res = close_strategy_position(api, "stable", "SPY", 500.0, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "reconciled"
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    assert get_trades_full("stable", db_path)[-1]["action"] == "RECONCILE"
