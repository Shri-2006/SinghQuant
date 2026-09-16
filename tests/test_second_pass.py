"""
Second-pass adversarial regressions: each test targets a defect found by
reviewing the FIRST-PASS fixes (ENGINEERING_AUDIT.md section 11).

S-01 orphan order (crash / timeout after the broker accepted) is recovered by client id
S-02 Polygon rate-limit spacing was lost when frames were fetched back to back
S-03 a strategy could sell another strategy's shares when the account held less than the sum of ledgers
S-04 reconciliation could raise forever, and a broker-unknown order blocked a symbol forever
S-05 fill booked and order row updated in separate transactions; fills without a price marked as applied
S-06 unsellable dust remainders were resubmitted and rejected every cycle
S-07 rejected entries were retried every cycle on the same bar
S-08 positions in symbols removed from the config were never marked or exited
S-09 a halted strategy never retried closing positions whose sell had failed
S-10 three strategy threads writing the same SQLite file concurrently
"""
import itertools
import sqlite3
import threading

import pytest

from core import portfolio
from core.execution import submit_and_track, close_strategy_position, reconcile_open_orders, has_open_order
from core.logger import get_trades_full, db_connection, utc_now_iso
from strategies.common import Decision, run_cycle, evaluate_risk, fire_kill_switch
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca, FakeAPIError
from tests.test_engine import make_ctx, always

FAST = dict(sleep_fn=lambda s: None)


def fast_clock():
    c = itertools.count(0, 30)
    return lambda: next(c)


# S-01 ----------------------------------------------------------------------

def test_s01_submit_exception_after_broker_accepted_is_recovered_not_duplicated(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="accept_then_raise")
    res = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    assert res.status == "filled" and res.filled_qty == pytest.approx(0.4)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    assert len(api.orders) == 1
    row = get_trades_full("stable", db_path)[-1]
    assert row["status"] == "filled" and row["order_id"] == list(api.orders)[0]


def test_s01_crash_between_submit_and_recording_broker_id_is_reconciled_on_restart(db_path):
    """Order row left in 'submitting' with a client id but no broker id."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    cid = "sq-stable-crash-test"
    row_id = portfolio.record_order("stable", "SPY", "buy", 0.4, 500.0, "submitting", reason="x",
                                    client_order_id=cid, db_path=db_path)
    api.submit_order("SPY", 0.4, "buy", client_order_id=cid)      # the broker got it and filled it
    assert portfolio.open_orders("stable", db_path=db_path)[0]["status"] == "submitting"   # first-pass ignored this state
    still = reconcile_open_orders(api, "stable", db_path)         # recovered by client id, fill booked, now terminal
    assert still == [] and not has_open_order(api, "stable", "SPY", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    with db_connection(db_path) as conn:
        status, fq, bid = conn.execute("SELECT status, filled_qty, broker_order_id FROM orders WHERE id = ?", (row_id,)).fetchone()
    assert status == "filled" and fq == pytest.approx(0.4) and bid == "ord-1"
    # and a fresh BUY signal does not double the position
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "at-max"
    assert len(api.orders) == 1


# S-02 ----------------------------------------------------------------------

def test_s02_polygon_spacing_is_kept_between_fetches(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0, "QQQ": 400.0, "DIA": 300.0})
    frames = {t: make_featured_df(price=p) for t, p in [("SPY", 500.0), ("QQQ", 400.0), ("DIA", 300.0)]}
    fetch_times, sleeps = [], []
    clock = itertools.count()
    ctx = make_ctx("stable", api, frames, decide=always("HOLD"), db_path=db_path, per_ticker_sleep=20)
    ctx.fetch_frame = lambda t: (fetch_times.append(next(clock)), frames[t])[1]
    run_cycle(ctx, api, sleep_fn=lambda s: (sleeps.append(s), [next(clock) for _ in range(int(s))]))
    assert sleeps == [20, 20]                       # one gap between each pair of fetches, none after the last
    assert all(b - a > 20 for a, b in zip(fetch_times, fetch_times[1:]))


# S-03 ----------------------------------------------------------------------

def test_s03_account_short_of_sum_of_ledgers_never_sells_another_strategys_shares(db_path):
    api = FakeAlpaca(prices={"TSLA": 345.82})
    submit_and_track(api, "stable", "TSLA", "buy", 50.0, 345.82, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "TSLA", "buy", 0.1446, 345.82, "x", db_path=db_path, **FAST)
    api.positions["TSLA"]["qty"] = 50.0             # risky1's 0.1446 sold outside the system
    res = close_strategy_position(api, "risky1", "TSLA", 345.82, "Momentum turned negative", db_path=db_path, **FAST)
    assert res.status == "reconciled"               # first-pass code would have sold 0.1446 of stable's shares
    assert [o for o in api.submitted if o["side"] == "sell"] == []
    assert api.positions["TSLA"]["qty"] == pytest.approx(50.0)
    assert portfolio.get_position_qty("risky1", "TSLA", db_path) == 0.0
    assert portfolio.get_position_qty("stable", "TSLA", db_path) == pytest.approx(50.0)
    res = close_strategy_position(api, "stable", "TSLA", 345.82, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "filled" and res.filled_qty == pytest.approx(50.0)


def test_s03_partial_shortfall_sells_only_what_is_left_after_others_claims(db_path):
    api = FakeAlpaca(prices={"AAPL": 200.0})
    submit_and_track(api, "stable", "AAPL", "buy", 1.0, 200.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "AAPL", "buy", 0.5, 200.0, "x", db_path=db_path, **FAST)
    api.positions["AAPL"]["qty"] = 1.2              # 0.3 vanished; stable's claim of 1.0 is honoured first
    res = close_strategy_position(api, "risky1", "AAPL", 200.0, "x", db_path=db_path, **FAST)
    assert res.filled_qty == pytest.approx(0.2)
    assert portfolio.get_position_qty("risky1", "AAPL", db_path) == 0.0
    assert api.positions["AAPL"]["qty"] == pytest.approx(1.0)
    rows = [r for r in get_trades_full("risky1", db_path) if r["action"] == "RECONCILE"]
    assert len(rows) == 1 and rows[0]["quantity"] == pytest.approx(0.3)


# S-04 ----------------------------------------------------------------------

def test_s04_oversized_sell_fill_does_not_crash_reconciliation(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                     sleep_fn=lambda s: None, clock_fn=fast_clock())
    api.fill_pending("ord-1")
    reconcile_open_orders(api, "stable", db_path)
    # a pending sell whose broker fill (corrupt) exceeds the ledger
    sres = submit_and_track(api, "stable", "SPY", "sell", 0.4, 500.0, "x", db_path=db_path,
                            sleep_fn=lambda s: None, clock_fn=fast_clock())
    api.orders[sres.order_id]["filled_qty"] = 5.0; api.orders[sres.order_id]["filled_avg_price"] = 500.0
    api.orders[sres.order_id]["status"] = "filled"
    still = reconcile_open_orders(api, "stable", db_path)      # must not raise
    assert still == []
    assert get_trades_full("stable", db_path)[-1]["status"] == "reconcile_error"
    # the strategy keeps running afterwards
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("HOLD"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] in ("hold", "at-max")


def test_s04_order_unknown_at_broker_stops_blocking_the_symbol(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    first = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                             sleep_fn=lambda s: None, clock_fn=fast_clock())
    api.forget_order(first.order_id)                              # paper account reset
    assert not has_open_order(api, "stable", "SPY", db_path)
    api.fill_mode = "fill"
    assert submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST).status == "filled"


def test_s04_unreachable_broker_blocks_until_the_order_is_stale(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    first = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                             sleep_fn=lambda s: None, clock_fn=fast_clock())
    api.fail_get_order = FakeAPIError("503 service unavailable", 503)
    assert has_open_order(api, "stable", "SPY", db_path)          # conservative while the broker is down
    with db_connection(db_path) as conn:                            # age the order past the limit
        conn.execute("UPDATE orders SET submitted_at = '2020-01-01T00:00:00' WHERE broker_order_id = ?", (first.order_id,))
    assert not has_open_order(api, "stable", "SPY", db_path)
    assert get_trades_full("stable", db_path)[-1]["status"] == "unresolved_stale"


# S-05 ----------------------------------------------------------------------

def test_s05_fill_without_price_is_not_marked_applied(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="pending")
    res = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                           sleep_fn=lambda s: None, clock_fn=fast_clock())
    api.fill_pending(res.order_id)
    api.orders[res.order_id]["filled_avg_price"] = None            # broker reports quantity but no price yet
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    with db_connection(db_path) as conn:
        fq = conn.execute("SELECT filled_qty FROM orders WHERE broker_order_id = ?", (res.order_id,)).fetchone()[0]
    assert fq == 0.0                                               # first-pass code recorded 0.4 as applied
    api.orders[res.order_id]["filled_avg_price"] = 500.0
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    reconcile_open_orders(api, "stable", db_path)                  # idempotent
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)


def test_s05_fill_and_order_row_are_updated_atomically(db_path):
    """A failure inside the same transaction leaves neither the ledger nor the order row changed."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    row_id = portfolio.record_order("stable", "SPY", "buy", 0.4, 500.0, "new", broker_order_id="x", db_path=db_path)
    with pytest.raises(sqlite3.Error):
        # an invalid order id type for the same-transaction update forces a failure after the ledger write
        portfolio.apply_fill("stable", "SPY", "buy", 0.4, 500.0, db_path=db_path,
                             order_update=(object(), "filled", 0.4, 500.0))
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == 1000.0


# S-06 ----------------------------------------------------------------------

def test_s06_dust_remainder_is_not_resubmitted_every_cycle(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    api.seed_position("SPY", 0.001, 500.0)
    portfolio.adopt_position("stable", "SPY", 0.001, 500.0, db_path)   # $0.50 of dust
    res = close_strategy_position(api, "stable", "SPY", 500.0, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "dust" and api.submitted == []
    # third pass T-01: dust is NOT written off (the shares were not sold); it stays owned
    # and no cash is credited. Exits simply skip it instead of resubmitting every cycle.
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.001)


# S-07 ----------------------------------------------------------------------

def test_s07_rejected_entry_is_not_retried_on_the_same_bar(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="reject")
    frames = {"SPY": make_featured_df(price=500.0, start="2026-08-01")}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-rejected"
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "entry-anti-churn"
    assert len(api.submitted) == 1
    frames["SPY"] = make_featured_df(price=500.0, start="2026-08-02")   # new bar: try again
    api.fill_mode = "fill"
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-filled"


# S-08 ----------------------------------------------------------------------

def test_s08_position_in_symbol_removed_from_config_is_still_stopped_out(db_path):
    api = FakeAlpaca(prices={"TSLA": 345.82, "NVDA": 100.0})
    submit_and_track(api, "risky1", "TSLA", "buy", 0.1446, 345.82, "x", db_path=db_path, **FAST)   # fills at 345.82
    api.set_price("TSLA", 300.0)
    frames = {"NVDA": make_featured_df(price=100.0), "TSLA": make_featured_df(price=300.0)}   # -13% vs entry
    ctx = make_ctx("risky1", api, {"NVDA": frames["NVDA"]}, decide=always("BUY"), db_path=db_path)
    ctx.fetch_frame = lambda t: frames[t]
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["TSLA"] == "emergency-exit"              # managed although not in ctx.assets
    assert portfolio.get_position_qty("risky1", "TSLA", db_path) == 0.0
    # and it is never re-entered because it is not a configured asset
    api.set_price("TSLA", 400.0); frames["TSLA"] = make_featured_df(price=400.0)
    portfolio.adopt_position("risky1", "TSLA", 0.01, 400.0, db_path)
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["TSLA"] in ("entry-blocked", "at-max", "hold")
    assert not any(o["symbol"] == "TSLA" and o["side"] == "buy" for o in api.submitted[1:])


# S-09 ----------------------------------------------------------------------

def test_s09_halted_strategy_keeps_trying_to_flatten_leftover_positions(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    evaluate_risk(ctx, {"SPY": 100.0}, verify_mark=lambda s: 100.0)
    risk = evaluate_risk(ctx, {"SPY": 100.0}, verify_mark=lambda s: 100.0)
    api.fill_mode = "reject"                                        # the emergency sell fails
    fire_kill_switch(ctx, api, risk, {"SPY": 100.0})
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    api.fill_mode = "fill"; api.set_price("SPY", 100.0)
    ctx.fetch_frame = lambda t: make_featured_df(price=100.0)
    out = run_cycle(ctx, api, **FAST)
    assert out["halted"] is True and out["halted_liquidation"] == [("SPY", "filled")]
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    assert api.close_all_calls == 0


# S-10 ----------------------------------------------------------------------

def test_s10_three_threads_write_the_ledger_concurrently(db_path):
    errors = []
    def worker(strategy, symbol, n):
        try:
            for _ in range(n):
                portfolio.apply_fill(strategy, symbol, "buy", 0.01, 100.0, db_path=db_path)
                portfolio.set_signal_state(strategy, symbol, "bar", "BUY", db_path=db_path)
                portfolio.get_strategy_state(strategy, db_path)
        except Exception as e:  # pragma: no cover
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(s, sym, 25))
               for s, sym in [("stable", "SPY"), ("risky1", "NVDA"), ("risky2", "X:BTCUSD")]]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    for s, sym in [("stable", "SPY"), ("risky1", "NVDA"), ("risky2", "X:BTCUSD")]:
        assert portfolio.get_position_qty(s, sym, db_path) == pytest.approx(0.25)
        assert portfolio.get_strategy_state(s, db_path)["cash_budget"] == pytest.approx(portfolio.CAPITAL[s] - 25.0)
