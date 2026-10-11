"""
Phase 1 remediation R-4: ambiguous order outcomes are reconciled, never
assumed rejected; every non-final broker status stays open.

Audit finding H-2: when submit_order raised and the client-id lookup also
failed transiently, the order was marked "rejected" (terminal) although the
broker had filled it: orphaned shares, no stop loss, and a duplicate buy on
the next bar. Now only AUTHORITATIVE evidence (an HTTP/Alpaca 4xx refusal)
makes a submission "rejected"; anything else is "unresolved" and is looked up
by client order id every cycle.
"""
import itertools
from datetime import datetime, timedelta, timezone

import pytest

from core import portfolio
from core.execution import submit_and_track, reconcile_open_orders, has_open_order, _is_definitive_rejection
from core.logger import db_connection, get_trades_full
from strategies.common import run_cycle
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca, FakeAPIError
from tests.test_engine import make_ctx, always

_clock = itertools.count(0, 30)    # fake monotonic clock: a pending order times out after one poll
FAST = dict(sleep_fn=lambda s: None)                                   # run_cycle
EXEC = dict(sleep_fn=lambda s: None, clock_fn=lambda: next(_clock))  # direct execution calls


class AlpacaCodeError(Exception):
    def __init__(self, msg, code=None, status_code=None):
        super().__init__(msg)
        self.code = code
        self.status_code = status_code


class AmbiguousBroker(FakeAlpaca):
    """submit_order may record (and fill) the order and THEN raise; lookups can fail on demand."""
    def __init__(self, submit_exc, accept=True, lookup_exc=None, **kw):
        super().__init__(**kw)
        self.submit_exc, self.accept, self.lookup_exc = submit_exc, accept, lookup_exc

    def submit_order(self, symbol, qty, side, type="market", time_in_force="day", client_order_id=None):
        if self.submit_exc is None:
            return super().submit_order(symbol, qty, side, type, time_in_force, client_order_id)
        if self.accept:
            super().submit_order(symbol, qty, side, type, time_in_force, client_order_id)
        else:
            self.submitted.append({"symbol": symbol, "qty": float(qty), "side": side, "tif": time_in_force,
                                   "client_order_id": client_order_id})
        raise self.submit_exc

    def get_order_by_client_order_id(self, client_order_id):
        if self.lookup_exc is not None:
            raise self.lookup_exc
        return super().get_order_by_client_order_id(client_order_id)


def _order_row(db_path, strategy="stable"):
    with db_connection(db_path) as conn:
        return conn.execute("SELECT id, status, filled_qty, broker_order_id, client_order_id FROM orders "
                            "WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,)).fetchone()


# ---------------------------------------------------------------------------
# Audit probe P2
# ---------------------------------------------------------------------------

def test_timeout_plus_failed_lookup_is_unresolved_and_the_fill_is_booked_once(db_path):
    api = AmbiguousBroker(FakeAPIError("read timed out", None), accept=True,
                          lookup_exc=FakeAPIError("503 service unavailable", 503),
                          cash=100000, prices={"SPY": 100.0})
    res = submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "unresolved"                         # was: "rejected" -> orphaned share
    _, status, _, _, cid = _order_row(db_path)
    assert status == "unresolved" and cid.startswith("sq-stable-")
    assert has_open_order(api, "stable", "SPY", db_path)      # lookup still failing: stays open
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    api.lookup_exc = None                                     # broker reachable again
    assert reconcile_open_orders(api, "stable", db_path) == []
    reconcile_open_orders(api, "stable", db_path)             # idempotent
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(1.0)
    assert api.positions["SPY"]["qty"] == pytest.approx(1.0)
    assert len(portfolio.list_ledger_events("stable", "SPY", event_type="FILL", db_path=db_path)) == 1


def test_unresolved_entry_is_not_duplicated_on_later_cycles(db_path):
    api = AmbiguousBroker(FakeAPIError("connection aborted", None), accept=True,
                          lookup_exc=FakeAPIError("gateway timeout", 504),
                          cash=100000, prices={"SPY": 100.0})
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-unresolved"
    api.submit_exc = None
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-refused"         # open order blocks a duplicate
    ctx.fetch_frame = lambda t: make_featured_df(price=100.0, atr=1.5)             # even on a new bar
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-refused"
    assert len(api.submitted) == 1
    api.lookup_exc = None
    out = run_cycle(ctx, api, **FAST)["tickers"]["SPY"]
    assert out == "at-max" and len(api.submitted) == 1
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(2.0)   # $200 target, booked once


def test_unresolved_buy_keeps_its_budget_reservation(db_path):
    api = AmbiguousBroker(FakeAPIError("read timed out", None), accept=False,
                          lookup_exc=FakeAPIError("503", 503), cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 3.0, 100.0, "x", db_path=db_path, **EXEC)
    assert portfolio.available_budget("stable", db_path) == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# What counts as authoritative evidence of failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    FakeAPIError("order rejected by broker", 403),
    FakeAPIError("unknown symbol", 422),
    FakeAPIError("bad request", 400),
    AlpacaCodeError("insufficient buying power", code=40310000),     # Alpaca code, no HTTP response attached
])
def test_definitive_4xx_rejection_is_rejected(db_path, exc):
    api = AmbiguousBroker(exc, accept=False, cash=100000, prices={"SPY": 100.0})
    res = submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "rejected"
    assert _order_row(db_path)[1] == "rejected"
    assert portfolio.available_budget("stable", db_path) == pytest.approx(1000.0)   # reservation released


@pytest.mark.parametrize("exc", [
    FakeAPIError("read timed out", None),
    ConnectionError("connection reset by peer"),
    FakeAPIError("internal server error", 500),
    FakeAPIError("service unavailable", 503),
    FakeAPIError("too many requests", 429),
    FakeAPIError("request timeout", 408),
    FakeAPIError("client_order_id must be unique", 409),
])
@pytest.mark.parametrize("lookup_exc", [FakeAPIError("order not found", 404), FakeAPIError("bad gateway", 502)])
def test_ambiguous_submission_is_never_rejected_without_evidence(db_path, exc, lookup_exc):
    api = AmbiguousBroker(exc, accept=False, lookup_exc=lookup_exc, cash=100000, prices={"SPY": 100.0})
    res = submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "unresolved"
    assert _order_row(db_path)[1] == "unresolved"
    assert get_trades_full("stable", db_path)[-1]["status"] == "unresolved"


def test_definitive_rejection_classifier():
    assert _is_definitive_rejection(FakeAPIError("x", 403))
    assert _is_definitive_rejection(AlpacaCodeError("x", code=42210000))
    assert not _is_definitive_rejection(AlpacaCodeError("x", code=42910000))        # rate limit
    assert not _is_definitive_rejection(FakeAPIError("x", 500))
    assert not _is_definitive_rejection(TimeoutError("x"))


def test_never_acknowledged_order_becomes_unknown_only_after_the_grace_period(db_path):
    api = AmbiguousBroker(FakeAPIError("read timed out", None), accept=False,
                          cash=100000, prices={"SPY": 100.0})                    # broker never got it
    submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **EXEC)
    assert [o["status"] for o in reconcile_open_orders(api, "stable", db_path)] == ["unresolved"]
    assert portfolio.available_budget("stable", db_path) == pytest.approx(800.0)
    row_id = _order_row(db_path)[0]
    eleven_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=11)).replace(tzinfo=None).isoformat()
    with db_connection(db_path) as conn:                                          # 11 minutes later (Python's clock,
        conn.execute("UPDATE orders SET submitted_at = ? WHERE id = ?", (eleven_minutes_ago, row_id))  # like the code)
    assert reconcile_open_orders(api, "stable", db_path) == []
    assert _order_row(db_path)[1] == "unknown_at_broker"
    assert portfolio.available_budget("stable", db_path) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Expanded open statuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broker_status", ["pending_cancel", "pending_replace", "calculated", "done_for_day",
                                           "stopped", "suspended", "some_future_status"])
def test_non_final_broker_statuses_stay_open_and_late_fills_are_booked(db_path, broker_status):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, fill_mode="pending")
    res = submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    api.orders[res.order_id]["status"] = broker_status
    still = reconcile_open_orders(api, "stable", db_path)
    assert [o["status"] for o in still] == [broker_status]
    assert has_open_order(api, "stable", "SPY", db_path)
    api.fill_pending(res.order_id)                                   # the fill arrives later
    assert reconcile_open_orders(api, "stable", db_path) == []
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(1.0)


def test_fill_without_price_on_submit_stays_open_until_priced(db_path):
    class NoPriceYet(FakeAlpaca):
        hide_price = True
        def _order_ns(self, o):
            ns = super()._order_ns(o)
            if self.hide_price:
                ns.filled_avg_price = None
                ns._raw["filled_avg_price"] = None
            return ns
    api = NoPriceYet(cash=100000, prices={"SPY": 100.0})
    res = submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    assert res.status == "awaiting_fill_price" and res.filled_qty == 0.0
    assert _order_row(db_path)[1] == "awaiting_fill_price"           # was: terminal "filled", fill lost forever
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    api.hide_price = False
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(1.0)
    assert _order_row(db_path)[1] == "filled"
