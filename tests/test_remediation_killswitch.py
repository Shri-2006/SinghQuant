"""
Phase 1 remediation R-1: missing broker positions must never disable the kill
switch, and persistent suspect states must be visible.

Owner decisions encoded here:
  D1-B  a position that vanished from the broker is FLAGGED, never written off
        automatically: qty, cost, cash and ownership are preserved, the flagged
        quantity is valued conservatively, and a human resolves it with
        tools/reconcile_position.py.
  D2    a suspect reading (unknown broker state) only alerts; entries stay
        blocked and nothing is liquidated because of an API outage.
Risk thresholds are not changed by any of this.
"""
import sqlite3

import pytest

from core import portfolio
from core.config import SUSPECT_ALERT_AFTER_CYCLES, SUSPECT_ALERT_REPEAT_CYCLES
from core.execution import (BrokerQuote, broker_quote, submit_and_track, close_all_strategy_positions,
                            reconcile_open_orders)
from core.logger import get_trades_full, get_heartbeat, init_db, db_connection
from strategies.common import evaluate_risk, run_cycle, refresh_discrepancies, trade_ticker
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca, FakeAPIError
from tests.test_engine import make_ctx, always

FAST = dict(sleep_fn=lambda s: None)


class AlpacaCodeError(Exception):
    """Mimics alpaca_trade_api.rest.APIError: a JSON `code`, optionally no HTTP status."""
    def __init__(self, msg, code=None, status_code=None):
        super().__init__(msg)
        self.code = code
        self.status_code = status_code


def _two_positions(db_path, api):
    """stable owns 4 SPY and 4 QQQ at $100 (cash budget left: $200)."""
    for sym in ("SPY", "QQQ"):
        submit_and_track(api, "stable", sym, "buy", 4.0, 100.0, "x", db_path=db_path, **FAST)


def _alerts(ctx):
    sent = []
    ctx.alert = lambda **kw: sent.append(kw)
    return sent


# ---------------------------------------------------------------------------
# The audit defect C-1, reproduced and fixed
# ---------------------------------------------------------------------------

def test_vanished_position_no_longer_blocks_a_confirmed_kill_switch(db_path):
    """Audit probe P4: critical drawdown + one held symbol gone at the broker.
    Before: 'suspect' on every cycle forever, kill switch never fired, no alert."""
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    _two_positions(db_path, api)
    del api.positions["QQQ"]                         # paper reset / manual close in the Alpaca UI
    api.set_price("SPY", 60.0)                       # a real, broker-corroborated loss
    frames = {"SPY": make_featured_df(price=60.0, atr=0.9), "QQQ": make_featured_df(price=100.0, atr=1.5)}
    ctx = make_ctx("stable", api, frames, db_path=db_path)
    sent = _alerts(ctx)

    first = run_cycle(ctx, api, **FAST)
    assert first["risk"].level == "critical" and first["risk"].kill_now is False     # reading 1 of 2
    assert first["risk"].discrepancy_symbols == ["QQQ"]
    second = run_cycle(ctx, api, **FAST)
    assert second["killed"] is True                                                  # confirmed: fires
    assert portfolio.get_strategy_state("stable", db_path)["halted"] is True

    # D1-B: the vanished position is preserved and flagged, never written off
    assert portfolio.get_position_qty("stable", "QQQ", db_path) == pytest.approx(4.0)
    assert portfolio.get_discrepancy_qty("stable", "QQQ", db_path) == pytest.approx(4.0)
    assert portfolio.list_ledger_events("stable", "QQQ", event_type="WRITE_OFF_DRIFT", db_path=db_path) == []
    assert not any(o["symbol"] == "QQQ" and o["side"] == "sell" for o in api.submitted)
    kinds = [a["alert_type"] for a in sent]
    assert "LEDGER DISCREPANCY" in kinds and kinds.count("LEDGER DISCREPANCY") == 1   # alerted once, not per cycle


def test_evaluate_risk_absent_quote_is_flagged_not_suspect(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    _two_positions(db_path, api)
    ctx = make_ctx("stable", api, {}, db_path=db_path)
    quotes = {"SPY": BrokerQuote("ok", 4.0, 60.0), "QQQ": BrokerQuote("absent", 0.0, None)}
    r1 = evaluate_risk(ctx, {"SPY": 60.0, "QQQ": 100.0}, verify_mark=quotes.get)
    r2 = evaluate_risk(ctx, {"SPY": 60.0, "QQQ": 100.0}, verify_mark=quotes.get)
    assert (r1.level, r1.kill_now) == ("critical", False)
    assert (r2.level, r2.kill_now) == ("critical", True)
    assert r2.discrepancy_symbols == ["QQQ"] and r2.suspect_symbols == []


def test_unknown_broker_state_during_critical_is_still_suspect(db_path):
    """Fail closed is kept for UNKNOWN state: an outage cannot confirm a kill on its own."""
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    _two_positions(db_path, api)
    ctx = make_ctx("stable", api, {}, db_path=db_path)
    quotes = {"SPY": BrokerQuote("ok", 4.0, 60.0), "QQQ": BrokerQuote("unknown")}
    r = evaluate_risk(ctx, {"SPY": 60.0, "QQQ": 100.0}, verify_mark=quotes.get)
    assert r.level == "suspect" and r.kill_now is False and r.suspect_symbols == ["QQQ"]
    assert portfolio.get_discrepancy_qty("stable", "QQQ", db_path) == 0.0          # nothing flagged on unknown


def test_strategy_holding_only_vanished_positions_can_still_be_killed(db_path):
    """A genuine gradual decline on vanished positions is followed (ratchet-down reference)."""
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    _two_positions(db_path, api)
    api.positions.clear()                                        # everything vanished
    frames = {}
    ctx = make_ctx("stable", api, frames, db_path=db_path)
    ctx.assets = ["SPY", "QQQ"]
    _alerts(ctx)
    outs = []
    for px in (60.0, 40.0, 40.0, 40.0):                         # -40% then -33%: each step plausible
        frames.update(SPY=make_featured_df(price=px), QQQ=make_featured_df(price=px))
        outs.append(run_cycle(ctx, api, **FAST))
    assert outs[0]["risk"].level == "critical" and not outs[0]["killed"]   # -32%: reading 1 of 2
    assert outs[1]["risk"].equity == pytest.approx(200.0 + 8 * 40.0)    # valued at the declined price
    assert outs[1]["killed"] is True and outs[2]["halted"] is True
    results = outs[2]["halted_liquidation"]
    assert sorted(results) == [("QQQ", "discrepancy"), ("SPY", "discrepancy")]   # no broker sell attempted
    assert [o for o in api.submitted if o["side"] == "sell"] == []
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(4.0)


def test_single_implausible_print_on_a_vanished_position_cannot_fake_a_kill(db_path):
    """No broker quote exists to check it, so a >50% one-step print is not trusted (T-06 rule kept)."""
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 4.0, 100.0, "x", db_path=db_path, **FAST)
    api.positions.clear()
    frames = {"SPY": make_featured_df(price=1.0)}               # bad tick: -99%
    ctx = make_ctx("stable", api, frames, db_path=db_path)
    _alerts(ctx)
    outs = [run_cycle(ctx, api, **FAST) for _ in range(3)]
    assert all(o["risk"].level == "safe" and not o["killed"] for o in outs)
    assert outs[-1]["risk"].equity == pytest.approx(1000.0)      # min(last accepted mark, cost)


# ---------------------------------------------------------------------------
# Conservative valuation without realizing anything
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mark, expected_px", [(150.0, 100.0), (70.0, 70.0)])
def test_flagged_quantity_is_valued_at_min_of_mark_last_mark_and_cost(db_path, mark, expected_px):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 4.0, 100.0, "x", db_path=db_path, **FAST)
    cash = portfolio.get_strategy_state("stable", db_path)["cash_budget"]
    portfolio.flag_discrepancy("stable", "SPY", 4.0, "test", db_path=db_path)
    eq, value, _ = portfolio.strategy_equity("stable", {"SPY": mark}, db_path)
    assert value == pytest.approx(4.0 * expected_px)        # a gain on unseen shares is never counted
    assert eq == pytest.approx(cash + 4.0 * expected_px)
    # nothing realized, nothing mutated
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == cash
    assert portfolio.get_position("stable", "SPY", db_path)[:2] == (pytest.approx(4.0), pytest.approx(100.0))
    assert [e["event_type"] for e in portfolio.list_ledger_events("stable", db_path=db_path)] == ["FILL"]


def test_partially_flagged_position_values_only_the_flagged_part_conservatively(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 4.0, 100.0, "x", db_path=db_path, **FAST)
    portfolio.flag_discrepancy("stable", "SPY", 1.0, "test", db_path=db_path)
    _, value, _ = portfolio.strategy_equity("stable", {"SPY": 120.0}, db_path)
    assert value == pytest.approx(3.0 * 120.0 + 1.0 * 100.0)


# ---------------------------------------------------------------------------
# Strict "absent" classification (D4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc, expected", [
    (FakeAPIError("position does not exist: SPY", 404), "absent"),
    (AlpacaCodeError("position not found", code=40410000), "absent"),          # code, no HTTP response attached
    (AlpacaCodeError("position does not exist"), "absent"),                    # message only
    (FakeAPIError("404 Not Found", 404), "unknown"),                           # bare 404: wrong route/proxy/etc.
    (FakeAPIError("gateway timeout", 504), "unknown"),
    (FakeAPIError("rate limited", 429), "unknown"),
    (ConnectionError("connection reset"), "unknown"),
])
def test_only_authoritative_answers_count_as_absent(exc, expected):
    api = FakeAlpaca(prices={"SPY": 100.0})
    api.fail_get_position = exc
    assert broker_quote(api, "SPY").status == expected


def test_bare_404_never_flags_a_position(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    api.fail_get_position = FakeAPIError("404 Not Found", 404)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=100.0)}, db_path=db_path)
    refresh_discrepancies(ctx, api)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == 0.0


# ---------------------------------------------------------------------------
# Flag lifecycle
# ---------------------------------------------------------------------------

def test_in_flight_sell_is_not_flagged_and_reconciles_cleanly(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    api.fill_mode = "pending"
    res = submit_and_track(api, "stable", "SPY", "sell", 1.0, 100.0, "x", db_path=db_path, **FAST)
    api.fill_pending(res.order_id)                          # sold at the broker, not yet booked locally
    ctx = make_ctx("stable", api, {}, db_path=db_path)
    refresh_discrepancies(ctx, api)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == 0.0   # open order explains the gap
    reconcile_open_orders(api, "stable", db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0


def test_flag_clears_itself_when_the_broker_holds_the_position_again(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    api.positions.clear()
    ctx = make_ctx("stable", api, {}, db_path=db_path)
    refresh_discrepancies(ctx, api)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == pytest.approx(1.0)
    api.seed_position("SPY", 1.0, 100.0)                     # e.g. the broker's view caught up
    refresh_discrepancies(ctx, api)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == 0.0
    statuses = [r["status"] for r in get_trades_full("stable", db_path) if r["action"] == "RECONCILE"]
    assert statuses == ["discrepancy_flagged", "discrepancy_cleared"]


def test_flagged_symbols_take_no_new_entries(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.5, 100.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "stable", "QQQ", "buy", 0.5, 100.0, "x", db_path=db_path, **FAST)
    api.positions["SPY"]["qty"] = 0.25                      # half of SPY vanished
    del api.positions["QQQ"]                                # all of QQQ vanished
    frames = {"SPY": make_featured_df(price=100.0), "QQQ": make_featured_df(price=100.0)}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    _alerts(ctx)
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["SPY"] == "entry-blocked-discrepancy"
    assert out["tickers"]["QQQ"] == "discrepancy-flagged"
    assert not any(o["side"] == "buy" for o in api.submitted[2:])


def test_kill_switch_liquidates_the_unflagged_part_and_keeps_the_flag(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    api.positions["SPY"]["qty"] = 0.6
    portfolio.flag_discrepancy("stable", "SPY", 0.4, "test", db_path=db_path)
    results = close_all_strategy_positions(api, "stable", {"SPY": 100.0}, "KILL", db_path=db_path, **FAST)
    assert [(r.status, round(r.filled_qty, 6)) for r in results] == [("filled", 0.6)]
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# D2: persistent suspect state alerts, blocks entries, never liquidates
# ---------------------------------------------------------------------------

def test_persistent_suspect_state_alerts_blocks_entries_and_never_liquidates(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    frames = {"QQQ": make_featured_df(price=100.0)}          # SPY's primary mark is missing ...
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    ctx.assets = ["SPY", "QQQ"]
    sent = _alerts(ctx)
    api.fail_get_position = FakeAPIError("service unavailable", 503)   # ... and the broker is down
    n = SUSPECT_ALERT_AFTER_CYCLES + SUSPECT_ALERT_REPEAT_CYCLES
    outs = [run_cycle(ctx, api, **FAST) for _ in range(n)]
    assert all(o["risk"].level == "suspect" for o in outs)
    assert [o["risk"].suspect_streak for o in outs] == list(range(1, n + 1))
    suspect_alerts = [a for a in sent if a["alert_type"] == "RISK STATE SUSPECT"]
    assert len(suspect_alerts) == 2                                      # at cycle 3 and cycle 15
    assert all(o["tickers"]["QQQ"] == "entry-blocked" for o in outs)    # entries blocked
    assert len(api.submitted) == 1                                      # only the original buy: nothing liquidated
    assert get_heartbeat("stable", db_path)[1] == "SUSPECT"
    # recovery: one alert, streak reset, heartbeat back to RUNNING
    api.fail_get_position = None
    frames["SPY"] = make_featured_df(price=100.0)
    out = run_cycle(ctx, api, **FAST)
    assert out["risk"].level == "safe" and out["risk"].suspect_streak == 0
    assert [a["alert_type"] for a in sent][-1] == "RISK STATE RECOVERED"
    assert portfolio.get_strategy_state("stable", db_path)["suspect_streak"] == 0
    assert get_heartbeat("stable", db_path)[1] == "RUNNING"


def test_short_suspect_blip_does_not_alert(db_path):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **FAST)
    frames = {}
    ctx = make_ctx("stable", api, frames, db_path=db_path)
    ctx.assets = ["SPY"]
    sent = _alerts(ctx)
    api.fail_get_position = FakeAPIError("service unavailable", 503)
    for _ in range(SUSPECT_ALERT_AFTER_CYCLES - 1):
        run_cycle(ctx, api, **FAST)
    api.fail_get_position = None
    frames["SPY"] = make_featured_df(price=100.0)
    run_cycle(ctx, api, **FAST)
    assert sent == []


# ---------------------------------------------------------------------------
# Manual reconciliation tool and migration
# ---------------------------------------------------------------------------

def test_reconcile_tool_is_dry_run_by_default_and_writes_off_only_with_confirm(db_path, capsys):
    from tools.reconcile_position import main
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **FAST)
    portfolio.flag_discrepancy("stable", "SPY", 2.0, "test", db_path=db_path)
    assert main([], db_path=db_path) == 0 and "missing at broker 2.000000" in capsys.readouterr().out
    main(["write-off", "stable", "SPY", "--mark", "90", "--note", "checked broker"], db_path=db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(2.0)   # dry run: unchanged
    main(["write-off", "stable", "SPY", "--mark", "90", "--note", "checked broker", "--confirm"], db_path=db_path)
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0
    ev = portfolio.list_ledger_events("stable", "SPY", event_type="WRITE_OFF_DRIFT", db_path=db_path)
    assert len(ev) == 1 and ev[0]["cash_delta"] == pytest.approx(180.0) and ev[0]["broker_order_id"] is None


def test_reconcile_tool_clear_removes_only_the_flag(db_path):
    from tools.reconcile_position import main
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 2.0, 100.0, "x", db_path=db_path, **FAST)
    portfolio.flag_discrepancy("stable", "SPY", 2.0, "test", db_path=db_path)
    cash = portfolio.get_strategy_state("stable", db_path)["cash_budget"]
    main(["clear", "stable", "SPY", "--note", "broker shows 2 shares", "--confirm"], db_path=db_path)
    assert portfolio.get_discrepancy_qty("stable", "SPY", db_path) == 0.0
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(2.0)
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == cash


def test_migration_adds_discrepancy_and_suspect_columns_to_an_audit_hardening_database(tmp_path):
    """A database created by audit-hardening (ledger tables, no R-1 columns) migrates in place."""
    path = str(tmp_path / "ah.db")
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE positions (strategy TEXT NOT NULL, symbol TEXT NOT NULL, qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL, opened_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    last_mark REAL, last_mark_at TEXT, PRIMARY KEY (strategy, symbol))''')
    conn.execute('''CREATE TABLE strategy_state (strategy TEXT PRIMARY KEY, cash_budget REAL NOT NULL,
                    peak_equity REAL NOT NULL, last_equity REAL, critical_streak INTEGER NOT NULL DEFAULT 0,
                    halted INTEGER NOT NULL DEFAULT 0, halted_reason TEXT, updated_at TEXT NOT NULL)''')
    conn.execute("INSERT INTO positions VALUES ('stable','SPY',3.0,100.0,'2026-09-01','2026-09-01',101.0,NULL)")
    conn.execute("INSERT INTO strategy_state VALUES ('stable',700.0,1010.0,1003.0,0,0,NULL,'2026-09-01')")
    conn.commit(); conn.close()
    init_db(path)
    init_db(path)                                                    # idempotent
    with db_connection(path) as c:
        pos_cols = {r[1] for r in c.execute("PRAGMA table_info(positions)")}
        st_cols = {r[1] for r in c.execute("PRAGMA table_info(strategy_state)")}
    assert {"discrepancy_qty", "discrepancy_reason", "discrepancy_since", "discrepancy_alerted"} <= pos_cols
    assert "suspect_streak" in st_cols
    assert portfolio.get_position("stable", "SPY", path)[:2] == (3.0, 100.0)     # existing ledger untouched
    st = portfolio.get_strategy_state("stable", path)
    assert (st["cash_budget"], st["peak_equity"], st["suspect_streak"]) == (700.0, 1010.0, 0)
    assert portfolio.get_discrepancy_qty("stable", "SPY", path) == 0.0


def test_flagging_a_position_does_not_change_the_atr_adjusted_thresholds(db_path):
    """Audit probe P4 exactly: the same loss must kill whether or not QQQ vanished at the broker."""
    outcomes = {}
    for vanish in (False, True):
        path = f"{db_path}.{vanish}.db"
        init_db(path)
        api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
        for s in ("SPY", "QQQ"):
            submit_and_track(api, "stable", s, "buy", 4.0, 100.0, "p", db_path=path, **FAST)
        if vanish:
            del api.positions["QQQ"]
        frames = {"QQQ": make_featured_df(price=100.0, atr=1.0)}
        ctx = make_ctx("stable", api, frames, db_path=path)
        ctx.assets = ["SPY", "QQQ"]
        _alerts(ctx)
        killed = []
        for px in (70.0, 50.0, 40.0):
            api.set_price("SPY", px)
            frames["SPY"] = make_featured_df(price=px, atr=px / 100)
            killed.append(run_cycle(ctx, api, **FAST)["killed"])
        outcomes[vanish] = killed
    assert outcomes[False] == outcomes[True] == [False, True, False]
