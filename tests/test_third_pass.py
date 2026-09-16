"""
Third-pass adversarial regressions (ENGINEERING_AUDIT.md section 12).

T-01 internal write-off / dust must never manufacture cash, equity or positive realized P&L
T-02 the LIVE kill-switch path uses ATR-adjusted thresholds, bounded to [0.5x, 1.5x]
T-03 unknown broker position state is never treated as zero inventory
T-04 zero / missing / NaN ATR baseline or ATR input falls back to static thresholds
T-05 every inventory mutation is journaled with enough detail to explain it
T-06 corrupt or stale marks cannot cause a false kill switch; genuine crashes still kill
T-07 a position-dependent decider (PPO-style) cannot oscillate within one bar
T-08 the core paper-trading runtime imports without RL / backtest / dashboard packages
"""
import math
import subprocess
import sys
import os

import numpy as np
import pytest

from core import portfolio
from core.execution import (submit_and_track, close_strategy_position, broker_position_qty,
                            close_all_strategy_positions)
from core.logger import get_trades_full
from metrics.risk_manager import (get_atr_adjusted_thresholds, risk_level_from_drawdown,
                                  portfolio_atr_fraction, atr_as_fraction)
from strategies.common import Decision, run_cycle, evaluate_risk, trade_ticker
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca, FakeAPIError
from tests.test_engine import make_ctx, always

FAST = dict(sleep_fn=lambda s: None)


def _equity(strategy, marks, db_path):
    return portfolio.strategy_equity(strategy, marks, db_path)[0]


# T-01 ----------------------------------------------------------------------

def test_t01_drift_write_off_never_increases_equity_or_books_a_gain(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    api.set_price("SPY", 600.0)                      # unrealized gain of $40
    api.positions.clear()                            # shares vanished (sold outside the system)
    before = _equity("stable", {"SPY": 600.0}, db_path)
    res = close_strategy_position(api, "stable", "SPY", 600.0, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "reconciled"
    after = _equity("stable", {}, db_path)
    assert after <= before + 1e-9                    # equity can only go down or stay
    assert after == pytest.approx(1000.0)            # credited at min(mark, cost) = cost: the gain is forfeited
    ev = portfolio.list_ledger_events("stable", "SPY", db_path=db_path)
    assert ev[-1]["event_type"] == "WRITE_OFF_DRIFT" and ev[-1]["broker_order_id"] is None
    assert ev[-1]["cash_delta"] == pytest.approx(200.0) and ev[-1]["qty_delta"] == pytest.approx(-0.4)
    assert "realized_estimate=0.000000" in ev[-1]["extra"]
    assert not any(o["side"] == "sell" for o in api.submitted)        # no fake broker sale


def test_t01_drift_write_off_at_a_loss_realizes_the_loss_not_a_gain(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    api.positions.clear()
    res = close_strategy_position(api, "stable", "SPY", 400.0, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "reconciled"
    assert _equity("stable", {}, db_path) == pytest.approx(1000.0 - 0.4 * 100.0)
    ev = portfolio.list_ledger_events("stable", "SPY", event_type="WRITE_OFF_DRIFT", db_path=db_path)[-1]
    assert ev["price"] == pytest.approx(400.0) and "realized_estimate=-40.000000" in ev["extra"]


def test_t01_dust_is_left_in_place_and_credits_nothing(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    api.seed_position("SPY", 0.001, 500.0)
    portfolio.adopt_position("stable", "SPY", 0.001, 500.0, db_path)          # $0.50 of dust
    cash_before = portfolio.get_strategy_state("stable", db_path)["cash_budget"]
    res = close_strategy_position(api, "stable", "SPY", 500.0, "ML SELL", db_path=db_path, **FAST)
    assert res.status == "dust" and api.submitted == []
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.001)   # still owned
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == cash_before  # nothing credited
    assert portfolio.list_ledger_events("stable", "SPY", event_type="WRITE_OFF_DRIFT", db_path=db_path) == []
    # kill-switch follow-up liquidation tolerates dust without looping on it
    results = close_all_strategy_positions(api, "stable", {"SPY": 500.0}, "KILL_SWITCH", db_path=db_path, **FAST)
    assert [r.status for r in results] == ["dust"]


def test_t01_apply_fill_is_not_used_for_internal_write_offs(db_path):
    """A write-off is a distinct event type; apply_fill only ever journals FILL."""
    api = FakeAlpaca(prices={"AAPL": 200.0})
    submit_and_track(api, "stable", "AAPL", "buy", 1.0, 200.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "AAPL", "buy", 0.5, 200.0, "x", db_path=db_path, **FAST)
    api.positions["AAPL"]["qty"] = 1.2
    close_strategy_position(api, "risky1", "AAPL", 200.0, "x", db_path=db_path, **FAST)
    kinds = [e["event_type"] for e in portfolio.list_ledger_events("risky1", "AAPL", db_path=db_path)]
    assert kinds == ["FILL", "WRITE_OFF_DRIFT", "FILL"]      # buy, drift write-off of 0.3, real sale of 0.2
    fills = [e for e in portfolio.list_ledger_events("risky1", "AAPL", event_type="FILL", db_path=db_path)]
    assert all(e["broker_order_id"] for e in fills)


# T-02 ----------------------------------------------------------------------

def test_t02_live_kill_switch_path_uses_atr_adjusted_thresholds_within_bounds(db_path):
    """stable: static kill -15%. ATR 2x baseline -> 1.5x -> -22.5%; ATR 0.5x baseline -> -7.5%."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    verify = lambda sym: 420.0                                    # broker agrees with the move
    # -16% strategy drawdown (0.4 * 80 = $32 on 1000... use a bigger position): adopt more
    portfolio.adopt_position("stable", "SPY", 1.6, 500.0, db_path)   # 2.0 sh @ 500 = $1000 exposure
    # SPY 500 -> 420: equity 1000 - 2*80 = 840 -> -16%
    volatile = {"SPY": 0.03}                                      # 2x the 0.015 baseline -> clamp 1.5x
    r = evaluate_risk(ctx, {"SPY": 420.0}, atr_fractions=volatile, verify_mark=verify)
    assert r.level == "warning" and r.atr_fraction == pytest.approx(0.03)   # -16% is inside -22.5%
    # Static path (no ATR) would have been critical at -16%
    assert risk_level_from_drawdown("stable", -0.16, None) == "critical"
    # Extreme ATR never widens beyond the clamp: -23% must be critical whatever the ATR
    ctx2 = make_ctx("risky1", api, {"NVDA": None}, db_path=db_path)
    assert risk_level_from_drawdown("risky1", -0.451, 100.0) == "critical"    # 1.5 * -0.30 = -0.45
    assert risk_level_from_drawdown("risky1", -0.449, 100.0) != "critical"
    # Calm market tightens: -8% with ATR at half baseline is critical for stable (-7.5%)
    assert risk_level_from_drawdown("stable", -0.08, 0.0075) == "critical"
    assert risk_level_from_drawdown("stable", -0.08, None) == "warning"


def test_t02_missing_or_unheld_atr_falls_back_to_static(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    portfolio.adopt_position("stable", "SPY", 2.0, 500.0, db_path)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    r = evaluate_risk(ctx, {"SPY": 420.0}, atr_fractions={"SPY": None}, verify_mark=lambda s: 420.0)
    assert r.atr_fraction is None and r.level == "critical"      # static -15% applies to -16%
    assert portfolio_atr_fraction({}, {}) is None
    assert portfolio_atr_fraction({"A": 100.0, "B": 300.0}, {"A": 0.01, "B": 0.03}) == pytest.approx(0.025)
    assert portfolio_atr_fraction({"A": 100.0}, {"A": float("nan")}) is None


def test_t02_run_cycle_passes_atr_into_the_decision(db_path):
    """End to end: a frame with a volatile ATR keeps -16% at warning; a calm ATR makes it critical."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    # 8 shares @ 500 ($4,000 exposure on a $1,000 budget): a 4% price move is a 16%
    # strategy drawdown WITHOUT tripping the 5% per-trade stop, so only the
    # strategy-level threshold decides the outcome.
    api.seed_position("SPY", 8.0, 500.0)
    portfolio.adopt_position("stable", "SPY", 8.0, 500.0, db_path)
    api.set_price("SPY", 480.0)
    frames = {"SPY": make_featured_df(price=480.0, atr=14.4)}      # atr/close = 0.03 -> clamp 1.5 -> kill at -22.5%
    ctx = make_ctx("stable", api, frames, decide=always("HOLD"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)
    assert out["risk"].drawdown == pytest.approx(-0.16)
    assert out["risk"].level == "warning" and out["risk"].atr_fraction == pytest.approx(0.03)
    assert out["tickers"]["SPY"] == "hold" and not any(o["side"] == "sell" for o in api.submitted)
    frames["SPY"] = make_featured_df(price=480.0, atr=3.6)        # atr/close = 0.0075 -> 0.5x -> kill at -7.5%
    out = run_cycle(ctx, api, **FAST)
    assert out["risk"].level == "critical" and out["risk"].atr_fraction == pytest.approx(0.0075)
    assert out["killed"] is False                                  # first critical reading: awaiting confirmation


def test_t02_strategy_level_warning_halves_size_even_if_ticker_atr_is_wide(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0, "QQQ": 400.0})
    portfolio.adopt_position("stable", "SPY", 2.0, 500.0, db_path)
    ctx = make_ctx("stable", api, {"QQQ": make_featured_df(price=400.0, atr=40.0)}, decide=always("BUY"), db_path=db_path)
    risk = evaluate_risk(ctx, {"SPY": 460.0}, atr_fractions={"SPY": 0.015}, verify_mark=lambda s: 460.0)   # -8%: warning
    assert risk.level == "warning"
    out = trade_ticker(ctx, api, "QQQ", make_featured_df(price=400.0, atr=40.0), risk)   # QQQ ATR 10%: ticker says "safe"
    assert out == "buy-filled"
    assert api.submitted[-1]["qty"] * 400.0 == pytest.approx(100.0, abs=0.05)              # half of $200


# T-03 ----------------------------------------------------------------------

@pytest.mark.parametrize("exc,expected", [
    (FakeAPIError("position does not exist", 404), (0.0, True)),
    (FakeAPIError("read timed out", None), (None, False)),
    (FakeAPIError("internal server error", 500), (None, False)),
    (FakeAPIError("too many requests", 429), (None, False)),
    (ConnectionError("connection reset"), (None, False)),
    (RuntimeError("weird transport failure"), (None, False)),
])
def test_t03_broker_position_lookup_classification(exc, expected):
    api = FakeAlpaca(prices={"SPY": 500.0})
    api.fail_get_position = exc
    assert broker_position_qty(api, "SPY") == expected


def test_t03_alpaca_style_error_objects_are_classified_correctly():
    class AlpacaLikeError(Exception):
        def __init__(self, message, code, status):
            super().__init__(message); self._code = code; self._status = status
        @property
        def code(self): return self._code
        @property
        def status_code(self): return self._status
    api = FakeAlpaca(prices={"SPY": 500.0})
    api.fail_get_position = AlpacaLikeError("position does not exist", 40410000, 404)
    assert broker_position_qty(api, "SPY") == (0.0, True)
    api.fail_get_position = AlpacaLikeError("forbidden", 40310000, 403)
    assert broker_position_qty(api, "SPY") == (None, False)
    class NoResponse(Exception):
        @property
        def status_code(self): return None      # alpaca APIError without an HTTP response attached
    api.fail_get_position = NoResponse("unexpected")
    assert broker_position_qty(api, "SPY") == (None, False)


def test_t03_successful_lookup_and_transient_failure_do_not_write_off(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    assert broker_position_qty(api, "SPY") == (pytest.approx(0.4), True)
    for exc in (FakeAPIError("timeout", None), FakeAPIError("server error", 500), FakeAPIError("rate limited", 429)):
        api.fail_get_position = exc
        res = close_strategy_position(api, "stable", "SPY", 500.0, "ML SELL", db_path=db_path, **FAST)
        assert res.status == "refused" and "unknown" in res.message
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)   # nothing written off
    assert portfolio.list_ledger_events("stable", "SPY", event_type="WRITE_OFF_DRIFT", db_path=db_path) == []
    assert not any(o["side"] == "sell" for o in api.submitted)
    api.fail_get_position = None
    assert close_strategy_position(api, "stable", "SPY", 500.0, "ML SELL", db_path=db_path, **FAST).status == "filled"


def test_t03_engine_keeps_managing_a_position_through_a_broker_outage(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    api.fail_get_position = FakeAPIError("gateway timeout", 504)
    api.set_price("SPY", 470.0)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=470.0)}, decide=always("HOLD"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)                              # stop loss wants to fire...
    assert out["tickers"]["SPY"] == "emergency-exit"               # ...but the sell is refused, not written off
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)
    api.fail_get_position = None
    out = run_cycle(ctx, api, **FAST)                              # broker back: the stop executes
    assert portfolio.get_position_qty("stable", "SPY", db_path) == 0.0


# T-04 ----------------------------------------------------------------------

def test_t04_bad_baseline_or_atr_falls_back_to_static_thresholds(monkeypatch):
    import metrics.risk_manager as rm
    static = (rm.WARNING_DRAWDOWN["stable"], rm.MAX_DRAWDOWN["stable"])
    for bad in (float("nan"), float("inf"), -0.01, None, "abc"):
        assert get_atr_adjusted_thresholds("stable", bad) == static
    monkeypatch.setitem(rm.ATR_BASELINE, "stable", 0.0)
    assert get_atr_adjusted_thresholds("stable", 0.02) == static
    monkeypatch.setitem(rm.ATR_BASELINE, "stable", float("nan"))
    assert get_atr_adjusted_thresholds("stable", 0.02) == static      # min(1.5, nan) would have widened
    monkeypatch.delitem(rm.ATR_BASELINE, "stable")
    assert get_atr_adjusted_thresholds("stable", 0.02) == static
    assert risk_level_from_drawdown("stable", -0.16, float("nan")) == "critical"


# T-05 ----------------------------------------------------------------------

def test_t05_every_inventory_change_is_journaled_with_provenance(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="partial", partial_fraction=0.5)
    import itertools
    clock = itertools.count(0, 30)
    res = submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path,
                           sleep_fn=lambda s: None, clock_fn=lambda: next(clock))
    api.fill_pending(res.order_id)
    from core.execution import reconcile_open_orders
    reconcile_open_orders(api, "stable", db_path)
    portfolio.adopt_position("stable", "QQQ", 0.5, 400.0, db_path)
    api.fill_mode = "fill"
    api.positions.clear()
    close_strategy_position(api, "stable", "SPY", 480.0, "ML SELL", db_path=db_path, **FAST)
    events = portfolio.list_ledger_events("stable", db_path=db_path)
    kinds = [(e["symbol"], e["event_type"]) for e in events]
    assert kinds == [("SPY", "FILL"), ("SPY", "FILL"), ("QQQ", "ADOPT"), ("SPY", "WRITE_OFF_DRIFT")]
    for e in events:
        assert e["timestamp"] and e["strategy"] == "stable" and e["reason"]
        assert e["qty_after"] == pytest.approx(e["qty_before"] + e["qty_delta"])
        if e["event_type"] == "FILL":
            assert e["broker_order_id"] == res.order_id
        else:
            assert e["broker_order_id"] is None
    wo = events[-1]
    assert "account_qty=0.0" in wo["extra"] and "others_claim=0.0" in wo["extra"]
    # the ledger's cash budget equals CAPITAL + the sum of all journaled cash deltas
    cash = portfolio.get_strategy_state("stable", db_path)["cash_budget"]
    assert cash == pytest.approx(1000.0 + sum(e["cash_delta"] for e in events))


# T-06 ----------------------------------------------------------------------

def _held_stable(db_path, api):
    submit_and_track(api, "stable", "SPY", "buy", 2.0, 500.0, "x", db_path=db_path, **FAST)


def test_t06_one_extreme_bad_bar_is_overridden_by_the_broker_quote(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    frames = {"SPY": make_featured_df(price=5.0)}                 # corrupt bar: -99%
    ctx = make_ctx("stable", api, frames, decide=always("HOLD"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)
    r = out["risk"]
    assert r.level == "safe" and r.accepted_marks["SPY"] == pytest.approx(500.0)
    assert r.equity == pytest.approx(1000.0) and not r.kill_now
    assert out["tickers"]["SPY"] == "hold"                        # no stop-loss on the bad price


def test_t06_two_identical_bad_bars_do_not_confirm_each_other(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    for _ in range(3):                                            # repeated, identical, no independent quote
        r = evaluate_risk(ctx, {"SPY": 5.0}, verify_mark=lambda s: None)
        assert r.level == "suspect" and r.kill_now is False and "SPY" in r.suspect_symbols
    assert portfolio.get_strategy_state("stable", db_path)["critical_streak"] == 0
    assert portfolio.get_strategy_state("stable", db_path)["peak_equity"] == pytest.approx(1000.0)


def test_t06_repeated_stale_bar_uses_the_broker_quote(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    api.set_price("SPY", 400.0)                                   # reality fell 20%
    stale = make_featured_df(price=500.0, start="2026-06-01")     # bar weeks old, still says 500
    ctx = make_ctx("stable", api, {"SPY": stale}, decide=always("BUY"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)
    r = out["risk"]
    assert r.accepted_marks["SPY"] == pytest.approx(400.0) and "SPY" in r.stale_symbols
    assert r.level == "critical" and r.drawdown == pytest.approx(-0.20)   # valued at the REAL price
    # the stop loss uses the validated price too: -20% vs entry closes the position,
    # and nothing is ever bought on the stale bar
    assert out["tickers"]["SPY"] == "emergency-exit"
    assert api.submitted[-1]["side"] == "sell" and not any(o["side"] == "buy" for o in api.submitted[1:])
    # the bar is still stale on the next cycle: still no entry, and the confirmed
    # critical reading fires the kill switch on what is left of the strategy
    out = run_cycle(ctx, api, **FAST)
    assert out["killed"] is True and not any(o["side"] == "buy" for o in api.submitted[1:])


def test_t06_genuine_rapid_crash_still_kills_quickly(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    api.set_price("SPY", 200.0)                                   # -60%, corroborated by the broker
    frames = {"SPY": make_featured_df(price=200.0)}
    ctx = make_ctx("stable", api, frames, decide=always("HOLD"), db_path=db_path)
    first = run_cycle(ctx, api, **FAST)
    assert first["risk"].level == "critical" and first["killed"] is False
    second = run_cycle(ctx, api, **FAST)
    assert second["killed"] is True
    assert portfolio.get_strategy_state("stable", db_path)["halted"] is True


def test_t06_recovery_after_a_bad_reading(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    bad = evaluate_risk(ctx, {"SPY": 5.0}, verify_mark=lambda s: None)
    assert bad.level == "suspect"
    good = evaluate_risk(ctx, {"SPY": 505.0}, verify_mark=lambda s: 505.0)
    assert good.level == "safe" and good.equity == pytest.approx(1010.0)
    assert portfolio.get_last_marks("stable", db_path)["SPY"] == pytest.approx(505.0)


def test_t06_suspect_symbol_blocks_stop_loss_on_the_bad_price(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    _held_stable(db_path, api)
    api.fail_get_position = FakeAPIError("timeout", None)         # no independent quote either
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=5.0)}, decide=always("SELL"), db_path=db_path)
    out = run_cycle(ctx, api, **FAST)
    assert out["risk"].level == "suspect" and out["tickers"]["SPY"] == "suspect-mark"
    assert not any(o["side"] == "sell" for o in api.submitted)


# T-07 ----------------------------------------------------------------------

def test_t07_position_dependent_decider_cannot_oscillate_within_a_bar(db_path):
    """PPO-style: BUY when flat, SELL when long, on the same static daily bar."""
    def ppo_like(ctx, ticker, df, owned_qty, avg, price):
        return Decision("SELL" if owned_qty > 0 else "BUY", None, "ppo")
    api = FakeAlpaca(prices={"BTCUSD": 60000.0})
    frames = {"X:BTCUSD": make_featured_df(price=60000.0, start="2026-08-01")}
    ctx = make_ctx("risky2", api, frames, decide=ppo_like, db_path=db_path)
    outs = [run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] for _ in range(6)]
    assert outs[0] == "buy-filled" and all(o == "hold-deferred" for o in outs[1:])
    assert len(api.submitted) == 1
    # a new bar allows the exit (min hold is enforced separately; simulate it elapsed)
    from core.logger import db_connection
    with db_connection(db_path) as conn:
        conn.execute("UPDATE positions SET opened_at = '2026-01-01T00:00:00' WHERE strategy = 'risky2'")
    frames["X:BTCUSD"] = make_featured_df(price=60000.0, start="2026-08-02")
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "sell-filled"
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "entry-anti-churn"   # no re-entry on the exit bar
    assert len(api.submitted) == 2


# T-08 ----------------------------------------------------------------------

def test_t08_core_runtime_imports_without_optional_packages():
    code = r"""
import sys
for name in ("torch", "stable_baselines3", "gymnasium", "vectorbt", "streamlit", "plotly", "google", "google.genai"):
    sys.modules[name] = None      # makes `import name` raise ImportError
import run, strategies.stable, strategies.risky1, strategies.risky2, data.macro_fetcher, models.train, core.execution
run.preflight()
print("CORE_OK")
"""
    env = dict(os.environ, POLYGON_API_KEY="test_key", ALPACA_API_KEY="test_key", ALPACA_SECRET_KEY="test_key",
               SINGHQUANT_DB_PATH=os.environ["SINGHQUANT_DB_PATH"])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                       cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CORE_OK" in r.stdout
