"""
Regression tests for strategies/common.py (the shared engine).

Invariants covered:
  4  repeated identical signals cannot create duplicate exposure
  6  emergency risk exits bypass ordinary cooldown / min-hold
  7  drawdown uses valid STRATEGY equity
  8  missing/stale/implausible data cannot create a fake catastrophic drawdown
 14  crypto/equity timing stay separate
 15  NaN/inf features fail safely
 plus: the 2026-08-27 oscillation, HALT still evaluating exits, kill-switch
 confirmation and strategy-only liquidation, ATR units, retrain scheduling,
 live-endpoint guard, PPO live observation.
"""
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from core import portfolio
from core.execution import submit_and_track
from core.logger import get_trades_full, log_trade
from strategies.common import (StrategyContext, Decision, run_cycle, trade_ticker, evaluate_risk,
                               fire_kill_switch, xgb_decider, utcnow)
from tests.conftest import make_featured_df, StubModel, StubHandle
from tests.fakes import FakeAlpaca

FAST = dict(sleep_fn=lambda s: None)


def make_ctx(name, api, frames, decide=None, db_path=None, **kw):
    """Engine context with injected data, no network, no sleeps."""
    def fetch(ticker):
        return frames.get(ticker)
    base = dict(name=name, assets=list(frames.keys()), decide=decide or (lambda *a: Decision("HOLD")),
                fetch_frame=fetch, uses_market_hours=(name != "risky2"), momentum_exit=(name == "risky1"),
                regime_gate=True, time_in_force=("gtc" if name == "risky2" else "day"),
                per_ticker_sleep=0, cycle_sleep=0, macro_context=lambda: None,
                regime_ok=lambda df, s, vix: True, db_path=db_path, exec_kw=dict(FAST))
    base.update(kw)
    return StrategyContext(**base)


def always(signal, note=""):
    return lambda ctx, t, df, q, avg, px: Decision(signal, 1 if signal == "BUY" else 0, note)


# ---------------------------------------------------------------------------
# Investigation 4: the Aug-27 oscillation
# ---------------------------------------------------------------------------

def test_no_entry_when_momentum_exit_condition_already_true(db_path):
    """Same bar, momentum_5 < 0 and model says BUY: the old code bought, sold, bought, sold every cycle."""
    api = FakeAlpaca(prices={"TSLA": 354.25})
    df = make_featured_df(price=345.82, momentum_5=-0.02)
    ctx = make_ctx("risky1", api, {"TSLA": df}, decide=always("BUY"), db_path=db_path)
    outcomes = [run_cycle(ctx, api, **FAST)["tickers"]["TSLA"] for _ in range(4)]
    assert outcomes == ["entry-anti-churn"] * 4
    assert api.submitted == []


def test_oscillation_sequence_collapses_to_one_round_trip(db_path):
    """Four cycles that used to produce BUY,SELL,BUY,SELL now produce at most one BUY and one SELL."""
    api = FakeAlpaca(prices={"TSLA": 345.82})
    bar_a_up = make_featured_df(price=345.82, momentum_5=+0.02, start="2026-08-01")
    bar_a_dn = make_featured_df(price=345.82, momentum_5=-0.02, start="2026-08-01")   # same bar id
    bar_b_dn = make_featured_df(price=345.82, momentum_5=-0.02, start="2026-08-02")   # next bar
    frames = {"TSLA": bar_a_up}
    ctx = make_ctx("risky1", api, frames, decide=always("BUY"), db_path=db_path)
    r1 = run_cycle(ctx, api, **FAST)["tickers"]["TSLA"]          # buys on bar A
    frames["TSLA"] = bar_a_dn
    r2 = run_cycle(ctx, api, **FAST)["tickers"]["TSLA"]          # same bar: exit deferred
    frames["TSLA"] = bar_b_dn
    r3 = run_cycle(ctx, api, **FAST)["tickers"]["TSLA"]          # new bar: momentum exit
    r4 = run_cycle(ctx, api, **FAST)["tickers"]["TSLA"]          # same bar B: no re-entry
    assert r1 == "buy-filled" and r2 == "hold-deferred" and r3 == "momentum-exit" and r4 == "entry-anti-churn"
    sides = [o["side"] for o in api.submitted]
    assert sides == ["buy", "sell"]
    assert api.submitted[0]["qty"] == pytest.approx(round(50 / 345.82, 4)) == pytest.approx(0.1446)


def test_repeated_identical_buy_signals_do_not_multiply_exposure(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path)
    results = [run_cycle(ctx, api, **FAST)["tickers"]["SPY"] for _ in range(5)]
    assert results[0] == "buy-filled" and all(r == "at-max" for r in results[1:])
    assert len(api.submitted) == 1
    qty = portfolio.get_position_qty("stable", "SPY", db_path)
    assert qty * 500.0 <= 200.0 + 1e-6            # MAX_POSITION_SIZE["stable"]


def test_no_dust_top_ups_when_price_drifts(db_path):
    """Export evidence: 143 of 239 stable BUY rows were < $5 top-ups as prices drifted below max."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    frames = {"SPY": make_featured_df(price=500.0)}
    ctx = make_ctx("stable", api, frames, decide=always("BUY"), db_path=db_path)
    # Position worth $190 of a $200 target (95%): the old code bought $10 of dust every cycle.
    api.seed_position("SPY", 0.38, 500.0)
    portfolio.adopt_position("stable", "SPY", 0.38, 500.0, db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "at-max"
    assert api.submitted == []
    # Position worth $150 (75%): a meaningful top-up of ~$50 is still allowed.
    portfolio.apply_fill("stable", "SPY", "sell", 0.08, 500.0, db_path)
    api.positions["SPY"]["qty"] = 0.30
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "buy-filled"
    assert len(api.submitted) == 1 and api.submitted[0]["qty"] * 500.0 == pytest.approx(50.0, abs=0.05)


def test_partial_fill_is_topped_up_not_duplicated(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0}, fill_mode="partial", partial_fraction=0.5)
    clock = itertools.count(0, 30)
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path,
                   exec_kw=dict(sleep_fn=lambda s: None, clock_fn=lambda: next(clock)))
    r1 = run_cycle(ctx, api, **FAST)["tickers"]["SPY"]
    assert r1 == "buy-partially_filled"
    r2 = run_cycle(ctx, api, **FAST)["tickers"]["SPY"]      # open order still pending: no second order
    assert r2 == "buy-refused" and len(api.submitted) == 1
    api.fill_pending(api.submitted and list(api.orders)[0])
    r3 = run_cycle(ctx, api, **FAST)["tickers"]["SPY"]      # reconciled to full size
    assert r3 == "at-max"


# ---------------------------------------------------------------------------
# Emergency exits, HALT, min hold
# ---------------------------------------------------------------------------

def test_stop_loss_bypasses_min_hold_but_discretionary_sell_waits(db_path):
    api = FakeAlpaca(prices={"BTCUSD": 60000.0})
    frames = {"X:BTCUSD": make_featured_df(price=60000.0)}
    ctx = make_ctx("risky2", api, frames, decide=always("BUY"), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "buy-filled"
    # PPO now says SELL seconds later: min hold (1800 s) defers it
    ctx.decide = always("SELL")
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "hold-deferred"
    assert len(api.submitted) == 1
    # price collapses 15% (> 12% stop): emergency exit ignores the hold
    api.set_price("BTCUSD", 51000.0)
    frames["X:BTCUSD"] = make_featured_df(price=51000.0)
    assert run_cycle(ctx, api, **FAST)["tickers"]["X:BTCUSD"] == "emergency-exit"
    assert api.submitted[-1]["side"] == "sell"
    assert portfolio.get_position_qty("risky2", "X:BTCUSD", db_path) == 0.0


def test_halt_blocks_entries_but_still_runs_exits(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path)
    run_cycle(ctx, api, **FAST)
    assert portfolio.get_position_qty("stable", "SPY", db_path) > 0
    # Create a HALT: > 12% drawdown in the equity-curve filter (closed SELL pnl rows)
    for pnl in (-50.0, -50.0, -70.0):   # curve 950, 900, 830: (950-830)/950 = 12.6% > 12% HALT
        log_trade("stable", "QQQ", "SELL", 1.0, 1.0, pnl=pnl, reason="t", db_path=db_path)
    from metrics.equity_curve_filter import get_trading_state
    assert get_trading_state("stable", db_path)[0] == "HALT"
    # BUY signals are ignored while halted...
    api.set_price("SPY", 480.0)
    ctx.fetch_frame = lambda t: make_featured_df(price=480.0)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "entry-blocked"
    # ...but a stop loss still fires
    api.set_price("SPY", 470.0)
    ctx.fetch_frame = lambda t: make_featured_df(price=470.0)   # -6% vs entry 500 > 5% stop
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "emergency-exit"


def test_buy_signal_ignored_when_entries_disabled_direct(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path)
    risk = evaluate_risk(ctx, {"SPY": 500.0})
    out = trade_ticker(ctx, api, "SPY", make_featured_df(price=500.0), risk, allow_entries=False)
    assert out == "entry-blocked" and api.submitted == []


# ---------------------------------------------------------------------------
# Kill switch: strategy equity, confirmation, strategy-only liquidation
# ---------------------------------------------------------------------------

def test_kill_switch_ignores_account_equity(db_path):
    """Investigation 1: the old switch compared ACCOUNT equity to strategy CAPITAL."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    api.account_overrides["portfolio_value"] = 70.0     # account reads as 95% down
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, db_path=db_path)
    risk = evaluate_risk(ctx, {"SPY": 500.0})
    assert risk.level == "safe" and not risk.kill_now and risk.equity == pytest.approx(1000.0)


def test_kill_switch_needs_two_confirmations_then_closes_only_own_positions(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0, "NVDA": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    submit_and_track(api, "risky1", "NVDA", "buy", 0.5, 100.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    # SPY falls 80%: stable equity 1000 -> 840 (-16%), beyond the -15% switch
    r1 = evaluate_risk(ctx, {"SPY": 100.0})
    assert r1.level == "critical" and r1.kill_now is False
    r2 = evaluate_risk(ctx, {"SPY": 100.0})
    assert r2.kill_now is True and r2.drawdown == pytest.approx(-0.16)
    api.set_price("SPY", 100.0)
    fire_kill_switch(ctx, api, r2, {"SPY": 100.0})
    assert "SPY" not in api.positions                       # stable's position closed
    assert api.positions["NVDA"]["qty"] == pytest.approx(0.5)  # risky1's untouched
    assert api.close_all_calls == 0
    assert portfolio.get_strategy_state("stable", db_path)["halted"] is True
    rows = get_trades_full("stable", db_path)
    assert rows[-1]["action"] == "KILL_SWITCH" and rows[-1]["pnl"] is None
    # halted strategy does nothing on the next cycle
    ctx.fetch_frame = lambda t: make_featured_df(price=100.0)
    assert run_cycle(ctx, api, **FAST)["halted"] is True


def test_logged_drawdown_equals_deciding_drawdown(db_path):
    """The number in the audit trail must be the number the decision used."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    evaluate_risk(ctx, {"SPY": 400.0})
    from core.logger import db_connection
    with db_connection(db_path) as conn:
        dd, level = conn.execute("SELECT drawdown, risk_level FROM equity_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    assert dd == pytest.approx(-0.04) and level == "safe"


def test_implausible_equity_jump_is_suspect_and_does_not_kill(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    assert evaluate_risk(ctx, {"SPY": 500.0}).level == "safe"
    portfolio.adopt_position("stable", "QQQ", 10.0, 90.0, db_path)   # 900 of QQQ, cash now -100
    glitch = evaluate_risk(ctx, {"SPY": 500.0, "QQQ": 1.0})           # QQQ marked at $1: equity collapses ~90%
    assert glitch.level == "suspect" and glitch.kill_now is False
    state = portfolio.get_strategy_state("stable", db_path)
    assert state["critical_streak"] == 0
    # No entries while suspect
    out = trade_ticker(ctx, api, "SPY", make_featured_df(price=500.0), glitch, allow_entries=True)
    assert out in ("entry-blocked", "hold", "no-model")
    # A persistent reading (two cycles agreeing) is accepted as reality
    real = evaluate_risk(ctx, {"SPY": 500.0, "QQQ": 1.0})
    assert real.level == "critical" and real.kill_now is False   # first confirmed critical reading


def test_nan_or_inf_equity_is_rejected():
    from metrics.risk_manager import validate_equity
    assert validate_equity(float("nan"))[0] is False
    assert validate_equity(float("inf"))[0] is False
    assert validate_equity(0)[0] is False
    assert validate_equity(None)[0] is False
    assert validate_equity(1000, 1000)[0] is True
    assert validate_equity(100, 1000)[0] is False


def test_missing_marks_do_not_create_drawdown(db_path):
    """Invariant 8: no data -> valued at entry -> no fake drawdown, no kill."""
    api = FakeAlpaca(prices={"SPY": 500.0})
    submit_and_track(api, "stable", "SPY", "buy", 0.4, 500.0, "x", db_path=db_path, **FAST)
    ctx = make_ctx("stable", api, {"SPY": None}, db_path=db_path)
    risk = evaluate_risk(ctx, {})
    assert risk.level == "safe" and risk.equity == pytest.approx(1000.0)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "no-data"
    assert api.submitted[1:] == []


# ---------------------------------------------------------------------------
# Data validity, market hours, ML feature contract
# ---------------------------------------------------------------------------

def test_nan_feature_row_never_reaches_the_broker(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    df = make_featured_df(price=500.0, nan_last=True)
    ctx = make_ctx("stable", api, {"SPY": df}, decide=xgb_decider(StubHandle(StubModel(1))), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "decision-error"
    assert api.submitted == []


def test_model_feature_width_mismatch_is_detected():
    from models.train import features_for_model, FeatureMismatch, FEATURE_COLUMNS
    df = make_featured_df()
    assert features_for_model(df.tail(1), StubModel(1)).shape == (1, len(FEATURE_COLUMNS))
    with pytest.raises(FeatureMismatch):
        features_for_model(df.tail(1), StubModel(1, n_features=5))
    with pytest.raises(FeatureMismatch):
        features_for_model(df.drop(columns=["rsi"]).tail(1))


def test_equity_strategies_pause_when_market_closed_but_crypto_trades(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0, "BTCUSD": 60000.0}, market_open=False)
    stable = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)}, decide=always("BUY"), db_path=db_path)
    s = run_cycle(stable, api, **FAST)
    assert s["paused"] > 0 and s["tickers"] == {} and api.submitted == []
    crypto = make_ctx("risky2", api, {"X:BTCUSD": make_featured_df(price=60000.0)}, decide=always("BUY"), db_path=db_path)
    c = run_cycle(crypto, api, **FAST)
    assert c["paused"] == 0 and c["tickers"]["X:BTCUSD"] == "buy-filled"
    assert api.submitted[-1]["symbol"] == "BTCUSD" and api.submitted[-1]["tif"] == "gtc"


def test_missing_model_skips_entries_without_crashing(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    ctx = make_ctx("stable", api, {"SPY": make_featured_df(price=500.0)},
                   decide=xgb_decider(StubHandle(None)), db_path=db_path)
    assert run_cycle(ctx, api, **FAST)["tickers"]["SPY"] == "no-model"


# ---------------------------------------------------------------------------
# ATR units, scheduler, live guard, PPO observation
# ---------------------------------------------------------------------------

def test_atr_dollars_are_converted_to_fraction_before_scaling():
    from metrics.risk_manager import atr_as_fraction, get_atr_adjusted_thresholds
    frac = atr_as_fraction(14.93, 652.04)              # realistic SPY-like values from build_features
    assert frac == pytest.approx(0.0229, abs=1e-3)
    calm = atr_as_fraction(5.0, 652.04)                # 0.77% daily range
    w_calm, k_calm = get_atr_adjusted_thresholds("stable", calm)
    assert k_calm > -0.15                              # tighter than static in calm markets
    # The old code passed 14.93 directly: scale 995 -> clamped to 1.5x for every asset
    _, k_wrong = get_atr_adjusted_thresholds("stable", 14.93)
    assert k_wrong == pytest.approx(-0.225)
    assert atr_as_fraction(float("nan"), 100) is None and atr_as_fraction(1.0, 0) is None


def test_retrain_job_is_actually_scheduled():
    from models.retrain import start_scheduler, first_run_time
    sched = start_scheduler(interval_days=3)
    try:
        job = sched.get_job("retrain_all")
        assert job is not None and job.next_run_time is not None
    finally:
        sched.shutdown(wait=False)
    from datetime import datetime
    assert first_run_time(3) > datetime.now() + timedelta(days=2, hours=23)


def test_live_endpoint_is_blocked_without_opt_in(monkeypatch):
    import paper_trading.alpaca_paper as ap
    monkeypatch.setitem(ap.PAPER_MODE, "risky2", False)
    monkeypatch.setattr(ap, "ALLOW_LIVE_TRADING", False)
    with pytest.raises(ap.LiveTradingBlocked):
        ap.resolve_base_url("risky2")
    monkeypatch.setitem(ap.PAPER_MODE, "risky2", True)
    assert ap.resolve_base_url("risky2") == ap.ALPACA_PAPER_URL
    for s in ("stable", "risky1", "risky2"):
        assert ap.resolve_base_url(s) == "https://paper-api.alpaca.markets"


def test_ppo_live_observation_uses_latest_bar_and_real_position():
    from models.rl_environment import build_live_observation, TradingEnvironment, feature_columns
    df = make_featured_df(rows=30, price=100.0)
    df["rsi"] = np.linspace(10, 90, 30)                # distinguishes first and last rows
    obs = build_live_observation(df, position_value=50.0, entry_price=80.0, initial_capital=200.0)
    cols = feature_columns(df)
    assert obs[cols.index("rsi")] == pytest.approx(90.0)          # last row, not row 0
    assert obs[-2] == pytest.approx(0.25) and obs[-1] == pytest.approx((100.0 - 80.0) / 80.0)
    env = TradingEnvironment(df, initial_capital=200.0)
    env_obs, _ = env.reset()
    assert env_obs.shape == obs.shape                              # same layout as training
    assert env_obs[cols.index("rsi")] == pytest.approx(10.0)       # the env legitimately starts at row 0
    with pytest.raises(ValueError):
        build_live_observation(make_featured_df(rows=5, nan_last=True), 0, 0, 200.0)


def test_equity_curve_filter_ignores_kill_switch_fraction_rows(db_path):
    from metrics.equity_curve_filter import _fetch_equity_history
    log_trade("risky1", "TSLA", "SELL", 1, 1, pnl=-3.0, reason="t", db_path=db_path)
    log_trade("risky1", "ALL", "KILL_SWITCH", 0, 0, pnl=-0.9495, reason="t", db_path=db_path)
    log_trade("risky1", "TSLA", "SELL", 1, 1, pnl=2.0, reason="t", db_path=db_path)
    curve = _fetch_equity_history("risky1", db_path=db_path)
    assert curve == pytest.approx([197.0, 199.0])
