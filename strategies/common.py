"""
Shared trading engine for all strategies.

Before the audit, stable.py and risky1.py were two ~230-line copies of the same
loop and risky2.py was a third variant with no risk controls. Every one of
them sized, exited and kill-switched against the ACCOUNT-wide Alpaca state.
This module is the single implementation of the lifecycle:

    market data -> features -> strategy decision -> STRATEGY position (ledger)
    -> sizing -> risk (strategy equity, confirmed kill switch, ATR as fraction)
    -> anti-churn state machine -> execution (own quantity only, fills read back)
    -> trades.db / heartbeat / Discord

A `StrategyContext` describes what differs between strategies (assets, the
decision function, whether it observes market hours, momentum exit, order
time-in-force). Everything else is common and covered by tests/test_engine.py.
"""
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from core import portfolio
from core.config import (CAPITAL, MAX_POSITION_SIZE, MIN_HOLD_SECONDS, ONE_ACTION_PER_BAR,
                         KILL_SWITCH_CONFIRMATIONS, STRATEGY_ASSETS, MIN_TOPUP_FRACTION)
from core.execution import (submit_and_track, close_strategy_position, close_all_strategy_positions,
                            reconcile_open_orders, to_broker_symbol)
from core.logger import log_trade, log_heartbeat
from metrics.risk_manager import (atr_as_fraction, risk_level_from_drawdown, get_position_size,
                                  check_stop_loss, validate_equity)

MIN_ORDER_NOTIONAL = 1.0  # Alpaca minimum for fractional orders


@dataclass
class Decision:
    signal: str                      # "BUY" | "SELL" | "HOLD"
    model_output: object = None      # raw prediction / action for the audit log
    note: str = ""


@dataclass
class RiskSnapshot:
    equity: float
    peak: float
    drawdown: float
    level: str                       # safe | warning | critical | suspect
    kill_now: bool
    halted: bool
    reason: str = ""


@dataclass
class StrategyContext:
    name: str
    assets: list
    decide: Callable[["StrategyContext", str, object, float, float, float], Decision]
    fetch_frame: Callable[[str], object]            # ticker -> featured DataFrame (or None)
    uses_market_hours: bool = True
    momentum_exit: bool = False
    regime_gate: bool = True
    time_in_force: str = "day"
    per_ticker_sleep: float = 20.0
    cycle_sleep: float = 60.0
    macro_context: Callable[[], Optional[float]] = lambda: None   # returns effective VIX or None
    regime_ok: Callable[[object, str, Optional[float]], bool] = lambda df, s, vix: True
    notify: Callable[..., None] = lambda **kw: None                # Discord heartbeat
    alert: Callable[..., None] = lambda **kw: None                 # Discord alert
    db_path: Optional[str] = None
    exec_kw: dict = field(default_factory=dict)                    # e.g. sleep_fn/clock_fn for tests
    # in-memory only: last rejected equity reading, used to confirm persistent moves
    _last_suspect: Optional[float] = field(default=None, repr=False)


def utcnow():
    return datetime.now(timezone.utc)


def _parse_ts(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def bar_id(df):
    """Identifier of the bar a decision was made on (the last index label)."""
    try:
        return str(df.index[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Risk evaluation (strategy-level, confirmed kill switch)
# ---------------------------------------------------------------------------

def evaluate_risk(ctx, marks, now=None):
    """
    Computes THIS strategy's equity from its ledger, validates the reading,
    updates the persisted peak, and decides the risk level. The kill switch
    only fires after KILL_SWITCH_CONFIRMATIONS consecutive critical readings.
    """
    s = ctx.name
    state = portfolio.get_strategy_state(s, ctx.db_path)
    equity, pos_value, missing = portfolio.strategy_equity(s, marks, ctx.db_path)
    if missing:
        print(f"[{s}] no mark price for {missing}; valued at entry (conservative)")

    ok, why = validate_equity(equity, state["last_equity"])
    if not ok:
        # A persistent reading (two cycles agreeing) is reality, not a glitch.
        if ctx._last_suspect is not None and abs(equity - ctx._last_suspect) <= 0.05 * max(abs(equity), 1e-9):
            ok, why = True, "confirmed by consecutive reading"
        else:
            ctx._last_suspect = equity
            print(f"[{s}] SUSPECT equity reading {equity:.2f} ({why}); no entries, no kill switch this cycle")
            portfolio.record_equity_snapshot(s, equity, state["cash_budget"], pos_value,
                                             state["peak_equity"], float("nan"), "suspect", ctx.db_path)
            return RiskSnapshot(equity, state["peak_equity"], float("nan"), "suspect", False, state["halted"], why)
    ctx._last_suspect = None

    peak = max(state["peak_equity"], equity)
    drawdown = (equity - peak) / peak if peak > 0 else 0.0
    level = risk_level_from_drawdown(s, drawdown, None)
    streak = state["critical_streak"] + 1 if level == "critical" else 0
    kill_now = level == "critical" and streak >= KILL_SWITCH_CONFIRMATIONS
    portfolio.update_strategy_state(s, ctx.db_path, peak_equity=peak, last_equity=equity, critical_streak=streak)
    portfolio.record_equity_snapshot(s, equity, state["cash_budget"], pos_value, peak, drawdown, level, ctx.db_path)
    if level == "critical" and not kill_now:
        print(f"[{s}] CRITICAL drawdown {drawdown:.2%} (reading {streak}/{KILL_SWITCH_CONFIRMATIONS}); awaiting confirmation")
    elif level == "warning":
        print(f"[{s}] WARNING drawdown {drawdown:.2%} from peak ${peak:.2f}; reducing position size")
    return RiskSnapshot(equity, peak, drawdown, level, kill_now, state["halted"])


def fire_kill_switch(ctx, api, risk, marks):
    """Closes ONLY this strategy's positions and halts this strategy."""
    s = ctx.name
    print(f"[{s}] KILL SWITCH FIRED - drawdown {risk.drawdown:.2%} from peak ${risk.peak:.2f}")
    results = close_all_strategy_positions(api, s, marks, reason=f"KILL_SWITCH drawdown {risk.drawdown:.2%}",
                                           time_in_force=ctx.time_in_force, risk_state="critical", db_path=ctx.db_path,
                                           **ctx.exec_kw)
    log_trade(s, "ALL", "KILL_SWITCH", 0, 0, pnl=None,
              reason=f"Drawdown {risk.drawdown:.2%} from peak ${risk.peak:.2f}; closed {len(results)} own positions",
              risk_state="critical", db_path=ctx.db_path)
    portfolio.set_halted(s, True, f"kill switch at {utcnow().isoformat()} drawdown {risk.drawdown:.2%}", ctx.db_path)
    ctx.alert(bot_name=s, alert_type="KILL SWITCH",
              message=f"Kill switch triggered - {s} positions closed. Bot halted.", portfolio_value=risk.equity)
    return results


# ---------------------------------------------------------------------------
# Anti-churn state machine
# ---------------------------------------------------------------------------

def can_discretionary_exit(ctx, opened_at, current_bar, sig_state, now=None):
    """
    Non-emergency exits (ML SELL, momentum exit, PPO SELL) are allowed only if
    the position has been held for MIN_HOLD_SECONDS and, for bar-based
    strategies, the exit is not on the same bar as the entry.
    Emergency exits never call this.
    """
    now = now or utcnow()
    opened = _parse_ts(opened_at)
    min_hold = MIN_HOLD_SECONDS.get(ctx.name, 0)
    if opened is not None and (now - opened).total_seconds() < min_hold:
        return False, f"min hold {min_hold}s not met"
    if ONE_ACTION_PER_BAR.get(ctx.name) and sig_state.get("last_action") == "BUY" \
            and sig_state.get("last_bar") == current_bar and current_bar is not None:
        return False, "entered on this bar; no discretionary exit on the same bar"
    return True, "ok"


def can_enter(ctx, current_bar, sig_state, exit_condition_active):
    """
    Entries are refused when the strategy's own exit condition is already true
    on this bar (it would exit next cycle: the oscillation seen on 2026-08-27)
    or when the strategy already sold on this same bar.
    """
    if exit_condition_active:
        return False, "exit condition already true on this bar"
    same_bar = sig_state.get("last_bar") == current_bar and current_bar is not None
    if ONE_ACTION_PER_BAR.get(ctx.name) and same_bar and sig_state.get("last_action") == "SELL":
        return False, "already exited on this bar; no re-entry until a new bar"
    if same_bar and sig_state.get("last_action") == "BUY_REJECTED":
        return False, "entry was rejected by the broker on this bar; not retrying until a new bar"
    return True, "ok"


# ---------------------------------------------------------------------------
# Per-ticker lifecycle
# ---------------------------------------------------------------------------

def _last_float(df, col):
    try:
        v = float(df[col].iloc[-1])
    except Exception:
        return None
    return v if math.isfinite(v) else None


def trade_ticker(ctx, api, ticker, df, risk, multiplier=1.0, vix=None, allow_entries=True, now=None, **exec_kw):
    """
    Full lifecycle for one ticker. Returns a short string describing the outcome.
    Exits are ALWAYS evaluated (even when allow_entries is False, i.e. HALT).
    """
    s = ctx.name
    if df is None or len(df) == 0:
        print(f"[{s}] No data for {ticker}, skipping")
        return "no-data"

    price = _last_float(df, "close")
    if price is None or price <= 0:
        print(f"[{s}] invalid price for {ticker}; skipping")
        return "bad-price"
    atr_frac = atr_as_fraction(_last_float(df, "atr"), price)
    current_bar = bar_id(df)

    owned_qty, avg_entry, opened_at = portfolio.get_position(s, ticker, ctx.db_path)
    owned_value = owned_qty * price
    sig_state = portfolio.get_signal_state(s, ticker, ctx.db_path)
    log_kw = dict(risk_state=risk.level, time_in_force=ctx.time_in_force, db_path=ctx.db_path,
                  **{**ctx.exec_kw, **exec_kw})

    # 1. Emergency exits: stop loss on OUR entry price, or confirmed critical risk.
    if owned_qty > 0:
        if check_stop_loss(s, avg_entry, price) or risk.kill_now:
            reason = "Stop loss" if check_stop_loss(s, avg_entry, price) else "Portfolio risk critical"
            res = close_strategy_position(api, s, ticker, price, reason=f"{reason} (emergency exit)", **log_kw)
            portfolio.set_signal_state(s, ticker, current_bar, "SELL", ctx.db_path)
            print(f"[{s}] {reason}: closed {ticker} -> {res.status}")
            return "emergency-exit"

    # 2. Momentum exit (risky1): discretionary, subject to anti-churn rules.
    momentum_5 = _last_float(df, "momentum_5")
    exit_condition_active = bool(ctx.momentum_exit and momentum_5 is not None and momentum_5 < 0)
    if owned_qty > 0 and exit_condition_active:
        ok, why = can_discretionary_exit(ctx, opened_at, current_bar, sig_state, now)
        if ok:
            res = close_strategy_position(api, s, ticker, price, reason="Momentum turned negative", **log_kw)
            portfolio.set_signal_state(s, ticker, current_bar, "SELL", ctx.db_path)
            print(f"[{s}] Momentum exit {ticker} -> {res.status}")
            return "momentum-exit"
        print(f"[{s}] HOLD {ticker}: momentum exit deferred ({why})")
        return "hold-deferred"

    # 3. Regime / macro gate (applies to model decisions, as in the original design).
    if ctx.regime_gate and not ctx.regime_ok(df, s, vix):
        print(f"[{s}] {ticker} regime unfavorable, sitting out")
        return "regime-blocked"

    # 4. Strategy decision.
    try:
        decision = ctx.decide(ctx, ticker, df, owned_qty, avg_entry, price)
    except Exception as e:
        print(f"[{s}] decision failed for {ticker}: {e}")
        return "decision-error"
    if decision is None:
        return "no-model"

    # 5. Sizing against STRATEGY equity and peak.
    base_size = MAX_POSITION_SIZE[s] * multiplier
    max_pos = get_position_size(s, risk.equity, base_size, atr_frac, risk.peak)
    if risk.level in ("critical", "suspect"):
        max_pos = 0.0

    if decision.signal == "BUY":
        if not allow_entries or max_pos <= 0:
            print(f"[{s}] {ticker} BUY signal ignored: entries disabled (risk={risk.level}, entries={allow_entries})")
            return "entry-blocked"
        if owned_value >= max_pos * (1 - MIN_TOPUP_FRACTION):
            print(f"[{s}] HOLD {ticker}: at max position (${owned_value:.2f} vs ${max_pos:.2f})")
            return "at-max"
        ok, why = can_enter(ctx, current_bar, sig_state, exit_condition_active)
        if not ok:
            print(f"[{s}] {ticker} BUY skipped: {why}")
            return "entry-anti-churn"
        qty = round((max_pos - owned_value) / price, 4)
        if qty <= 0 or qty * price < MIN_ORDER_NOTIONAL:
            print(f"[{s}] Skipping {ticker}: order ${qty * price:.2f} below minimum")
            return "below-min"
        res = submit_and_track(api, s, ticker, "buy", qty, price,
                               reason=f"{decision.note or 'signal BUY'}", model_output=decision.model_output, **log_kw)
        if res.status == "rejected":
            # Do not hammer the broker with the same rejected entry every cycle on this bar
            # (second-pass S-07); a new bar, or a manual reset, re-enables the attempt.
            portfolio.set_signal_state(s, ticker, current_bar, "BUY_REJECTED", ctx.db_path)
        elif res.status != "refused":
            portfolio.set_signal_state(s, ticker, current_bar, "BUY", ctx.db_path)
        print(f"[{s}] BUY {qty} {ticker} @ signal ${price:.2f} -> {res.status} {res.message}")
        return f"buy-{res.status}"

    if decision.signal == "SELL" and owned_qty > 0:
        ok, why = can_discretionary_exit(ctx, opened_at, current_bar, sig_state, now)
        if not ok:
            print(f"[{s}] HOLD {ticker}: SELL deferred ({why})")
            return "hold-deferred"
        res = close_strategy_position(api, s, ticker, price, reason=f"{decision.note or 'signal SELL'}",
                                      model_output=decision.model_output, **log_kw)
        portfolio.set_signal_state(s, ticker, current_bar, "SELL", ctx.db_path)
        print(f"[{s}] SELL {ticker} @ signal ${price:.2f} -> {res.status}")
        return f"sell-{res.status}"

    print(f"[{s}] HOLD {ticker}")
    return "hold"


# ---------------------------------------------------------------------------
# Cycle and run loop
# ---------------------------------------------------------------------------

def adopt_unowned_broker_positions(ctx, api):
    """
    Migration helper for databases created before the ledger existed. If the
    strategy's ledger is empty and the broker holds symbols from this
    strategy's universe (and no other strategy's), they are adopted only when
    SINGHQUANT_ADOPT_BROKER_POSITIONS=1; otherwise they are reported and left
    alone so no strategy touches inventory it cannot prove it owns.
    """
    s = ctx.name
    owned = portfolio.list_positions(s, ctx.db_path)
    others = {sym for name, syms in STRATEGY_ASSETS.items() if name != s for sym in syms}
    adopt = os.getenv("SINGHQUANT_ADOPT_BROKER_POSITIONS", "") == "1"
    adopted = []
    try:
        broker_positions = api.list_positions()
    except Exception as e:
        print(f"[{s}] could not list broker positions: {e}")
        return []
    for p in broker_positions:
        broker_sym = str(p.symbol).replace("/", "")   # Alpaca may report crypto as BTC/USD or BTCUSD
        matches = [t for t in ctx.assets if to_broker_symbol(t) == broker_sym]
        if not matches or any(t in others for t in matches):
            continue
        ticker = matches[0]
        if ticker in owned:
            continue   # already in this strategy's ledger; only unowned symbols are candidates
        qty, avg = float(p.qty), float(p.avg_entry_price)
        if adopt:
            portfolio.adopt_position(s, ticker, qty, avg, ctx.db_path)
            adopted.append((ticker, qty))
            print(f"[{s}] adopted broker position {qty} {ticker} @ {avg}")
        else:
            print(f"[{s}] NOTE: broker holds {qty} {ticker} not in this strategy's ledger. "
                  f"Set SINGHQUANT_ADOPT_BROKER_POSITIONS=1 to adopt it, otherwise it is left untouched.")
    return adopted


def run_cycle(ctx, api, sleep_fn=time.sleep, now=None):
    """
    One full cycle for a strategy. Returns a dict summarising what happened so
    tests (and the caller) can inspect it. Never raises for per-ticker errors.
    """
    s = ctx.name
    summary = {"strategy": s, "tickers": {}, "paused": 0, "killed": False, "halted": False}
    portfolio.ensure_strategy_state(s, ctx.db_path)
    try:
        reconcile_open_orders(api, s, ctx.db_path)
    except Exception as e:  # reconciliation must never take the whole cycle down (second-pass S-04)
        print(f"[{s}] order reconciliation failed: {e}")

    if ctx.uses_market_hours:
        from paper_trading.alpaca_paper import sleep_seconds_until_open
        secs = sleep_seconds_until_open(api.get_clock())
        if secs > 0:
            log_heartbeat(s, "PAUSED", ctx.db_path)
            summary["paused"] = secs
            return summary

    # Symbols to manage: the configured universe PLUS anything the ledger still
    # owns (a symbol removed from config must still be marked, stopped out and
    # exited; second-pass S-08). Entries are only allowed for configured assets.
    owned = list(portfolio.list_positions(s, ctx.db_path).keys())
    symbols = list(ctx.assets) + [t for t in owned if t not in ctx.assets]

    # Fetch every frame first so the strategy is valued on one consistent set of
    # marks. The per-ticker sleep lives HERE: Polygon's free tier allows about
    # five requests a minute, and the first-pass code fetched all frames back to
    # back (second-pass S-02, a regression against the original 20 s spacing).
    frames = {}
    for i, ticker in enumerate(symbols):
        try:
            frames[ticker] = ctx.fetch_frame(ticker)
        except Exception as e:
            print(f"[{s}] data error for {ticker}: {e}")
            frames[ticker] = None
        log_heartbeat(s, "RUNNING", ctx.db_path)
        if ctx.per_ticker_sleep and i < len(symbols) - 1:
            sleep_fn(ctx.per_ticker_sleep)
    marks = {t: _last_float(df, "close") for t, df in frames.items() if df is not None and len(df)}
    marks = {t: p for t, p in marks.items() if p and p > 0}

    risk = evaluate_risk(ctx, marks, now)
    summary["risk"] = risk
    if risk.halted:
        log_heartbeat(s, "HALTED", ctx.db_path)
        summary["halted"] = True
        # Keep trying to flatten anything the kill switch could not close
        # (rejected or pending sells): a halted strategy must not carry open risk
        # with no management (second-pass S-09).
        if owned:
            results = close_all_strategy_positions(api, s, marks, reason="KILL_SWITCH follow-up liquidation",
                                                   time_in_force=ctx.time_in_force, risk_state="halted",
                                                   db_path=ctx.db_path, **ctx.exec_kw)
            summary["halted_liquidation"] = [(r.symbol, r.status) for r in results]
        print(f"[{s}] halted by kill switch; run `python tools/reset_kill_switch.py {s}` after review")
        return summary
    if risk.kill_now:
        fire_kill_switch(ctx, api, risk, marks)
        summary["killed"] = True
        log_heartbeat(s, "HALTED", ctx.db_path)
        return summary

    from metrics.equity_curve_filter import get_trading_state
    state, multiplier = get_trading_state(s, ctx.db_path)
    allow_entries = state != "HALT" and risk.level in ("safe", "warning")
    summary["trading_state"] = state
    vix = ctx.macro_context()

    for ticker in symbols:
        try:
            summary["tickers"][ticker] = trade_ticker(ctx, api, ticker, frames.get(ticker), risk, multiplier, vix,
                                                      allow_entries and ticker in ctx.assets, now)
        except Exception as e:
            print(f"[{s}] Error trading {ticker}: {e}")
            summary["tickers"][ticker] = f"error: {e}"
        log_heartbeat(s, "RUNNING", ctx.db_path)

    log_heartbeat(s, "RUNNING", ctx.db_path)
    return summary


def run_forever(ctx, api=None, sleep_fn=time.sleep, max_cycles=None):
    """Main loop used by run.py threads. `max_cycles` is for tests."""
    from paper_trading.alpaca_paper import get_api
    s = ctx.name
    api = api or get_api(s)
    print(f"{s} bot started (paper endpoint check: {getattr(api, '_base_url', 'n/a')})")
    portfolio.ensure_strategy_state(s, ctx.db_path)
    adopt_unowned_broker_positions(ctx, api)
    last_trade_time = utcnow()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            summary = run_cycle(ctx, api, sleep_fn)
            if summary["paused"]:
                secs = summary["paused"]
                print(f"[{s}] Market is closed - sleeping for {secs // 3600}h {(secs % 3600) // 60}m")
                _safe_notify(ctx, api, s, "Market closed", f"Sleeping {secs // 3600}h - market closed")
                sleep_fn(secs)
                continue
            if summary["halted"] or summary["killed"]:
                sleep_fn(300)
                continue
            if any(str(v).startswith(("buy", "sell", "momentum", "emergency")) for v in summary["tickers"].values()):
                last_trade_time = utcnow()
            print(f"[{s}] Cycle completed at {utcnow()} - sleeping {ctx.cycle_sleep:.0f}s")
            _safe_notify(ctx, api, s, str(last_trade_time), "Cycle complete.")
            sleep_fn(ctx.cycle_sleep)
        except KeyboardInterrupt:
            print(f"{s} bot manually stopped")
            break
        except Exception as e:
            print(f"[{s}] Unexpected error: {e} - restarting in 60 seconds")
            sleep_fn(60)


def _safe_notify(ctx, api, s, last_trade_time, extra):
    try:
        account = api.get_account()
        ctx.notify(bot_name=s, is_alive=True, portfolio_value=float(account.portfolio_value),
                   last_trade_time=last_trade_time, last_sync_time=str(utcnow()), extra_info=extra)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Default data / macro helpers shared by the concrete strategies
# ---------------------------------------------------------------------------

def make_frame_fetcher(api_key):
    """Polygon daily bars -> features -> sentiment, for one ticker."""
    def fetch(ticker):
        from data.polygon_fetcher import get_latest_bar
        from core.features import build_features
        from data.sentiment_fetcher import add_sentiment_to_df
        df = get_latest_bar(ticker, api_key=api_key)
        if df is None or df.empty:
            return None
        df = build_features(df)
        df = add_sentiment_to_df(df, ticker)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df if len(df) else None
    return fetch


def effective_vix():
    """VIX from FRED, raised by the macro circuit breaker (DANGER=35, CAUTION=22)."""
    from models.regime_detector import get_vix
    from data.macro_fetcher import get_macro_signal
    vix = get_vix()
    macro = get_macro_signal()
    if macro == "DANGER":
        vix = max(vix or 0, 35)
    elif macro == "CAUTION":
        vix = max(vix or 0, 22)
    else:
        vix = vix or 0
    return vix if vix > 0 else None


def xgb_decider(model_handle):
    """Builds a decide() for the XGBoost strategies from a ModelHandle."""
    from models.train import features_for_model
    def decide(ctx, ticker, df, owned_qty, avg_entry, price):
        model = model_handle.model
        if model is None:
            return None
        X = features_for_model(df.tail(1), model)
        pred = int(model.predict(X)[0])
        if pred == 1:
            return Decision("BUY", pred, "ML signals BUY, regime favorable")
        if pred == 0:
            return Decision("SELL", pred, "ML signals SELL")
        return Decision("HOLD", pred)
    return decide
