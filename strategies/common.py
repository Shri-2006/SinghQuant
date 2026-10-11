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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from core import lifecycle, portfolio
from core.config import (CAPITAL, MAX_POSITION_SIZE, MIN_HOLD_SECONDS, ONE_ACTION_PER_BAR,
                         KILL_SWITCH_CONFIRMATIONS, STRATEGY_ASSETS, MIN_TOPUP_FRACTION,
                         MARK_MAX_STEP_CHANGE, STALE_BAR_MAX_DAYS, MARK_CROSSCHECK_TOLERANCE,
                         SUSPECT_ALERT_AFTER_CYCLES, SUSPECT_ALERT_REPEAT_CYCLES, MODEL_ALERT_REPEAT_SECONDS,
                         PAUSED_HEARTBEAT_SECONDS)
from core.execution import (submit_and_track, close_strategy_position, close_all_strategy_positions,
                            reconcile_open_orders, to_broker_symbol, BrokerQuote, broker_quote,
                            record_discrepancy)
from core.logger import log_trade, log_heartbeat
from metrics.risk_manager import (atr_as_fraction, risk_level_from_drawdown, get_position_size,
                                  check_stop_loss, validate_equity, portfolio_atr_fraction)

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
    accepted_marks: dict = field(default_factory=dict)   # validated marks used for valuation this cycle
    suspect_symbols: list = field(default_factory=list)  # symbols whose mark could not be validated
    atr_fraction: Optional[float] = None                 # strategy-level ATR fraction used for thresholds
    stale_symbols: list = field(default_factory=list)    # symbols whose latest bar is too old for entries
    discrepancy_symbols: list = field(default_factory=list)  # held symbols wholly flagged as missing at broker (R-1)
    suspect_streak: int = 0                              # consecutive suspect readings including this one (R-1)
    prev_suspect_streak: int = 0                         # streak before this reading (recovery alerts)


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
    # Also require the shared ACCOUNT's cash to cover a buy (restores main's risky2
    # guard, commit 523f7f6: crypto cannot be bought on margin). Fails closed. (R-3)
    check_account_cash: bool = False
    runtime: dict = field(default_factory=dict)                    # per-process state (alert de-duplication)
    # Phase 2 I-3: () -> (ok, reason). ok=False means the strategy's model
    # artifacts do not validate: new entries are blocked (the decider gets no
    # model) while stops, momentum exits and the kill switch keep running.
    model_status: Callable[[], tuple] = lambda: (True, "")
    # Phase 2 I-4: set by run.py on SIGTERM. Sleeps wake up, the cycle stops
    # between tickers (never in the middle of an order) and run_forever returns.
    stop_event: Optional[threading.Event] = None


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


def _bar_age_days(df, now=None):
    """Age in days of the latest bar (None if the index is not a timestamp)."""
    try:
        import pandas as pd
        ts = pd.Timestamp(df.index[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ref = now or utcnow()
        if getattr(ref, "tzinfo", None) is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return (pd.Timestamp(ref) - ts).total_seconds() / 86400.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Risk evaluation (strategy-level, confirmed kill switch)
# ---------------------------------------------------------------------------

def _quote(verify_mark, symbol):
    """
    Normalises a verify_mark result into a BrokerQuote. verify_mark may return
    a BrokerQuote, a price (treated as "ok"), or None / raise (treated as
    "unknown": nothing can be inferred from a failed lookup).
    """
    if verify_mark is None:
        return BrokerQuote("unknown")
    try:
        q = verify_mark(symbol)
    except Exception:
        return BrokerQuote("unknown")
    if isinstance(q, BrokerQuote):
        return q
    try:
        q = float(q) if q is not None else None
    except (TypeError, ValueError):
        q = None
    if q is not None and math.isfinite(q) and q > 0:
        return BrokerQuote("ok", None, q)
    return BrokerQuote("unknown")


def _wholly_flagged(ctx):
    """Held symbols whose ENTIRE ledger quantity is flagged as missing at the broker."""
    out = set()
    for (_, sym), d in portfolio.get_discrepancies(ctx.name, ctx.db_path).items():
        if d["qty"] - d["missing_qty"] <= 1e-9:
            out.add(sym)
    return out


def _flag_absent(ctx, symbol, owned_qty, mark, risk_state=None):
    """
    The broker authoritatively reports no position for a symbol this strategy
    owns. Flag it (never write it off: owner decision D1-B) unless one of this
    strategy's orders for the symbol is still open (an in-flight sell explains
    the gap until it is reconciled). Returns True when the symbol is now
    treated as wholly flagged.
    """
    if portfolio.open_orders(ctx.name, symbol, db_path=ctx.db_path):
        return False
    others = portfolio.total_ledger_qty(symbol, exclude_strategy=ctx.name, db_path=ctx.db_path)
    record_discrepancy(ctx.name, symbol, owned_qty, mark, "broker reports no position", risk_state,
                       owned_qty, 0.0, others, ctx.db_path)
    return True


def _validate_marks(ctx, marks, bar_ages, verify_mark):
    """
    Per-symbol INPUT validation (third pass T-06, extended by remediation R-1).

    Missing or implausible primary marks are cross-checked against an
    independent broker quote. A valid broker quote is used for the cycle.
    If the broker AUTHORITATIVELY reports the position absent, the symbol is
    flagged as a discrepancy and valued conservatively (never suspect: a
    vanished position must not be able to switch the kill switch off). If the
    broker state is UNKNOWN, the symbol is SUSPECT: its last accepted mark /
    entry may still be used for valuation, but no new entry, peak update, or
    kill-switch action is taken from that untrusted reading.

    Returns (accepted_marks, suspect_symbols, flagged_symbols, notes).
    """
    s = ctx.name
    positions = portfolio.list_positions(s, ctx.db_path)
    last_marks = portfolio.get_last_marks(s, ctx.db_path)
    flagged = _wholly_flagged(ctx)
    accepted = dict(marks)
    suspect, notes = [], []

    for symbol, (qty, avg, _) in positions.items():
        px = marks.get(symbol)
        ref = last_marks.get(symbol) or avg
        age = (bar_ages or {}).get(symbol)
        problems = []
        if px is None:
            problems.append("primary mark missing")
        else:
            if ref and ref > 0 and abs(px - ref) / ref > MARK_MAX_STEP_CHANGE:
                problems.append(f"mark {px:.4f} moved {abs(px - ref) / ref:.0%} from last accepted {ref:.4f}")
            if age is not None and age > STALE_BAR_MAX_DAYS:
                problems.append(f"bar is {age:.1f} days old")
        if not problems:
            continue

        if symbol in flagged:
            # Already known to be missing at the broker: there is no broker quote to
            # check against. Drop the doubtful mark; valuation falls back to the
            # conservative min(last accepted mark, entry).
            accepted.pop(symbol, None)
            notes.append(f"{symbol}: {'; '.join(problems)}; flagged discrepancy, valued conservatively")
            continue

        quote = _quote(verify_mark, symbol)
        if quote.status == "ok" and quote.price is not None:
            if px is not None and abs(quote.price - px) / quote.price <= MARK_CROSSCHECK_TOLERANCE:
                notes.append(f"{symbol}: {'; '.join(problems)}; independent quote {quote.price:.4f} agrees, accepted")
            else:
                accepted[symbol] = quote.price
                notes.append(f"{symbol}: {'; '.join(problems)}; using independent quote {quote.price:.4f}")
            continue
        if quote.status == "absent" and _flag_absent(ctx, symbol, qty, px or ref):
            flagged.add(symbol)
            accepted.pop(symbol, None)
            notes.append(f"{symbol}: {'; '.join(problems)}; broker reports no position, flagged discrepancy")
            continue
        accepted.pop(symbol, None)
        suspect.append(symbol)
        notes.append(f"{symbol}: {'; '.join(problems)}; no independent quote, SUSPECT")
    return accepted, suspect, sorted(flagged), notes


def _position_values(ctx, held, accepted, flagged):
    """
    Position values that weight the strategy-level ATR (T-02). Flagged
    positions are included at their conservative price so that flagging a
    discrepancy never changes the kill/warning thresholds by itself (R-1).
    """
    last_marks = portfolio.get_last_marks(ctx.name, ctx.db_path)
    values = {}
    for t, (q, avg, _) in held.items():
        if t in flagged:
            px = portfolio.conservative_price(accepted.get(t), last_marks.get(t), avg)
        else:
            px = accepted.get(t)
        if px:
            values[t] = q * px
    return values


def _record_suspect(ctx, state, equity, pos_value, reason, accepted, suspect, flagged, atr_frac, stale):
    s = ctx.name
    prev = state.get("suspect_streak", 0)
    streak = prev + 1
    print(f"[{s}] SUSPECT reading ({reason}); no entries, no kill switch this cycle (suspect streak {streak})")
    portfolio.update_strategy_state(s, ctx.db_path, suspect_streak=streak)
    portfolio.record_equity_snapshot(s, equity, state["cash_budget"], pos_value,
                                     state["peak_equity"], None, "suspect", ctx.db_path)
    return RiskSnapshot(equity, state["peak_equity"], float("nan"), "suspect", False, state["halted"], reason,
                        accepted, suspect, atr_frac, stale, flagged, streak, prev)


def evaluate_risk(ctx, marks, now=None, atr_fractions=None, bar_ages=None, verify_mark=None):
    """
    Computes THIS strategy's equity from its ledger, validates the INPUT marks
    (T-06), updates the persisted peak, and decides the risk level with the
    strategy-level ATR-adjusted thresholds (T-02). The kill switch only fires
    after KILL_SWITCH_CONFIRMATIONS consecutive critical readings.

    Two independent guards, on purpose:
      * input validation (per mark, cross-checked against the broker) decides
        whether the equity number can be trusted at all;
      * state confirmation (consecutive critical readings) decides whether a
        trusted critical equity is acted on.

    Remediation R-1: a held symbol the broker AUTHORITATIVELY reports absent
    is flagged and valued conservatively; it never makes the reading suspect,
    so it can never disable the kill switch. Only UNKNOWN broker state does,
    and consecutive suspect readings are counted for alerting.
    """
    s = ctx.name
    state = portfolio.get_strategy_state(s, ctx.db_path)
    accepted, suspect, flagged, notes = _validate_marks(ctx, marks, bar_ages, verify_mark)
    for n in notes:
        print(f"[{s}] mark check: {n}")
    stale = sorted(t for t, a in (bar_ages or {}).items() if a is not None and a > STALE_BAR_MAX_DAYS)
    equity, pos_value, missing = portfolio.strategy_equity(s, accepted, ctx.db_path)
    if missing:
        print(
            f"[{s}] no validated mark for {missing}; "
            "using last accepted mark / entry for fallback valuation only"
        )

    structurally_ok = isinstance(equity, (int, float)) and math.isfinite(equity)
    if suspect or not structurally_ok:
        reason = f"unvalidated marks for {suspect}" if suspect else "equity not finite"
        return _record_suspect(ctx, state, equity if structurally_ok else 0.0, pos_value, reason,
                               accepted, suspect, flagged, None, stale)

    portfolio.set_last_marks(s, {t: p for t, p in accepted.items() if t not in flagged}, ctx.db_path)
    portfolio.lower_last_marks(s, {t: p for t, p in accepted.items() if t in flagged}, ctx.db_path)
    held = portfolio.list_positions(s, ctx.db_path)
    position_values = _position_values(ctx, held, accepted, flagged)
    atr_frac = portfolio_atr_fraction(position_values, atr_fractions or {})

    peak = max(state["peak_equity"], equity)
    drawdown = (equity - peak) / peak if peak > 0 else 0.0
    level = risk_level_from_drawdown(s, drawdown, atr_frac)

    if level == "critical" and verify_mark is not None and held:
        # A CRITICAL state is acted on only when an independent quote corroborates
        # every held mark (T-06): a bad print inside the step limit must not be
        # able to confirm itself across cycles. Disagreeing quotes replace the
        # marks; an UNKNOWN quote makes the reading suspect (fail closed). A symbol
        # the broker authoritatively reports absent is flagged and valued
        # conservatively instead: it no longer blocks the kill switch (R-1).
        corrected, unverifiable = {}, []
        for t in held:
            if t in flagged or t not in accepted:
                continue
            q = _quote(verify_mark, t)
            if q.status == "absent":
                if _flag_absent(ctx, t, held[t][0], accepted[t], risk_state="critical"):
                    flagged = sorted(set(flagged) | {t})
                    continue
                unverifiable.append(t)          # open order in flight: cannot judge this cycle
            elif q.status != "ok" or q.price is None:
                unverifiable.append(t)
            elif abs(q.price - accepted[t]) / q.price > MARK_CROSSCHECK_TOLERANCE:
                corrected[t] = q.price
        if unverifiable:
            reason = f"critical reading could not be corroborated for {unverifiable}"
            return _record_suspect(ctx, state, equity, pos_value, reason, accepted, unverifiable, flagged,
                                   atr_frac, stale)
        for t, q in corrected.items():
            print(f"[{s}] mark check: {t} critical-state corroboration disagrees ({accepted[t]:.4f} vs {q:.4f}); using quote")
        accepted.update(corrected)
        portfolio.set_last_marks(s, {t: p for t, p in accepted.items() if t not in flagged}, ctx.db_path)
        equity, pos_value, missing = portfolio.strategy_equity(s, accepted, ctx.db_path)
        position_values = _position_values(ctx, held, accepted, flagged)
        atr_frac = portfolio_atr_fraction(position_values, atr_fractions or {})
        peak = max(state["peak_equity"], equity)
        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        level = risk_level_from_drawdown(s, drawdown, atr_frac)

    prev_suspect = state.get("suspect_streak", 0)
    streak = state["critical_streak"] + 1 if level == "critical" else 0
    kill_now = level == "critical" and streak >= KILL_SWITCH_CONFIRMATIONS
    portfolio.update_strategy_state(s, ctx.db_path, peak_equity=peak, last_equity=equity, critical_streak=streak,
                                    suspect_streak=0)
    portfolio.record_equity_snapshot(s, equity, state["cash_budget"], pos_value, peak, drawdown, level, ctx.db_path)
    if flagged:
        print(f"[{s}] DISCREPANCY: {flagged} flagged as missing at broker; valued conservatively, awaiting manual reconciliation")
    if level == "critical" and not kill_now:
        print(f"[{s}] CRITICAL drawdown {drawdown:.2%} (reading {streak}/{KILL_SWITCH_CONFIRMATIONS}); awaiting confirmation")
    elif level == "warning":
        print(f"[{s}] WARNING drawdown {drawdown:.2%} from peak ${peak:.2f}; reducing position size")
    return RiskSnapshot(equity, peak, drawdown, level, kill_now, state["halted"], "", accepted, [], atr_frac, stale,
                        list(flagged), 0, prev_suspect)


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
    if ticker in getattr(risk, "suspect_symbols", ()):
        # The bar's price could not be validated and no independent quote exists:
        # act on nothing for this symbol (no stop, no signal) this cycle (T-06).
        print(f"[{s}] {ticker}: mark could not be validated this cycle; no action")
        return "suspect-mark"
    # Risk decisions use the validated mark, which may be the broker quote when
    # the bar disagreed with it.
    price = (getattr(risk, "accepted_marks", None) or {}).get(ticker, price)
    atr_frac = atr_as_fraction(_last_float(df, "atr"), price)
    current_bar = bar_id(df)

    owned_qty, avg_entry, opened_at = portfolio.get_position(s, ticker, ctx.db_path)
    flagged_qty = portfolio.get_discrepancy_qty(s, ticker, ctx.db_path) if owned_qty > 0 else 0.0
    if owned_qty > 0 and owned_qty - flagged_qty <= 1e-9:
        # Nothing of this position is at the broker: there is nothing to stop out
        # or sell, and buying more would stack on an unreconciled ledger (R-1).
        print(f"[{s}] {ticker}: {owned_qty} flagged as missing at broker; no action until manually reconciled")
        return "discrepancy-flagged"
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
    # The strategy-level decision (T-02) is authoritative: a per-ticker ATR may
    # widen a ticker's own threshold, but it can never undo a strategy-level
    # warning or critical state.
    if risk.level == "warning":
        max_pos = min(max_pos, base_size / 2)
    if risk.level in ("critical", "suspect"):
        max_pos = 0.0

    if decision.signal == "BUY":
        if ticker in getattr(risk, "stale_symbols", ()):
            print(f"[{s}] {ticker} BUY signal ignored: latest bar is stale")
            return "stale-bar"
        if not allow_entries or max_pos <= 0:
            print(f"[{s}] {ticker} BUY signal ignored: entries disabled (risk={risk.level}, entries={allow_entries})")
            return "entry-blocked"
        if flagged_qty > 0:
            print(f"[{s}] {ticker} BUY signal ignored: {flagged_qty} of the position is flagged as missing at broker")
            return "entry-blocked-discrepancy"
        if owned_value >= max_pos * (1 - MIN_TOPUP_FRACTION):
            print(f"[{s}] HOLD {ticker}: at max position (${owned_value:.2f} vs ${max_pos:.2f})")
            return "at-max"
        ok, why = can_enter(ctx, current_bar, sig_state, exit_condition_active)
        if not ok:
            print(f"[{s}] {ticker} BUY skipped: {why}")
            return "entry-anti-churn"
        # Strict per-strategy budget (remediation R-3): never size beyond the cash this
        # strategy has left after its open buy orders. execution.submit_and_track
        # re-checks atomically when it reserves the order.
        budget = portfolio.available_budget(s, ctx.db_path)
        target_value = max_pos - owned_value
        capped = target_value > budget
        qty = round(min(target_value, budget) / price, 4)
        if qty * price > budget:
            qty = math.floor(budget / price * 1e4) / 1e4      # rounding must never exceed the budget
        if qty <= 0 or qty * price < MIN_ORDER_NOTIONAL:
            if capped:
                print(f"[{s}] Skipping {ticker}: strategy budget exhausted (available ${budget:.2f})")
                return "budget-exhausted"
            print(f"[{s}] Skipping {ticker}: order ${qty * price:.2f} below minimum")
            return "below-min"
        if ctx.check_account_cash:
            try:
                account_cash = float(api.get_account().cash)
            except Exception as e:
                print(f"[{s}] {ticker} BUY skipped: account cash unknown ({e})")
                return "account-cash-unknown"
            if account_cash < qty * price:
                print(f"[{s}] {ticker} BUY skipped: insufficient account cash (${account_cash:.2f} < ${qty * price:.2f})")
                return "insufficient-account-cash"
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
        if portfolio.open_orders(s, ticker, db_path=ctx.db_path):
            # An order for it is still open locally: its fill will be (or was just)
            # booked by reconciliation. Adopting too would count it twice (Phase 2 I-4).
            print(f"[{s}] NOTE: broker holds {p.qty} {ticker} but an order for it is still open; not adopting")
            continue
        qty, avg = float(p.qty), float(p.avg_entry_price)
        if adopt:
            portfolio.adopt_position(s, ticker, qty, avg, ctx.db_path)
            adopted.append((ticker, qty))
            print(f"[{s}] adopted broker position {qty} {ticker} @ {avg}")
        else:
            print(f"[{s}] NOTE: broker holds {qty} {ticker} not in this strategy's ledger. "
                  f"Set SINGHQUANT_ADOPT_BROKER_POSITIONS=1 to adopt it, otherwise it is left untouched.")
    return adopted


def refresh_discrepancies(ctx, api):
    """
    Compares every held ledger position with the broker once per cycle
    (remediation R-1). Returns {symbol: BrokerQuote}, reused as the cycle's
    independent quote source.
      * broker UNKNOWN -> nothing changes (no flag set or cleared);
      * an open order for the symbol -> skipped (an in-flight fill explains a gap);
      * broker holds less than this strategy's claim after every other
        strategy's unflagged claim -> the shortfall is FLAGGED (ledger preserved);
      * broker holds the full claim again -> an existing flag is cleared.
    """
    s = ctx.name
    quotes = {}
    for symbol, (qty, avg, _) in portfolio.list_positions(s, ctx.db_path).items():
        q = broker_quote(api, to_broker_symbol(symbol))
        quotes[symbol] = q
        if q.status == "unknown" or portfolio.open_orders(s, symbol, db_path=ctx.db_path):
            continue
        others = portfolio.total_ledger_qty(symbol, exclude_strategy=s, db_path=ctx.db_path)
        available = max(0.0, (q.qty or 0.0) - others)
        missing = qty - available
        if missing > 1e-9 * max(1.0, qty):
            reason = "broker reports no position" if q.status == "absent" else "broker holds less than the ledger"
            record_discrepancy(s, symbol, missing, q.price or avg, reason, None, qty, q.qty, others, ctx.db_path)
        elif portfolio.clear_discrepancy(s, symbol, ctx.db_path):
            print(f"[{s}] {symbol}: broker holds the full ledger quantity again; discrepancy flag cleared")
            log_trade(s, symbol, "RECONCILE", q.price or avg, qty, pnl=None, status="discrepancy_cleared",
                      reason=f"broker holds {q.qty}, ledger {qty}, others {others}", db_path=ctx.db_path)
    return quotes


def _safe_alert(ctx, alert_type, message, equity):
    try:
        ctx.alert(bot_name=ctx.name, alert_type=alert_type, message=message,
                  portfolio_value=float(equity) if equity is not None and math.isfinite(equity) else 0.0)
    except Exception as e:
        print(f"[{ctx.name}] alert failed: {e}")


def _alert_new_discrepancies(ctx, equity):
    """One alert per new (or changed) flag; the flag stays visible until reconciled."""
    for (_, symbol), d in portfolio.get_discrepancies(ctx.name, ctx.db_path).items():
        if d["alerted"]:
            continue
        _safe_alert(ctx, "LEDGER DISCREPANCY",
                    f"{ctx.name} ledger owns {d['qty']} {symbol} but {d['missing_qty']} is not held at the broker "
                    f"({d['reason']}). Ledger preserved; valued conservatively. Resolve with "
                    f"`python tools/reconcile_position.py`.", equity)
        portfolio.mark_discrepancy_alerted(ctx.name, symbol, ctx.db_path)


def _alert_suspect_state(ctx, risk):
    """Alert-only escalation for persistent suspect readings (owner decision D2)."""
    n = risk.suspect_streak
    if risk.level == "suspect":
        if n == SUSPECT_ALERT_AFTER_CYCLES or (n > SUSPECT_ALERT_AFTER_CYCLES and
                                               (n - SUSPECT_ALERT_AFTER_CYCLES) % SUSPECT_ALERT_REPEAT_CYCLES == 0):
            _safe_alert(ctx, "RISK STATE SUSPECT",
                        f"{ctx.name} risk reading has been suspect for {n} consecutive cycles ({risk.reason}). "
                        "New entries are blocked and the kill switch cannot act on unverified data. "
                        "No positions are being liquidated; check broker/API connectivity.", risk.equity)
    elif risk.prev_suspect_streak >= SUSPECT_ALERT_AFTER_CYCLES:
        _safe_alert(ctx, "RISK STATE RECOVERED",
                    f"{ctx.name} risk reading is valid again after {risk.prev_suspect_streak} suspect cycles.",
                    risk.equity)


def _alert_negative_budget(ctx, equity):
    """
    Market orders fill at the live price while sizing uses the signal price, so
    slippage can push cash_budget slightly below zero. Sizing then blocks every
    new buy (available <= 0) until it recovers; alert once per episode (R-3).
    """
    cash = portfolio.get_strategy_state(ctx.name, ctx.db_path)["cash_budget"]
    if cash < -0.01:
        if not ctx.runtime.get("negative_budget_alerted"):
            _safe_alert(ctx, "BUDGET OVERDRAWN",
                        f"{ctx.name} cash budget is ${cash:.2f} (fill slippage above the signal price). "
                        "New buys are blocked until it is positive again.", equity)
            ctx.runtime["negative_budget_alerted"] = True
    else:
        ctx.runtime["negative_budget_alerted"] = False


def artifact_status(*handles):
    """model_status callable for StrategyContext over one or more verified artifact handles."""
    def status():
        problems = [f"{h.name}: {h.status} ({h.detail})" for h in handles if not h.ready]
        if problems:
            return False, f"{len(problems)}/{len(handles)} model artifact(s) unusable: " + "; ".join(problems)
        return True, ""
    return status


def _check_models(ctx, now=None):
    """Evaluates the model artifacts and alerts on loss (repeated) and on recovery (Phase 2 I-3)."""
    try:
        ok, reason = ctx.model_status()
    except Exception as e:
        ok, reason = False, f"model status check failed: {e}"
    now = now or utcnow()
    last = ctx.runtime.get("model_alert_at")
    if not ok:
        if last is None or (now - last).total_seconds() >= MODEL_ALERT_REPEAT_SECONDS:
            _safe_alert(ctx, "MODEL UNAVAILABLE",
                        f"{ctx.name}: {reason}. NEW ENTRIES ARE BLOCKED for the affected model(s); stop losses, "
                        "momentum exits and the kill switch keep managing existing positions. Publish or register "
                        "the model (tools/model_manifest.py) to restore entries.", None)
            ctx.runtime["model_alert_at"] = now
    elif last is not None:
        _safe_alert(ctx, "MODEL RESTORED", f"{ctx.name}: all model artifacts verify again; entries re-enabled.", None)
        ctx.runtime.pop("model_alert_at", None)
    return ok, reason


def _stopping(ctx):
    return ctx.stop_event is not None and ctx.stop_event.is_set()


def _entry_gates(ctx):
    """
    Process-level reasons NEW entries are blocked (Phase 2 I-4). Exits, stops
    and the kill switch are never gated by these.
      * startup reconciliation has not completed since this process started;
      * persistent SAFE MODE after a crash loop (survives restarts).
    """
    reasons = []
    if not ctx.runtime.get("startup_reconciled"):
        reasons.append("startup reconciliation pending")
    try:
        safe = lifecycle.get_safe_mode(ctx.db_path)
    except Exception as e:                      # unknown state must not allow entries
        safe = {"reason": f"safe-mode state unreadable: {e}"}
    if safe:
        reasons.append(f"SAFE MODE: {safe.get('reason')}")
    return reasons


def _readiness(model_ok, model_reason, risk=None, trading_state=None, extra_reasons=()):
    """
    (ready, degraded_reason) for the heartbeat. ready = the strategy may open
    NEW entries. Not ready / degraded never means "dead": exits and the kill
    switch keep running (Phase 2 I-4 separates this from liveness).
    """
    blocking, notes = [], []
    if not model_ok:
        blocking.append(model_reason)
    if risk is not None:
        if risk.halted:
            blocking.append("halted by kill switch")
        elif risk.level == "suspect":
            blocking.append(f"risk reading suspect ({risk.reason})")
        elif risk.level == "critical":
            blocking.append("critical drawdown")
        if risk.discrepancy_symbols:
            notes.append(f"ledger discrepancy flagged for {risk.discrepancy_symbols}")
    if trading_state == "HALT":
        blocking.append("equity-curve filter HALT")
    blocking.extend(r for r in extra_reasons if r)
    reasons = blocking + notes
    return (not blocking), ("; ".join(reasons) if reasons else None)


def run_cycle(ctx, api, sleep_fn=time.sleep, now=None):
    """
    One full cycle for a strategy. Returns a dict summarising what happened so
    tests (and the caller) can inspect it. Never raises for per-ticker errors.
    """
    s = ctx.name
    summary = {"strategy": s, "tickers": {}, "paused": 0, "killed": False, "halted": False}
    portfolio.ensure_strategy_state(s, ctx.db_path)
    orders_reconciled = True
    try:
        reconcile_open_orders(api, s, ctx.db_path)
    except Exception as e:  # reconciliation must never take the whole cycle down (second-pass S-04)
        orders_reconciled = False
        print(f"[{s}] order reconciliation failed: {e}")
    model_ok, model_reason = _check_models(ctx, now)
    summary["model_ok"], summary["model_reason"] = model_ok, model_reason

    if ctx.uses_market_hours:
        from paper_trading.alpaca_paper import sleep_seconds_until_open
        secs = sleep_seconds_until_open(api.get_clock())
        if secs > 0:
            halted = portfolio.get_strategy_state(s, ctx.db_path)["halted"]
            ready, degraded = _readiness(model_ok, model_reason,
                                         extra_reasons=["halted by kill switch"] if halted else ())
            log_heartbeat(s, "PAUSED", ctx.db_path, ready=ready, degraded_reason=degraded)
            summary["paused"] = secs
            summary["ready"], summary["degraded_reason"] = ready, degraded
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
        if _stopping(ctx):
            print(f"[{s}] shutdown requested during data fetch; no trading this cycle")
            summary["stopped"] = True
            return summary
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
    atr_fracs = {t: atr_as_fraction(_last_float(df, "atr"), marks.get(t))
                 for t, df in frames.items() if df is not None and len(df)}
    bar_ages = {t: _bar_age_days(df, now) for t, df in frames.items() if df is not None and len(df)}

    # One broker lookup per held symbol: flags or clears discrepancies and
    # provides the independent quotes used to validate marks (R-1).
    try:
        broker_quotes = refresh_discrepancies(ctx, api)
        unknown = sorted(t for t, q in broker_quotes.items() if q.status == "unknown")
        positions_checked = not unknown
        if unknown and not ctx.runtime.get("startup_reconciled"):
            print(f"[{s}] startup reconciliation incomplete: broker state unknown for {unknown}")
    except Exception as e:
        print(f"[{s}] discrepancy check failed: {e}")
        broker_quotes = {}
        positions_checked = False
    # Phase 2 I-4: after a (re)start, no NEW entry until open orders have been
    # reconciled and every held position compared with the broker in one cycle.
    if orders_reconciled and positions_checked and not ctx.runtime.get("startup_reconciled"):
        ctx.runtime["startup_reconciled"] = True
        print(f"[{s}] startup reconciliation complete; entries may resume")

    def verify_mark(symbol):
        q = broker_quotes.get(symbol)
        return q if q is not None else broker_quote(api, to_broker_symbol(symbol))

    risk = evaluate_risk(ctx, marks, now, atr_fractions=atr_fracs, bar_ages=bar_ages, verify_mark=verify_mark)
    marks = risk.accepted_marks or marks
    summary["risk"] = risk
    _alert_new_discrepancies(ctx, risk.equity)
    _alert_suspect_state(ctx, risk)
    _alert_negative_budget(ctx, risk.equity)
    if risk.halted:
        ready, degraded = _readiness(model_ok, model_reason, risk)
        summary["ready"], summary["degraded_reason"] = ready, degraded
        log_heartbeat(s, "HALTED", ctx.db_path, ready=ready, degraded_reason=degraded)
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
        ready, degraded = _readiness(model_ok, model_reason, risk, extra_reasons=["kill switch fired"])
        summary["ready"], summary["degraded_reason"] = ready, degraded
        log_heartbeat(s, "HALTED", ctx.db_path, ready=ready, degraded_reason=degraded)
        return summary

    from metrics.equity_curve_filter import get_trading_state
    state, multiplier = get_trading_state(s, ctx.db_path)
    gate_reasons = _entry_gates(ctx)
    allow_entries = state != "HALT" and risk.level in ("safe", "warning") and not gate_reasons
    summary["trading_state"] = state
    summary["entry_gates"] = gate_reasons
    vix = ctx.macro_context()

    for ticker in symbols:
        if _stopping(ctx):
            print(f"[{s}] shutdown requested; remaining tickers skipped this cycle")
            break
        try:
            summary["tickers"][ticker] = trade_ticker(ctx, api, ticker, frames.get(ticker), risk, multiplier, vix,
                                                      allow_entries and ticker in ctx.assets, now)
        except Exception as e:
            print(f"[{s}] Error trading {ticker}: {e}")
            summary["tickers"][ticker] = f"error: {e}"
        log_heartbeat(s, "RUNNING", ctx.db_path)

    _alert_new_discrepancies(ctx, risk.equity)     # flags raised by sells during this cycle
    ready, degraded = _readiness(model_ok, model_reason, risk, state, gate_reasons)
    summary["ready"], summary["degraded_reason"] = ready, degraded
    log_heartbeat(s, "SUSPECT" if risk.level == "suspect" else "RUNNING", ctx.db_path,
                  ready=ready, degraded_reason=degraded)
    return summary


def _paused_wait(ctx, secs, sleep_fn):
    """
    A market-closed sleep (up to 48 h) in chunks of PAUSED_HEARTBEAT_SECONDS,
    writing a PAUSED heartbeat each time, so liveness never goes stale while
    the bot is healthy and a stop request is honoured promptly (Phase 2 I-4).
    """
    remaining = float(secs)
    while remaining > 0 and not _stopping(ctx):
        chunk = min(PAUSED_HEARTBEAT_SECONDS, remaining)
        sleep_fn(chunk)
        remaining -= chunk
        if not _stopping(ctx):
            log_heartbeat(ctx.name, "PAUSED", ctx.db_path)


def run_forever(ctx, api=None, sleep_fn=None, max_cycles=None, stop_event=None):
    """
    Main loop used by run.py threads. `max_cycles` is for tests.

    Phase 2 I-4: sleeps wait on the stop event (SIGTERM wakes them), long
    market-closed sleeps keep the heartbeat fresh, and startup reconciles open
    orders BEFORE adopting any broker position (a fill booked while the
    process was down must not be adopted and then booked again).
    """
    from paper_trading.alpaca_paper import get_api
    s = ctx.name
    if stop_event is not None:
        ctx.stop_event = stop_event
    if ctx.stop_event is None:
        ctx.stop_event = threading.Event()
    stop = ctx.stop_event
    sleep_fn = sleep_fn or (lambda secs: stop.wait(secs))
    api = api or get_api(s)
    print(f"{s} bot started (paper endpoint check: {getattr(api, '_base_url', 'n/a')})")
    portfolio.ensure_strategy_state(s, ctx.db_path)
    try:
        reconcile_open_orders(api, s, ctx.db_path)
    except Exception as e:
        print(f"[{s}] startup order reconciliation failed: {e}")
    adopt_unowned_broker_positions(ctx, api)
    last_trade_time = utcnow()
    cycles = 0
    while (max_cycles is None or cycles < max_cycles) and not stop.is_set():
        cycles += 1
        try:
            summary = run_cycle(ctx, api, sleep_fn)
            if summary.get("stopped") or stop.is_set():
                break
            if summary["paused"]:
                secs = summary["paused"]
                print(f"[{s}] Market is closed - sleeping for {secs // 3600}h {(secs % 3600) // 60}m")
                _safe_notify(ctx, api, s, "Market closed", f"Sleeping {secs // 3600}h - market closed")
                _paused_wait(ctx, secs, sleep_fn)
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
    if stop.is_set():
        log_heartbeat(s, "STOPPED", ctx.db_path, ready=False, degraded_reason="process shutting down")
        print(f"[{s}] stopped cleanly")


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
