"""
Per-trade and per-strategy risk rules.

Units (see ENGINEERING_AUDIT.md issue H-01):
  * `current_atr` in every function here is a FRACTION OF PRICE (e.g. 0.02 for
    a 2% daily range). core/features.py produces ATR in price units, so
    callers convert with `atr_as_fraction(atr_dollars, price)` first.
  * drawdown values are negative fractions (-0.15 == 15% below the baseline).
"""
import math

from core.config import (MAX_DRAWDOWN, WARNING_DRAWDOWN, STOP_LOSS_PER_TRADE, CAPITAL,
                         ATR_BASELINE, EQUITY_MAX_STEP_CHANGE)

# ATR scale is clamped so thresholds never go to an extreme in either direction
ATR_SCALE_MIN = 0.5
ATR_SCALE_MAX = 1.5


def atr_as_fraction(atr_price_units, price):
    """
    Converts an ATR in price units (what core.features.build_features returns)
    into a fraction of price, the unit ATR_BASELINE is expressed in.
    Returns None when either input is missing or non-positive so callers fall
    back to the static thresholds instead of a meaningless scale.
    """
    try:
        atr = float(atr_price_units)
        px = float(price)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(atr) and math.isfinite(px)) or atr < 0 or px <= 0:
        return None
    return atr / px


def get_atr_adjusted_thresholds(strategy, current_atr):
    """
    Returns (warning_threshold, kill_threshold) scaled by current market volatility. If ATR is double the baseline, thresholds widen so normal volatility doesn't trip the kill switch. If ATR is below baseline, thresholds tighten to protect capital in calm markets. Scale is kept between 0.5x and 1.5x so it can never go extreme and break everythinng.
    `current_atr` must be a fraction of price (see atr_as_fraction).
    """
    scale =current_atr /ATR_BASELINE[strategy]
    scale = max(ATR_SCALE_MIN, min(ATR_SCALE_MAX, scale))  #prevent too far
    adjusted_warning= WARNING_DRAWDOWN[strategy] *scale
    adjusted_kill= MAX_DRAWDOWN[strategy]*scale
    return (adjusted_warning, adjusted_kill)


def risk_level_from_drawdown(strategy, drawdown, current_atr=None):
    """
    Maps a drawdown fraction (negative = loss) to "safe" | "warning" | "critical".
    This is the single decision function; every caller (kill switch, position
    sizing, stop logic, dashboard) must use it so that the number that is
    logged is the number that decided (audit issue C-03).
    """
    if drawdown is None or not math.isfinite(drawdown):
        # Unknown drawdown must never look like a catastrophe or like safety.
        return "warning"
    if current_atr is not None:
        warning_threshold, kill_threshold = get_atr_adjusted_thresholds(strategy, current_atr)
    else:
        warning_threshold = WARNING_DRAWDOWN[strategy]
        kill_threshold    = MAX_DRAWDOWN[strategy]

    if drawdown <= kill_threshold:
        return "critical"
    elif drawdown <= warning_threshold:
        return "warning"
    else:
        return "safe"


def get_portfolio_risk_level(strategy, current_equity, current_atr=None, peak_equity=None):
    """
    Returns "safe" | "warning" | "critical" for a strategy.
    Drawdown baseline is `peak_equity` when given (the intended behaviour per
    DECISION_LOG 2026-04-11), otherwise the strategy's starting CAPITAL (the
    original behaviour, kept for backward compatibility and tests).
    `current_atr` is a fraction of price or None.
    """
    baseline = peak_equity if peak_equity else CAPITAL[strategy]
    if baseline <= 0:
        return "warning"
    drawdown = (current_equity - baseline) / baseline
    return risk_level_from_drawdown(strategy, drawdown, current_atr)


def get_position_size(strategy, current_equity, base_size, current_atr=None, peak_equity=None):
    """
    Returns the adjusted position size based on the current level of risk. If safe, full position size. if warning, half position size (to reduce risk before kill switch is tripped), if critical 0 (stop buying immediately)
    base size is the normal max position size from config btw
    """
    risk_level = get_portfolio_risk_level(strategy, current_equity, current_atr, peak_equity)

    if risk_level == "critical":  #sell now
        return 0.0
    elif risk_level == "warning":
        return base_size * (1/2) #half the position
    else:
        return base_size  #safe


def check_stop_loss(strategy,entry_price,current_price):
    """
    if stop loss has been activated for a position it will hit true. It will compare current price to entry price using the stop_loss-per_trade limit
    """
    if entry_price<=0:
        return False
    loss_per_trade=((current_price-entry_price)/entry_price)
    return (loss_per_trade<=-STOP_LOSS_PER_TRADE[strategy])


def should_close_position(strategy, entry_price, current_price, current_equity, current_atr=None, peak_equity=None):
    """
    This is a master check that determines if a position must be closed immediately. Combines both stop loss check and portfolio risk level.
    Returns trueif the position should be closed, false if safe to hold.
    """
    #close if stop loss hits
    if check_stop_loss(strategy,entry_price,current_price):
        print(f"Stop loss was triggered for {strategy} \n entry was ${entry_price:.2f} and current price is ${current_price:.2f}")
        return True

    #if portfolio has hit critical, close now
    risk_level=get_portfolio_risk_level(strategy,current_equity,current_atr, peak_equity)
    if risk_level=="critical":
        print(f"Portfolio for {strategy} has hit critical, closing positions")
        return True

    return False


def validate_equity(equity, previous_equity=None, max_step_change=EQUITY_MAX_STEP_CHANGE):
    """
    Decides whether an equity reading can be trusted for a risk decision.
    Returns (ok, reason). A reading is rejected when it is missing, NaN, inf,
    non-positive, or moves more than `max_step_change` (fraction) from the
    previous reading in a single cycle. Rejected readings must not update the
    peak and must not fire the kill switch on their own (audit issue C-03:
    a single bad broker value liquidated the whole account).
    """
    try:
        e = float(equity)
    except (TypeError, ValueError):
        return False, "equity not numeric"
    if not math.isfinite(e):
        return False, "equity not finite"
    if e <= 0:
        return False, "equity non-positive"
    if previous_equity is not None and previous_equity > 0:
        step = abs(e - previous_equity) / previous_equity
        if step > max_step_change:
            return False, f"equity moved {step:.0%} in one cycle (limit {max_step_change:.0%})"
    return True, "ok"
