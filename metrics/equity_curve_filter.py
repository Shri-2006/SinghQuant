import os
import sqlite3
import numpy as np

from core.config import  CAPITAL

#print("equity_curve_filter")
EQUITY_MA_PERIOD= 10 # how many closed trades to use for the MA
HISTORY_WINDOW = 50 # how far back to look (in closed trades)
THROTTLE_DRAWDOWN_P= 0.05  # 5%  then  half the position size
HALT_DRAWDOWN_P= 0.12  # 12% then no new trades

#Should map the state name to the position size multipier.
TRADING_STATES = {"FULL": 1.0,"THROTTLE": 0.5,"HALT":0.0,}

from core.logger import DB_PATH  # same database as the trade log (honours SINGHQUANT_DB_PATH)

# Only genuine closing trades carry a dollar pnl. KILL_SWITCH rows historically
# stored a drawdown FRACTION in the pnl column (audit issue M-04), so they are
# excluded here rather than added to a dollar running total.
CLOSING_ACTIONS = ("SELL",)


def _fetch_equity_history(strategy: str, n: int = HISTORY_WINDOW, db_path: str = None) -> list[float]:
    """
    This function should reconstruct the equity curve from closed traders per strategy. (equity=starting capital+cumulative profit/loss(pnl) over the last n closed trades, where n is  a selected amoutn of trades. Trades where profit/loss is null is not counted (except open positions)) This will return a list of quity values from oldest to newest.
    """
    syst=sqlite3.connect(db_path or DB_PATH, timeout=30)
    cursor=syst.cursor()
    placeholders = ",".join("?" for _ in CLOSING_ACTIONS)
    cursor.execute(f'''SELECT pnl FROM trades
                   WHERE strategy =? AND pnl IS NOT NULL AND action IN ({placeholders})
                   ORDER BY id DESC
                   LIMIT ?
                   ''',(strategy,*CLOSING_ACTIONS,n))
    rows=cursor.fetchall()
    syst.close()

    #learning SQL while doing this haha

    if not rows:
        return []
    pnls=[row[0] for row in reversed(rows)]
    starting_capital=CAPITAL[strategy]
    equity_curve=[]
    running_total=starting_capital
    for pnl in pnls:
        if pnl is None or not np.isfinite(pnl):
            continue
        running_total += pnl
        equity_curve.append(running_total)

    return equity_curve


def _compute_ma(values: list[float], period: int) -> float:
    """
    Simple moving average of the last `period` values.
    If fewer values than period exist, averages what's available.
    """
    window = values[-period:] if len(values) >= period else values
    return float(np.mean(window))


def _compute_drawdown(values: list[float]) -> float:
    """
    Computes drawdown from the peak equity within the window.
    Formula: (peak - current) / peak
    Returns 0.0 if equity is at or above peak (no drawdown).
    """
    if not values:
        return 0.0
    peak = max(values)
    current = values[-1]
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - current) / peak)


# Public interface for everyone

def get_trading_state(strategy: str, db_path: str = None) -> tuple[str, float]:
    """
    Master function — call this before placing any trade.
    Pulls equity history, computes MA and drawdown, then returns:
        FULL,     1.0)  then trade normally
        THROTTLE, 0.5)  then reduce position size by half
        HALT     0.0)  thendo not open new positions

    HALT overrides THROTTLE. Logic:
        - Drawdown > 12% then HALT  (hard stop)
        - Equity < its MA then HROTTLE (cold streak)
        - Drawdown > 5% then THROTTLE (approaching danger)
        - Otherwise do FULL

    NOTE: HALT means "no NEW positions". Callers must still evaluate exits
    (stop loss, momentum exit, kill switch) while halted; see audit issue H-05.
    """
    equity_curve = _fetch_equity_history(strategy, db_path=db_path)

    # Not enough data yet — trade normally, don't penalize early on
    if len(equity_curve) < 2:
        print(f"[equity_curve_filter] {strategy}: insufficient history, defaulting to FULL")
        return ("FULL", TRADING_STATES["FULL"])

    current_equity = equity_curve[-1]
    ma             = _compute_ma(equity_curve, EQUITY_MA_PERIOD)
    drawdown       = _compute_drawdown(equity_curve)

    # Determine state
    if drawdown >= HALT_DRAWDOWN_P:
        state = "HALT"
    elif drawdown >= THROTTLE_DRAWDOWN_P or current_equity < ma:
        state = "THROTTLE"
    else:
        state = "FULL"

    multiplier = TRADING_STATES[state]

    print(
        f"[equity_curve_filter] {strategy} | "
        f"equity=${current_equity:.2f} | "
        f"MA=${ma:.2f} | "
        f"drawdown={drawdown*100:.2f}% | "
        f"state={state} | "
        f"multiplier={multiplier}"
    )

    return (state, multiplier)
