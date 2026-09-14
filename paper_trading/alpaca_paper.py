"""
Thin wrapper around the Alpaca REST client.

PAPER-ONLY GUARD (audit issue C-01): `get_api` raises unless the strategy is
in PAPER_MODE, or the operator has explicitly opted in to live trading with
the SINGHQUANT_ALLOW_LIVE_TRADING environment variable. The audit found
risky2 silently pointed at the live URL.
"""
import alpaca_trade_api as tradeapi
from core.config import (ALPACA_PAPER_URL, ALPACA_LIVE_URL, PAPER_MODE, ALLOW_LIVE_TRADING,
                         ALPACA_KEYS_PER_STRATEGY, LIVE_TRADING_OPT_IN_VALUE)


class LiveTradingBlocked(RuntimeError):
    """Raised when code tries to build a live-money broker client without opt-in."""


def resolve_base_url(strategy):
    """
    Returns the Alpaca base URL for a strategy, refusing the live URL unless
    the operator opted in. Pure function so it can be unit tested.
    """
    if PAPER_MODE.get(strategy, True):
        return ALPACA_PAPER_URL
    if not ALLOW_LIVE_TRADING:
        raise LiveTradingBlocked(
            f"PAPER_MODE['{strategy}'] is False but live trading is not enabled. "
            f"SinghQuant is paper-only; set SINGHQUANT_ALLOW_LIVE_TRADING={LIVE_TRADING_OPT_IN_VALUE} "
            "only if you truly intend to trade real money."
        )
    return ALPACA_LIVE_URL


def get_api(strategy):
    """
    returns the authenticated alpaca api for the strategy. Paper URL unless the
    live opt-in described above is present.
    """
    url = resolve_base_url(strategy)
    key, secret = ALPACA_KEYS_PER_STRATEGY.get(strategy, (None, None))
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set (check your .env)")
    return tradeapi.REST(key, secret, url)


def is_market_open(strategy):
    """
    Checks if the us stock market is currently open (bots cannot and shouldnt trade when market is closed). Risky2 ignores this since crypto is 24/7
    """
    api=get_api(strategy)
    clock=api.get_clock()
    return clock.is_open


def sleep_seconds_until_open(clock):
    """
    Pure helper: how long to sleep given an Alpaca clock object.
    24 hours: Sleep 12
    12 hours: Sleep 8
    8 hours: Sleep 3
    1 hour: sleep 30 min
    0 hours: Sleep 1 min
    market open: return 0 (trade now)
    """
    if clock.is_open:
        return 0
    seconds_left=(clock.next_open-clock.timestamp).total_seconds()
    x=3600
    if seconds_left>(72*x):
        return 48*x
    elif seconds_left>(48*x):
        return 24*x
    elif seconds_left>24*x:
        return 12*x
    elif seconds_left>12*x:
        return x*8
    elif seconds_left>8*x:
        return x*3
    elif seconds_left>1*x:
        return (1/2)*x
    else:
        return 60


def get_sleep_duration(strategy, api=None):
    """
    To save compute credits, this will return how long to sleep based on time until the opening of the market.
    """
    api = api or get_api(strategy)
    return sleep_seconds_until_open(api.get_clock())


def get_account_info(strategy):
    """
    Returns account-wide equity/cash/buying power. NOTE: this is the SHARED
    account, not a strategy's equity; use core.portfolio.strategy_equity for
    risk decisions.
    """
    api=get_api(strategy)
    account=api.get_account()
    return{
        "equity":float(account.equity),"cash":float(account.cash),"buying_power":float(account.buying_power)
    }


def broker_position(api, symbol):
    """
    Returns (qty, avg_entry_price, current_price, market_value) of the
    ACCOUNT-wide broker position, or (0, 0, None, 0) if none.
    """
    try:
        pos = api.get_position(symbol)
    except Exception:
        return 0.0, 0.0, None, 0.0
    qty = float(pos.qty)
    avg = float(pos.avg_entry_price)
    px = None
    try:
        px = float(pos.current_price)
    except Exception:
        pass
    mv = float(pos.market_value) if getattr(pos, "market_value", None) is not None else (qty * px if px else 0.0)
    return qty, avg, px, mv


def get_position(strategy,ticker):
    """
    Returns the ACCOUNT-wide position value of a ticker in dollars (0 if none).
    Kept for the dashboard; strategies must not size or exit from this.
    """
    api=get_api(strategy)
    return broker_position(api, ticker)[3]


def submit_order(strategy,ticker,qty,side, time_in_force='day'):
    """
    Submits a market order. Prefer core.execution.submit_and_track, which
    also waits for the fill and reconciles the strategy ledger.
    """
    api=get_api(strategy)
    order=api.submit_order(symbol=ticker,qty=qty,side=side,type='market',time_in_force=time_in_force)
    print(f"Order has been submitted: {side} {qty} {ticker}")
    return order
