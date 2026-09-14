import os
import time
import pandas as pd
from datetime import datetime, timedelta
from core.config import POLYGON_API_KEY, POLYGON_API_KEY_RISKY1, POLYGON_API_KEY_RISKY2


def get_client(api_key=None):
    """Returns a Polygon client with the specified or default API key"""
    from polygon import RESTClient  # lazy: importing this module must not need a key
    return RESTClient(api_key=api_key or POLYGON_API_KEY)


def bars_to_dataframe(bars):
    """Converts Polygon agg bars (or any objects with the same attributes) to an OHLCV DataFrame."""
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(bar.timestamp, unit='ms'),
        'open'     : bar.open,
        'high'     : bar.high,
        'low'      : bar.low,
        'close'    : bar.close,
        'volume'   : bar.volume
    } for bar in bars])
    if df.empty:
        return df
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_historical_data(ticker, start, end, timespan="day", api_key=None):
    """
    Fetches historical OHLCV data from polygon.io
    ticker: e.g. "SPY" or "AAPL" or "X:BTCUSD" for crypto
    start : start date string "YYYY-MM-DD"
    end   : end date string "YYYY-MM-DD"
    timespan: "day", "hour", or "minute"
    api_key: optional override — pass strategy-specific key to avoid rate limits
    """
    client = get_client(api_key)
    bars = client.get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan=timespan,
        from_=start,
        to=end,
        limit=50000
    )
    return bars_to_dataframe(bars)


def get_latest_bar(ticker, timespan="day", api_key=None, lookback_days=200):
    """
    Fetches the most recent `lookback_days` of bars for live trading (the
    strategies need history for rolling indicators).
    Returns a DataFrame sorted oldest -> newest, or None if empty.

    NOTE (audit issue 4.5): on Polygon's free tier the last DAILY bar is the
    previous session's close, so `df['close'].iloc[-1]` is a stale signal
    price, not a live quote.
    """
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    df    = get_historical_data(ticker, start, end, timespan, api_key=api_key)

    if df.empty:
        print(f"Warning, No data is in df for {ticker}")
        return None
    return df


def get_multiple_tickers(tickers, start, end, timespan="day", api_key=None, delay=12):
    """Fetch hisotrical data for multiple tickers with a rate limit delay of 12 to prevent rate limiting"""
    data = {}
    for t in tickers:
        print(f"Fetching ticker {t}...")
        df = get_historical_data(t, start, end, timespan, api_key=api_key)
        if not df.empty:
            data[t] = df
        else:
            print(f"Warning: no data for {t}, skipping")
        time.sleep(delay)
    return data


def is_crypto(ticker):
    """
    Checks if a ticker is a crypto pair, if it is use X: prefix for polygon
    e.g. X:BTCUSD, X:ETHUSD
    """
    return ticker.startswith("X:")


def get_latest_price(ticker, api_key=None):
    """
    Latest available close for stocks or crypto.
    BUG FIXED: previously returned `.iloc[0]`, the OLDEST bar in the window.
    """
    bar = get_latest_bar(ticker, api_key=api_key)
    if bar is None:
        return None
    return float(bar['close'].iloc[-1])
