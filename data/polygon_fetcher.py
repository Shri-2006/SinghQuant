import os
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from core.config import POLYGON_API_KEY, POLYGON_API_KEY_RISKY1, POLYGON_API_KEY_RISKY2

#first initialize the polygon client
#client=RESTClient(api_key=POLYGON_API_KEY)
def get_client(api_key=None):
    """Returns a Polygon client with the specified or default API key"""
    return RESTClient(api_key=api_key or POLYGON_API_KEY)


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
    #conversion of bars to dataframe
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(bar.timestamp, unit='ms'),
        'open'     : bar.open,
        'high'     : bar.high,
        'low'      : bar.low,
        'close'    : bar.close,
        'volume'   : bar.volume
    } for bar in bars])

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_latest_bar(ticker, timespan="day", api_key=None):
    """
    Fetches the most recent price bar for live trading and is used by the bots to make real time decisions
    ticker: e.g. "SPY" or "AAPL", basically what the bots are invested in
    Returns: single rowed pandas dataframe
    """

    #set dates
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')#/ doesn't work idk why, prob a python thing
    df    = get_historical_data(ticker, start, end, timespan, api_key=api_key)
    
    #set the dataframe with historical data
  #  df=get_historical_data(ticker,start,end,timespan)

    #if no data in df, send a warning
    if df.empty:
        print(f"Warning, No data is in df for {ticker}")
        return None
    #return only the most recent row
    return df

import time
#adding a sleep function
def get_multiple_tickers(tickers, start, end, timespan="day", api_key=None):
    """Fetch hisotrical data for multiple tickers with a rate limit delay of 12 to prevent rate limiting"""
    data = {}
    for t in tickers:
        print(f"Fetching ticker {t}...")
        df = get_historical_data(t, start, end, timespan, api_key=api_key)
        if not df.empty:
            data[t] = df
        else:
            print(f"Warning: no data for {t}, skipping")
        time.sleep(12)
    return data


#Adding crypto detector and crypto latest bar function
def is_crypto(ticker):
    """
    Checks if a ticker is a crypto pair, if it is use X: prefix for polygon
    e.g. X:BTCUSD, X:ETHUSD
    """
    return ticker.startswith("X:")

def get_latest_price(ticker, api_key=None):
    """
    Works for both stocks and crypto, and Automatically handles the X: prefix for crypto
    """
    bar = get_latest_bar(ticker, api_key=api_key)
    if bar is None:
        return None
    return bar['close'].iloc[0]