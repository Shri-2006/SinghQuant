import numpy as np
import pandas as pd


import time as _time

DEFAULT_RISK_FREE_RATE = 0.04
_rf_cache = {"rate": None, "ts": 0.0}
RISK_FREE_CACHE_SECONDS = 6 * 3600


def get_risk_free_rate():
    """
    Current US 10-year treasury yield, cached for 6 hours; falls back to 4%.

    NOTE (audit, LOW): Polygon's `get_last_trade("I:TNX")` is an index with no
    trades, so this has always fallen back to 4%. The result is now cached so
    the dashboard does not make a failing network call on every metric.
    """
    now = _time.time()
    if _rf_cache["rate"] is not None and now - _rf_cache["ts"] < RISK_FREE_CACHE_SECONDS:
        return _rf_cache["rate"]
    rate = DEFAULT_RISK_FREE_RATE
    try:
        from polygon import RESTClient
        from core.config import POLYGON_API_KEY
        client=RESTClient(api_key=POLYGON_API_KEY)
        #The ticker is I:TNX for polygon 10 year treasury yield
        data=client.get_last_trade("I:TNX")
        rate=data.price/100
        print(f"Current risk-free rate that is fetched is : {data.price:.2f}%")
    except Exception as e:
        print(f"WARNING, falling back to default rate of 4% since treasury rate could not be fetched: {e}")
    _rf_cache.update(rate=rate, ts=now)
    return rate
    





def sharpe_ratio(returns,risk_free_rate=None):
    """
    Calculates the annualized sharpe ratio. Returns is pandas series of daily returnss, risk_free_rate is the annual risk free rate, default is 4%, will change to live us treasury if working right. 
    The higher the ratio, the better. If above 1, it is good, above 2 is amazing, and above that is insanely good
    """
    if (risk_free_rate is None):
        risk_free_rate=get_risk_free_rate()
    
    daily_return_free=risk_free_rate/252 #divided by 252 business days per year
    excess=returns-daily_return_free
    if(excess.std()<1e-10):
        return 0.00000000000000000#muscle spasm hehe
    return(excess.mean()/excess.std())*np.sqrt(252)#annualize sharpe ratio through sqrt(252)* mean/std of excess returns


#https://www.investopedia.com/terms/s/sortinoratio.asp

def sortino_ratio(returns,risk_free_rate=None):
    """
    Sortino ratio is similar to sharpe but only penalizes downturn volatility. This is better since we like upside volatility.
    """
    if risk_free_rate is None:
        risk_free_rate=get_risk_free_rate()
    daily_rf=risk_free_rate/252 #252 business days
    excess = returns-daily_rf
    #this will only check negative returns for the downside turns
    downside=excess[excess<0]

    if len(downside)<2:
        return 0.0
    downside_std=downside.std()
    if downside_std<3e-10:
        return 0.0
    return (excess.mean()/downside_std)*(252**(1/2))


def max_drawdown(returns):
    """
    Calculates max drawdown (worst peak to trough loss). returns is pandas series of daily returns
    result is negative (-.23=23%)
    """

    combined=(1+returns).cumprod()#for every 1 dollar invested what is growth
    peak = combined.cummax()#running max
    drawdown=(combined-peak)/peak
    return drawdown.min()#worst point in this

def win_loss_ratio(returns):
    """
    Percentage of days with positive returns out of total days. .55=55% profitable days
    """
    if (len(returns)==0):
        return 0.0;
    wins=(returns>0).sum();
    return wins/len(returns)



def compute_all_metrics(returns):
    """
    It runs all three functions to get the 3 metrics and return in a clean dictioonary. The dashboard will use it to display the comparision of strategies
    returns is the pandas series of daily returns
    """

    return {

        "sharpe_ratio":sharpe_ratio(returns),
        "sortino_ratio":sortino_ratio(returns),
        "max_drawdown":max_drawdown(returns),
        "win_loss_ratio":win_loss_ratio(returns)
    }

