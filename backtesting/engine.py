import vectorbt as vbt
import pandas as pd
import numpy as np
from core.features import build_features
from models.train import load_model

#this file is meant to take historical OHLCV data, run buildfeatures, generate buy/sell signals from the ml model, feed to vectorBT, and it will return the returns, drawdown and Sharpe ratio

def generate_signals(df,model):
    """
    Takes the trained MML model and generates BUY/SELL signalls. It will also return two boolean series (entries and exits), where entries is true if model says BUY and exits is True when model says Sell

    """

    feature_cols=[c for c in df.columns if c not in ['open','high','low','close','volume']]
    X=df[feature_cols].values
    predictions=model.predict(X)
    entries=pd.Series(predictions==1,index=df.index)#BUY signals is entries
    exits=pd.Series(predictions==0,index=df.index)#Sell signals is exits
    return entries, exits

def run_backtest(df,model,strategy_name,init_cash=1000.00):#starting cash is 1000 to make it easier to see what the returns would be
    """
    Runs a full backtest on each ticker. 
    df: featured DataFrame with OHLCV + indicators
    model: trained XGBoost model
    strategy_name: "stable", "risky1", or "risky2"
    init_cash : starting capital
    returns VectorBT portfolio object with full stats
    """
    entries,exits=generate_signals(df,model)

    portfolio=vbt.Portfolio.from_signals(close=df['close'], entries=entries, exits=exits, init_cash=init_cash, fees=.001, slippage=.001, freq='D')# .1% fee per trade is acceptable for Alpaca and .1% slippage (difference between expected price and price trade is done at according to investopedia)
    print(f"\n {strategy_name} Backtest Results: -------")
    print(f"Total Return: {portfolio.total_return():.2%}")
    print(f"Sharpe Ratio : {portfolio.sharpe_ratio():.2f}")
    print(f"Max Drawdown : {portfolio.max_drawdown():.2%}")
    print(f"Total Trades conducted: {portfolio.trades.count()}")
    return portfolio


def run_backtest_multiple(tickers_data,model,strategy_name,init_cash=1000.00): #cash starts with 1000 to make it easier to see how well the model is doing 
    """
    Runs backtests across multiple tickers and combines the result per model
    Tickers_data= dictionary of {ticker:featured df}
    it will return a dictionary of {ticker:portfolio objects}
    """
    results={}
    for ticker, df in tickers_data.items():
        print(f"Backtesting {ticker} right now ...")
        try:
            portfolio=run_backtest(df,model,strategy_name,init_cash)
            results[ticker]=portfolio
        except Exception as e:
            print(f"Warning: backtest failed for {ticker}-{e}")
    return results

def get_backtest_summary(results):
    """
    This will take the results from the dictionary and return a df summary. It compares all the tickers side by side and will be used by the dashboard in comparison table for easy understanding
    """

    rows=[]
    for ticker, portfolio in results.items():
        rows.append({
            'ticker':ticker,'total_return':portfolio.total_return(), 'sharpe_ratio':portfolio.sharpe_ratio(),'max_drawdown':portfolio.max_drawdown(), 'total_trades':portfolio.trades.count()
        })
    if not rows:
        print("No successful Backtests to summarize here")
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('ticker')