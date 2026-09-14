import os
import pandas as pd
from core.logger import get_trades
from core.config import CAPITAL
from metrics.risk import compute_all_metrics

TRADE_COLUMNS = ['id','timestamp','strategy','asset','action','price','quantity','pnl','reason']


def get_strategy_returns(strategy):
    """
    Pulls closed trades from SQLite and returns per-trade RETURNS (fractions of
    the strategy's capital), which is what metrics.risk expects.

    BUG FIXED (audit issue M-04): the original returned raw dollar pnl and fed
    it to Sharpe/Sortino/max-drawdown, which compound (1 + r); a $17 win was
    treated as a 1,700% return.
    """
    trades=get_trades(strategy)
    if not trades:
        return None
    df=pd.DataFrame(trades,columns=TRADE_COLUMNS)
    df=df[df['action']=='SELL']
    df['pnl']=pd.to_numeric(df['pnl'],errors='coerce')
    df=df[df['pnl'].notna()]
    if df.empty:
        return None
    return (df['pnl'] / CAPITAL[strategy]).reset_index(drop=True)


def print_comparision():
    """
    Prints a side by side comparision for all strategies
    """
    strategies=["stable","risky1","risky2"]
    rows=[]
    for strategy in strategies:
        returns = get_strategy_returns(strategy)
        if returns is None or len(returns)==0:
            rows.append({"strategy":strategy,"sharpe_ratio":"N/A","max_drawdown":"N/A","win_loss":"N/A","total_trades":0,"status":"No data"})
        else:
            metrics = compute_all_metrics(returns)
            rows.append({"strategy": strategy,
                         "sharpe_ratio":f"{metrics['sharpe_ratio']:.2f}",
                         "max_drawdown": f"{metrics['max_drawdown']:.2%}"
                         ,"win_loss": f"{metrics['win_loss_ratio']:.2%}",
                         "total_trades": len(returns),
                         "status": "RUNNING"})

    df=pd.DataFrame(rows).set_index("strategy")
    print("\n======TRADING SYSTEM STRATEGY COMPARISION TABLE======\n")
    print(df.to_string())
    print()

if __name__=="__main__":
    print_comparision()
