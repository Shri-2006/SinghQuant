import os
os.makedirs(os.path.join("backtesting", "results"), exist_ok=True)
import pandas as pd
from datetime import datetime, timedelta
from data.polygon_fetcher import get_multiple_tickers
from core.features import build_features
from models.train import load_model
from backtesting.engine import run_backtest_multiple, get_backtest_summary
from core.config import STABLE_ASSETS

# Date range for backtesting — last 2 years
END   = datetime.today().strftime('%Y-%m-%d')
START = (datetime.today() - timedelta(days=365*2)).strftime('%Y-%m-%d')

def backtest_strategy(strategy):
    """
    Runs full backtest for a given strategy
    strategy: "stable" or "risky1"
    """
    print(f"\nRunning backtest for {strategy}...")

    # Load the right assets and model
    if strategy == "stable":
        tickers  = STABLE_ASSETS
        model    = load_model("stable_model.pkl")
        cash     = 1000.0
    elif strategy == "risky1":
        tickers  = ["SPY"]  #just a placeholder until ML builds more tickers
        model    = load_model("risky1_model.pkl")
        cash     = 1000.0
    else:
        print(f"Unknown strategy: {strategy}")
        return

    # Fetch and feature data for all tickers
    raw_data      = get_multiple_tickers(tickers, START, END)
    featured_data = {t: build_features(df) for t, df in raw_data.items()}

    # Run backtest and store in summary
    results = run_backtest_multiple(featured_data, model, strategy, cash)
    summary = get_backtest_summary(results)

    print(f"\n=== {strategy.upper()} SUMMARY ===")
    print(summary.to_string())
    # Save results to CSV file backtest.csv
    out_path = os.path.join("backtesting", "results", f"{strategy}_backtest.csv")
    summary.to_csv(out_path)
    print(f"Results saved to {out_path}")
    return summary

def run_all():
    """
    Runs backtests for all supervised strategies
    """
    stable_summary = backtest_strategy("stable")
    risky1_summary = backtest_strategy("risky1")
    print("\n All Backtests are complete now")
    return stable_summary, risky1_summary

if __name__ == "__main__":
    run_all()