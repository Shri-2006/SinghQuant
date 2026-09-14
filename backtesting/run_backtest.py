"""
Runs the vectorbt backtest for the supervised strategies.

IMPORTANT (audit, ML section): models are trained on the most recent two
years and the original backtest ran over that SAME window, so every number in
RESULTS.md is IN-SAMPLE. By default this script now evaluates only the
hold-out tail of each series (the last `HOLDOUT_FRACTION` of rows, matching
models.train.time_split). Pass --full to reproduce the old in-sample numbers.
"""
import os
import sys
os.makedirs(os.path.join("backtesting", "results"), exist_ok=True)
import pandas as pd
from datetime import datetime, timedelta
from data.polygon_fetcher import get_multiple_tickers
from core.features import build_features
from models.train import load_model, FEATURE_COLUMNS
from backtesting.engine import run_backtest_multiple, get_backtest_summary
from core.config import STABLE_ASSETS, RISKY1_ASSETS
from data.sentiment_fetcher import add_sentiment_to_df

# Date range for backtesting — last 2 years
END   = datetime.today().strftime('%Y-%m-%d')
START = (datetime.today() - timedelta(days=365*2)).strftime('%Y-%m-%d')
HOLDOUT_FRACTION = 0.2


def backtest_strategy(strategy, holdout_fraction=HOLDOUT_FRACTION):
    """
    Runs full backtest for a given strategy
    strategy: "stable" or "risky1"
    holdout_fraction: use only the last fraction of each series (0 = full, in-sample)
    """
    print(f"\nRunning backtest for {strategy} ({'hold-out' if holdout_fraction else 'IN-SAMPLE'})...")

    # Load the right assets and model
    if strategy == "stable":
        tickers  = STABLE_ASSETS
        model    = load_model("stable_model.pkl")
        cash     = 1000.0
    elif strategy == "risky1":
        tickers  = RISKY1_ASSETS
        model    = load_model("risky1_model.pkl")
        cash     = 1000.0
    else:
        print(f"Unknown strategy: {strategy}")
        return

    # Fetch and feature data for all tickers
    raw_data      = get_multiple_tickers(tickers, START, END)
    featured_data= {}
    for t,df in raw_data.items():
        df=build_features(df)
        df=add_sentiment_to_df(df,t, score=0.0)   # neutral, as in training
        df=df.dropna()
        if holdout_fraction:
            cut = int(len(df) * (1 - holdout_fraction))
            df = df.iloc[cut:]
        featured_data[t]=df
    # Run backtest and store in summary
    results = run_backtest_multiple(featured_data, model, strategy, cash)
    summary = get_backtest_summary(results)

    print(f"\n=== {strategy.upper()} SUMMARY ===")
    print(summary.to_string())
    # Save results to CSV file backtest.csv
    tag = "holdout" if holdout_fraction else "insample"
    out_path = os.path.join("backtesting", "results", f"{strategy}_backtest_{tag}.csv")
    summary.to_csv(out_path)
    print(f"Results saved to {out_path}")
    return summary


def run_all(holdout_fraction=HOLDOUT_FRACTION):
    """
    Runs backtests for all supervised strategies
    """
    stable_summary = backtest_strategy("stable", holdout_fraction)
    risky1_summary = backtest_strategy("risky1", holdout_fraction)
    print("\n All Backtests are complete now")
    return stable_summary, risky1_summary


if __name__ == "__main__":
    run_all(0.0 if "--full" in sys.argv else HOLDOUT_FRACTION)
