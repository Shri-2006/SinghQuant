import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from models.rl_environment import TradingEnvironment
from data.polygon_fetcher import get_historical_data
from core.features import build_features
from data.sentiment_fetcher import add_sentiment_to_df
from core.config import CAPITAL, MAX_POSITION_SIZE

MODEL_PATH = "models/risky2_model.zip"
STRATEGY   = "risky2"

# Default crypto tickers to train on
DEFAULT_TICKERS = [
    "X:BTCUSD", "X:ETHUSD", "X:SOLUSD",
    "X:AVAXUSD", "X:LINKUSD", "X:ADAUSD"
]


def prepare_rl_data(ticker, days_to_look_back=365):
    """
    Prepares historical data for a single ticker.
    Fetches OHLCV, builds features, adds sentiment, drops NaN rows.
    Uses 1 year of data by default to capture multiple market regimes.
    """
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days_to_look_back)).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} from {start} to {end}...")
    df = get_historical_data(ticker, start, end)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker)
    df = df.dropna()

    print(f"{ticker}: {len(df)} rows ready")
    return df


def train_rl_model(ticker="X:BTCUSD", days_to_look_back=365, timesteps=500000):
    """
    Trains PPO agent on a single crypto ticker.
    Good for quick experiments or retraining on one asset.
    
    ticker:           Polygon format e.g. X:BTCUSD
    days_to_look_back: how far back to fetch training data
    timesteps:        how long to train — 500k is the sweet spot
    """
    df  = prepare_rl_data(ticker, days_to_look_back)
    env = TradingEnvironment(
        df=df,
        initial_capital=CAPITAL[STRATEGY],
        max_position_pct=MAX_POSITION_SIZE[STRATEGY] / CAPITAL[STRATEGY]
    )

    print("Checking environment...")
    check_env(env, warn=True)

    print("Building PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,   # 0.99 = values future rewards almost as much as immediate
        seed=42
    )

    print(f"Training on {ticker} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def train_rl_model_multi(tickers=None, days_to_look_back=365, timesteps=500000):
    """
    Trains PPO agent on combined data from multiple crypto tickers.
    More diverse training = better generalization across different cryptos.
    Skips tickers with no data or fewer than 50 rows.
    
    tickers:          list of Polygon format tickers, defaults to DEFAULT_TICKERS
    days_to_look_back: lookback window per ticker
    timesteps:        total PPO training steps
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days_to_look_back)).strftime('%Y-%m-%d')

    dfs = []
    for ticker in tickers:
        try:
            df = get_historical_data(ticker, start, end)
            if df.empty:
                print(f"No data for {ticker}, skipping")
                continue
            df = build_features(df)
            df = add_sentiment_to_df(df, ticker)
            df = df.dropna()
            if len(df) < 50:
                print(f"{ticker} has too few rows ({len(df)}), skipping")
                continue
            dfs.append(df)
            print(f"{ticker}: {len(df)} rows")
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    if not dfs:
        raise ValueError("No data fetched for any ticker — cannot train")

    combined = pd.concat(dfs).sort_index().reset_index(drop=True)
    print(f"\nCombined dataset: {len(combined)} rows from {len(dfs)} tickers")

    env = TradingEnvironment(
        df=combined,
        initial_capital=CAPITAL[STRATEGY],
        max_position_pct=MAX_POSITION_SIZE[STRATEGY] / CAPITAL[STRATEGY]
    )

    print("Checking environment...")
    check_env(env, warn=True)

    print("Building PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        seed=42
    )

    print(f"Training on {len(dfs)} tickers for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_rl_model():
    """
    Loads trained PPO model from disk.
    Returns None if no model exists yet — bot will skip trading.
    """
    if os.path.exists(MODEL_PATH):
        print("loading an existing RL model from disk.....")
        return PPO.load(MODEL_PATH)
    print("No RL model found — needs training first")
    return None


if __name__ == "__main__":
    train_rl_model_multi()