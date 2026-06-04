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
from core.config import CAPITAL, MAX_POSITION_SIZE, RISKY2_ASSETS

STRATEGY    = "risky2"
MODEL_DIR   = "models"

# Default crypto tickers to train on
DEFAULT_TICKERS = [
    "X:BTCUSD", "X:ETHUSD", "X:SOLUSD",
    "X:AVAXUSD", "X:LINKUSD", "X:ADAUSD",
    "X:XRPUSD", "X:DOGEUSD"
]


def get_model_path(ticker):
    """
    Returns model path for a specific ticker.
    X:BTCUSD → models/risky2_BTCUSD.zip
    """
    clean = ticker.replace("X:", "").replace("/", "")
    return os.path.join(MODEL_DIR, f"risky2_{clean}.zip")


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


def train_rl_model(ticker="X:BTCUSD", days_to_look_back=365, timesteps=100000):
    """
    Trains a PPO agent for a single crypto ticker.
    Saves model to ticker-specific path e.g. models/risky2_BTCUSD.zip
    
    ticker:            Polygon format e.g. X:BTCUSD
    days_to_look_back: how far back to fetch training data
    timesteps:         100k per ticker is enough for per-asset training
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
        gamma=0.99,
        seed=42
    )

    model_path = get_model_path(ticker)
    print(f"Training {ticker} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def train_all_tickers(tickers=None, days_to_look_back=365, timesteps=100000):
    """
    Trains a separate PPO model for each crypto ticker.
    Each model learns that ticker's specific price patterns.
    Skips tickers that fail to fetch data.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results = {}
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Training {ticker}...")
        print(f"{'='*50}")
        try:
            model = train_rl_model(ticker, days_to_look_back, timesteps)
            results[ticker] = "success"
        except Exception as e:
            print(f"Failed to train {ticker}: {e}")
            results[ticker] = f"failed: {e}"

    print(f"\n{'='*50}")
    print("Training complete. Results:")
    for ticker, status in results.items():
        print(f"  {ticker}: {status}")
    return results


def load_rl_model(ticker="X:BTCUSD"):
    """
    Loads trained PPO model for a specific ticker.
    Returns None if no model exists yet — bot will skip trading.
    Falls back to legacy risky2_model.zip if ticker model not found.
    """
    model_path = get_model_path(ticker)

    if os.path.exists(model_path):
        print(f"Loading model for {ticker} from {model_path}")
        return PPO.load(model_path)

    # Legacy fallback
    legacy_path = os.path.join(MODEL_DIR, "risky2_model.zip")
    if os.path.exists(legacy_path):
        print(f"No ticker model found for {ticker}, using legacy model")
        return PPO.load(legacy_path)

    print(f"No RL model found for {ticker} — needs training first")
    return None


if __name__ == "__main__":
    train_all_tickers()