import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.rl_environment import TradingEnvironment
from core.config import CAPITAL, MAX_POSITION_SIZE, RISKY2_ASSETS

from models import artifacts

STRATEGY    = "risky2"
# SINGHQUANT_MODEL_DIR (default: this folder). Phase 2 I-3: PPO models are
# published and verified through models/artifacts.py like the XGBoost ones.
MODEL_DIR   = artifacts.model_dir()

# Default crypto tickers to train on
DEFAULT_TICKERS = list(RISKY2_ASSETS)


def get_model_path(ticker):
    """
    Logical artifact path for a ticker (X:BTCUSD -> <model dir>/risky2_BTCUSD.zip).
    The file actually loaded is the content-addressed one the manifest names.
    """
    return os.path.join(artifacts.model_dir(), artifacts.rl_artifact_name(ticker))


def prepare_rl_data(ticker, days_to_look_back=365):
    """
    Prepares historical data for a single ticker.
    Fetches OHLCV, builds features, adds sentiment, drops NaN rows.
    Uses 1 year of data by default to capture multiple market regimes.
    """
    from data.polygon_fetcher import get_historical_data
    from core.features import build_features
    from data.sentiment_fetcher import add_sentiment_to_df
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days_to_look_back)).strftime('%Y-%m-%d')

    print(f"Fetching {ticker} from {start} to {end}...")
    df = get_historical_data(ticker, start, end)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker, score=0.0)  # neutral constant during training (audit M-03)
    df = df.dropna()

    print(f"{ticker}: {len(df)} rows ready")
    return df


def train_rl_model(ticker="X:BTCUSD", days_to_look_back=365, timesteps=100000):
    """
    Trains a PPO agent for a single crypto ticker.
    Saves model to ticker-specific path e.g. models/risky2_BTCUSD.zip
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
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

    print(f"Training {ticker} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    import stable_baselines3
    from datetime import datetime, timezone
    artifacts.publish(artifacts.rl_artifact_name(ticker), lambda tmp: model.save(tmp), STRATEGY,
                      datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                      meta={"ticker": ticker, "timesteps": timesteps, "days_to_look_back": days_to_look_back,
                            "stable_baselines3": stable_baselines3.__version__})
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
            train_rl_model(ticker, days_to_look_back, timesteps)
            results[ticker] = "success"
        except Exception as e:
            print(f"Failed to train {ticker}: {e}")
            results[ticker] = f"failed: {e}"

    print(f"\n{'='*50}")
    print("Training complete. Results:")
    for ticker, status in results.items():
        print(f"  {ticker}: {status}")
    return results


def _load_ppo(path):
    from stable_baselines3 import PPO
    return PPO.load(path)


def _check_ppo(model):
    return None if hasattr(model, "predict") else "object has no predict()"


def rl_model_handle(ticker, directory=None):
    """
    Verified, hot-reloading PPO model for one ticker (Phase 2 I-3). The old
    silent fallback to an unverified risky2_model.zip is gone: register that
    file under each ticker's name with tools/model_manifest.py if it is wanted.
    """
    return artifacts.ArtifactHandle(artifacts.rl_artifact_name(ticker), _load_ppo, _check_ppo, directory)


def load_rl_model(ticker="X:BTCUSD"):
    """The verified PPO model for `ticker`, or None (the bot then takes no entries for it)."""
    return rl_model_handle(ticker).model


if __name__ == "__main__":
    train_all_tickers()
