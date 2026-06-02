import pandas as pd
import numpy as np
from core.features import build_features
from ta.trend import ADXIndicator
import requests

# Regime labels
TRENDING  = "TRENDING"
RANGING   = "RANGING"
VOLATILE  = "VOLATILE"

# ADX threshold per strategy — risky1 needs lower threshold since
# momentum stocks don't always hit ADX 25 but still trend clearly
ADX_THRESHOLD = {
    "stable": 25,
    "risky1": 20,
    "risky2": 20,
    "default": 25
}

def detect_regime(df, strategy="default"):
    """
    Detects the current market regime from a featured DataFrame.
    df must already have features built via build_features()
    Returns: "TRENDING", "RANGING", or "VOLATILE"

    How it works:
    - ADX > threshold  = strong trend (TRENDING)
    - ATR spike        = high volatility (VOLATILE)
    - everything else  = ranging market (RANGING)
    
    ADX threshold is strategy-specific:
    - stable: 25 (conservative)
    - risky1: 20 (lower to allow more momentum trades)
    - risky2: 20 (crypto trends are valid at lower ADX)
    """
    if df.empty or len(df) < 2:
        return RANGING

    latest = df.iloc[-1]
    threshold = ADX_THRESHOLD.get(strategy, ADX_THRESHOLD["default"])

    # Check for volatility
    atr_mean = df['atr'].rolling(window=20).mean().iloc[-1]
    if latest['atr'] > atr_mean * 2:
        return VOLATILE

    # Trend check
    if len(df) >= 14:
        adx_value = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().iloc[-1]
        if adx_value > threshold:
            return TRENDING
    else:
        # SMA fallback if not enough bars for ADX
        if latest['sma_20'] > latest['sma_50'] * 1.01 or \
           latest['sma_20'] < latest['sma_50'] * 0.99:
            return TRENDING

    return RANGING


def get_regime_for_strategy(df, strategy, vix=None):
    """
    Returns whether the current regime is suitable for a given strategy.
    strategy: "stable" | "risky1" | "risky2"
    vix: optional — blocks risky bots when fear is high
    Returns: True if conditions are good, False if bot should sit out
    """
    regime = detect_regime(df, strategy)

    if vix is not None:
        if vix > 30:
            if strategy in ["risky1", "risky2"]:
                print(f"VIX={vix:.1f} — high fear, {strategy} sitting out")
                return False

    if strategy == "stable":
        return regime in [RANGING, TRENDING]

    elif strategy == "risky1":
        return regime == TRENDING

    elif strategy == "risky2":
        return regime in [TRENDING, VOLATILE]

    return True


def regime_summary(df):
    """
    Returns a readable summary of current market conditions.
    Used by the dashboard to display regime status.
    """
    regime = detect_regime(df)
    descriptions = {
        TRENDING: "Market is trending — momentum conditions favorable",
        RANGING:  "Market is ranging — grid trading conditions favorable",
        VOLATILE: "Market is volatile — reduced position sizing recommended",
    }
    return {
        "regime":      regime,
        "description": descriptions[regime]
    }


def get_vix():
    """
    Returns the latest VIX value from FRED API.
    VIX measures market fear — higher is worse.
    Returns None on failure so it never crashes the system.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
        response = requests.get(url, timeout=10)
        lines = response.text.strip().split("\n")
        for line in reversed(lines[1:]):
            date, value = line.split(",")
            if value.strip() != ".":
                return float(value.strip())
        return None
    except Exception:
        return None



















#  def get_regime_for_strategy(df, strategy):
#     """
#     Returns whether the current regime is suitable for a given strategy
#     strategy: "stable" | "risky1" | "risky2"
#     Returns : True if conditions are good, False if bot should sit out
#     """
#     regime = detect_regime(df)

    # if strategy == "stable":
    #     # Grid trading works best in ranging markets
    #     # Still okay in trending — just less optimal but it sits out when volatile
    #     return regime in [RANGING, TRENDING]

    # elif strategy == "risky1":
    #     # Momentum works best when trending
    #     # Sit out when ranging or volatile
    #     return regime == TRENDING

    # elif strategy == "risky2":
    #     # Crypto RL bot is trained to handle volatility and it only sits out when completely ranging with no movement
    #     return regime in [TRENDING, VOLATILE]

    # return True  # default to allowing trade if unknown strategy

