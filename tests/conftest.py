"""
Shared pytest setup.

* Dummy API keys so modules that build clients at import do not crash.
* SINGHQUANT_DB_PATH points the module-level database at a temp file so the
  test-suite never touches the project's real trades.db.
"""
import os
import tempfile

os.environ.setdefault("POLYGON_API_KEY", "test_key")
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_key")
os.environ.pop("SINGHQUANT_ALLOW_LIVE_TRADING", None)
os.environ.pop("SINGHQUANT_ADOPT_BROKER_POSITIONS", None)
_TMP_DB_DIR = tempfile.mkdtemp(prefix="singhquant-tests-")
os.environ["SINGHQUANT_DB_PATH"] = os.path.join(_TMP_DB_DIR, "trades_test.db")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from core.logger import init_db  # noqa: E402
from models.train import FEATURE_COLUMNS  # noqa: E402


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "trades.db")
    init_db(p)
    return p


def make_featured_df(rows=60, price=500.0, momentum_5=0.01, atr=5.0, start="2026-08-01", nan_last=False):
    """A featured DataFrame with every FEATURE_COLUMNS column; last close == price."""
    idx = pd.date_range(start, periods=rows, freq="D")
    closes = np.linspace(price * 0.9, price, rows)
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.01, "low": closes * 0.99, "close": closes,
        "volume": 1_000_000.0,
    }, index=idx)
    for c in FEATURE_COLUMNS:
        df[c] = 0.5
    df["sma_20"] = closes; df["sma_50"] = closes; df["ema_12"] = closes; df["ema_26"] = closes
    df["bb_upper"] = closes * 1.02; df["bb_middle"] = closes; df["bb_lower"] = closes * 0.98
    df["atr"] = atr
    df["momentum_5"] = momentum_5
    df["volume_ma_20"] = 1_000_000.0
    df["volume_ratio"] = 1.0
    if nan_last:
        df.loc[df.index[-1], "rsi"] = np.nan
    return df


class StubModel:
    """Predicts a constant; mimics the sklearn/xgboost surface the engine uses."""
    def __init__(self, prediction=1, n_features=len(FEATURE_COLUMNS)):
        self.prediction = prediction
        self.n_features_in_ = n_features
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.array([self.prediction] * len(X))


class StubHandle:
    def __init__(self, model):
        self.model = model
