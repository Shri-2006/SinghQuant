"""
XGBoost training / loading for the supervised strategies (stable, risky1).

Audit changes (ENGINEERING_AUDIT.md issues M-02, M-03, H-06, ML section):
  * FEATURE_COLUMNS is an explicit, ordered list shared by training and live
    inference; `features_for_model` raises if the layout does not match the
    model's expected input width.
  * The train/test split is now time-based PER TICKER before stacking, so the
    test set is the most recent 20% of each series instead of "the last
    ticker in the list" (which leaked correlated dates across tickers).
  * Sentiment is stamped as 0.0 during training: the original stamped today's
    score on two years of history, which carried no information.
  * risky1 trains on RISKY1_ASSETS instead of the ["SPY"] placeholder.
  * ModelHandle reloads a .pkl when its mtime changes so retraining takes
    effect without a restart; missing models no longer crash import.
  * Phase 2 I-3: models live in SINGHQUANT_MODEL_DIR and are published and
    verified through models/artifacts.py (manifest + SHA-256). ModelHandle
    only returns a model whose hash matches the manifest and whose feature
    width matches FEATURE_COLUMNS; otherwise the strategy takes no new
    entries (existing positions are still risk-managed).
"""
import os
from datetime import datetime, timezone
import pickle #to save python objects to file and go back. XGBoost will use .pkl file type
import numpy as np #numpy is used by XGBoost and sklearn for math operations in arrays
import pandas as pd #DataFrames library
from core.features import build_features #OHLCV into indicators, i built this
from core.config import STABLE_ASSETS, RISKY1_ASSETS, RETRAIN_INTERVAL_DAYS

from models import artifacts

#Location of saved trained models (SINGHQUANT_MODEL_DIR, default: this folder)
MODEL_DIR = artifacts.model_dir()

# Exact feature layout the models are trained on and fed at inference.
# Order matters: XGBoost consumes a positional numpy array.
FEATURE_COLUMNS = [
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi', 'momentum_5', 'momentum_15',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'zscore',
    'volume_change', 'volume_ma_20', 'volume_ratio',
    'sentiment',
]
NON_FEATURE_COLUMNS = ['label', 'open', 'high', 'low', 'close', 'volume']


class FeatureMismatch(ValueError):
    """Raised when the live feature row does not match what the model expects."""


def features_for_model(df, model=None):
    """
    Returns the feature matrix (rows x len(FEATURE_COLUMNS)) from a featured
    DataFrame, validating that every expected column exists, contains only
    finite values, and (if `model` is given) matches its input width.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise FeatureMismatch(f"missing feature columns: {missing}")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise FeatureMismatch("feature matrix contains NaN or inf")
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != X.shape[1]:
        raise FeatureMismatch(f"model expects {expected} features, got {X.shape[1]} ({FEATURE_COLUMNS})")
    return X


def create_labels(df):
    """
    Creates a buy and sell label from the price data.
    If tomorrows close > todays close -> 1 and buy. If tomorrows close < todays close -> sell or hold
    The last row has no "tomorrow" and is dropped (it used to get a bogus 0 label).
    """
    df=df.copy() #make a copy and edit that dataframe
    next_close = df['close'].shift(-1)
    df['label']= (next_close > df['close']).astype(int)
    df = df[next_close.notna()]
    return df


def prepare_data(df, ticker, sentiment_score=0.0):
    """
    raw data-> features -> labels-> data for training
    Features and Labels will be returned in numpy arrays.
    Sentiment is a neutral constant during training (see module docstring).
    """
    from data.sentiment_fetcher import add_sentiment_to_df
    df=build_features(df)
    df=add_sentiment_to_df(df,ticker, score=sentiment_score)
    df=create_labels(df)
    X=features_for_model(df)
    Y=df['label'].values
    return X,Y


def time_split(X, Y, test_size=0.2):
    """Chronological split of one ticker's rows: first (1-test_size) train, rest test."""
    n = len(X)
    cut = int(n * (1 - test_size))
    return X[:cut], X[cut:], Y[:cut], Y[cut:]


def build_model():
    from xgboost import XGBClassifier
    # `use_label_encoder` was removed in XGBoost 2.x (it only produced a warning).
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss')


def train_model(X_train, Y_train, X_test=None, Y_test=None):
    """
    Trains the classifier and returns (model, accuracy on the held-out set or None).
    """
    from sklearn.metrics import accuracy_score
    model = build_model()
    model.fit(X_train, Y_train)
    accuracy = None
    if X_test is not None and len(X_test):
        accuracy = accuracy_score(Y_test, model.predict(X_test))
        print(f"Model Accuracy (out-of-time): {accuracy: .2%}")
    return model, accuracy


def _strategy_for(filename):
    return filename.split("_model")[0] if "_model" in filename else None


def save_model(model, filename, version=None, meta=None, directory=None):
    """
    Publishes a trained model atomically (temp file + fsync + content-addressed
    name) and records it in the manifest with its SHA-256. Running bots pick it
    up on their next cycle. The filename is the LOGICAL name, e.g. "stable_model.pkl".
    """
    def write(tmp):
        with open(tmp, 'wb') as f:
            pickle.dump(model, f)
    version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifacts.publish(filename, write, _strategy_for(filename), version, meta, directory)


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(filename):
    """
    Loads a model for offline use (backtests). A registered model is verified
    against the manifest first; an unregistered legacy file in the model
    directory is loaded with a warning (live trading never does this).
    """
    try:
        _, path = artifacts.verify(filename)
    except artifacts.ArtifactError as e:
        if e.status != artifacts.UNREGISTERED:
            raise
        path = os.path.join(artifacts.model_dir(), filename)
        print(f"[models] WARNING: {filename} is not in the manifest; loading the unverified file for offline use")
    return _load_pickle(path)


def check_feature_width(model):
    """None if `model` accepts the live feature layout, else a description of the mismatch."""
    if not hasattr(model, "predict"):
        return "object has no predict()"
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != len(FEATURE_COLUMNS):
        return f"model expects {expected} features, live inference provides {len(FEATURE_COLUMNS)}"
    return None


class ModelHandle(artifacts.ArtifactHandle):
    """
    Verified, hot-reloading XGBoost model for a strategy. `.model` is None
    unless the artifact is registered, present, matches its SHA-256 and has
    the right feature width; `.status` / `.detail` explain why not.
    """
    def __init__(self, filename, directory=None):
        super().__init__(filename, _load_pickle, check_feature_width, directory)
        self.filename = filename


def train_and_save(strategy, start, end, test_size=0.2, delay=12):
    """
    Fetches data, prepares, trains model, and then saves the model.
    The possible strategies are "stable" or "risky1" (risky2 is done differently)
    start and end date strings are in "YYYY-MM-DD" format
    """
    from data.polygon_fetcher import get_multiple_tickers
    print(f"Training {strategy} model...")
    if strategy== "stable":
        tickers= STABLE_ASSETS
        filename="stable_model.pkl"
    elif strategy == "risky1":
        tickers= RISKY1_ASSETS
        filename="risky1_model.pkl"
    else:
        print(f"unknown strategy, are you sure you have the right name? : {strategy}")
        return None
    Xtr, Xte, Ytr, Yte = [], [], [], []
    data=get_multiple_tickers(tickers,start,end, delay=delay)
    for t, df in data.items():
        X,Y=prepare_data(df,t)
        a,b,c,d = time_split(X, Y, test_size)
        Xtr.append(a); Xte.append(b); Ytr.append(c); Yte.append(d)
    if not Xtr:
        print(f"No training data for {strategy}; model not updated")
        return None
    model,accuracy=train_model(np.vstack(Xtr),np.concatenate(Ytr),np.vstack(Xte),np.concatenate(Yte))
    import sklearn, xgboost
    save_model(model, filename, meta={
        "train_start": start, "train_end": end, "tickers": list(data.keys()),
        "out_of_time_accuracy": None if accuracy is None else float(accuracy),
        "feature_columns": FEATURE_COLUMNS, "n_features": len(FEATURE_COLUMNS),
        "xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__,
    })
    print(f"{strategy} model was trained; out-of-time accuracy {accuracy: .2%}")
    return model, accuracy


if __name__ == "__main__":
    from models.retrain import get_date_range
    s, e = get_date_range()
    train_and_save("stable", s, e)
    train_and_save("risky1", s, e)
