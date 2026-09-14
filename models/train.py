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
"""
import os
import pickle #to save python objects to file and go back. XGBoost will use .pkl file type
import numpy as np #numpy is used by XGBoost and sklearn for math operations in arrays
import pandas as pd #DataFrames library
from core.features import build_features #OHLCV into indicators, i built this
from core.config import STABLE_ASSETS, RISKY1_ASSETS, RETRAIN_INTERVAL_DAYS

#Location of saved trained models
MODEL_DIR= os.path.join(os.path.dirname(__file__))

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


def save_model(model,filename):
    """
    Time to save the trained model in .pkl file. The filename is going to be something like "stable_model.pkl"
    """
    path=os.path.join(MODEL_DIR,filename)
    with open(path,'wb') as f:
        pickle.dump(model,f)
    print(f"Model saved to {path}")


def load_model(filename):
    """
    Load the saved model from file.  The filename is going to be something like "stable_model.pkl" again
    """
    path=os.path.join(MODEL_DIR,filename)
    with open(path,'rb')as f:
        model=pickle.load(f)
    return model


class ModelHandle:
    """
    Lazily loads a model file and transparently reloads it when the file
    changes on disk (retraining). `.model` is None when no file exists, and
    strategies then skip entries instead of crashing at import.
    """
    def __init__(self, filename):
        self.filename = filename
        self.path = os.path.join(MODEL_DIR, filename)
        self._model = None
        self._mtime = None

    @property
    def model(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if self._model is None:
                print(f"[models] {self.filename} not found in {MODEL_DIR}; strategy will not open new positions")
            return self._model
        if self._model is None or mtime != self._mtime:
            try:
                self._model = load_model(self.filename)
                self._mtime = mtime
                print(f"[models] loaded {self.filename} (mtime {mtime})")
            except Exception as e:
                print(f"[models] failed to load {self.filename}: {e}")
        return self._model

    def predict(self, X):
        m = self.model
        if m is None:
            return None
        return m.predict(X)


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
    save_model(model,filename)
    print(f"{strategy} model was trained; out-of-time accuracy {accuracy: .2%}")
    return model, accuracy


if __name__ == "__main__":
    from models.retrain import get_date_range
    s, e = get_date_range()
    train_and_save("stable", s, e)
    train_and_save("risky1", s, e)
