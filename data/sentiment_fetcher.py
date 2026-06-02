import time
import requests
from datetime import date
from textblob import TextBlob
from polygon import RESTClient
from core.config import POLYGON_API_KEY

SEARXNG_URL = "http://localhost:8080/search"

# Daily cache per ticker
_sentiment_cache = {}
_cache_date = {}

# Polygon fallback client
_polygon_client = RESTClient(api_key=POLYGON_API_KEY)

def _get_sentiment_from_searxng(ticker):
    """
    Fetches news headlines from local SearXNG and scores sentiment.
    Returns float score or None if failed.
    """
    try:
        # Clean ticker for search (remove X: prefix for crypto)
        clean_ticker = ticker.replace("X:", "")
        query = f"{clean_ticker} stock news today"
        response = requests.get(
            SEARXNG_URL,
            params={"q": query, "format": "json"},
            timeout=10
        )
        data = response.json()
        texts = []
        for result in data.get("results", []):
            texts.append(result.get("title", ""))
            texts.append(result.get("content", ""))

        if not texts:
            return None

        scores = [TextBlob(t).sentiment.polarity for t in texts if t.strip()]
        return sum(scores) / len(scores) if scores else None

    except Exception:
        return None


def _get_sentiment_from_polygon(ticker, limit=10):
    """
    Fallback — fetches news from Polygon API and scores sentiment.
    Returns float score or None if failed.
    """
    try:
        news = _polygon_client.list_ticker_news(ticker, limit=limit)
        scores = []
        for article in news:
            score = TextBlob(article.title).sentiment.polarity
            scores.append(score)
        return sum(scores) / len(scores) if scores else None
    except Exception as e:
        print(f"Error, Warning the sentiment failed to fetch for {ticker}, using neutral 0.0 \n-{e}")
        return None


def get_sentiment(ticker):
    """
    Returns sentiment score for a ticker — cached once per day.
    Tries SearXNG first (local, unlimited), falls back to Polygon.
    Returns 0.0 (neutral) if both fail.
    """
    today = date.today()

    # Return cached result if from today
    if _cache_date.get(ticker) == today and ticker in _sentiment_cache:
        return _sentiment_cache[ticker]

    # Try SearXNG first
    score = _get_sentiment_from_searxng(ticker)

    # Fall back to Polygon if SearXNG failed
    if score is None:
        score = _get_sentiment_from_polygon(ticker)

    # Default to neutral if both failed
    if score is None:
        score = 0.0

    # Cache result
    _sentiment_cache[ticker] = score
    _cache_date[ticker] = today
    return score


def get_sentiment_label(score):
    """Converts numeric score to readable label"""
    if score > .1:
        return "POSITIVE"
    elif score < -.1:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def add_sentiment_to_df(df, ticker):
    """
    Adds sentiment score column to any OHLCV dataframe.
    Called from build_features before ML training.
    """
    score = get_sentiment(ticker)
    df = df.copy()
    df['sentiment'] = score
    return df