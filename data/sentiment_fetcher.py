import time
import requests
import feedparser
from datetime import date
from textblob import TextBlob
from core.config import POLYGON_API_KEY

USE_POLYGON_FALLBACK = False  # disabled — too many 429s

_polygon_client = None  # created lazily; importing this module must not need a key


def _get_polygon_client():
    global _polygon_client
    if _polygon_client is None:
        from polygon import RESTClient
        _polygon_client = RESTClient(api_key=POLYGON_API_KEY)
    return _polygon_client

# Daily cache per ticker
_sentiment_cache = {}
_cache_date = {}


def _get_sentiment_from_rss(ticker):
    """
    Fetches news from Yahoo Finance and Reuters RSS feeds.
    Yahoo Finance gives ticker-specific news.
    Reuters gives general market context.
    No API key, no rate limits, free forever.
    NOTE: feeds.reuters.com has been offline for years; feedparser returns an
    empty feed for it, so only the Yahoo feed contributes in practice.
    """
    try:
        clean_ticker = ticker.replace("X:", "")
        texts = []

        # Yahoo Finance ticker-specific RSS
        yahoo_url = f"https://finance.yahoo.com/rss/headline?s={clean_ticker}"
        yahoo_feed = feedparser.parse(yahoo_url)
        for entry in yahoo_feed.entries[:10]:
            texts.append(entry.get("title", ""))
            texts.append(entry.get("summary", ""))

        # Reuters general market news
        reuters_url = "https://feeds.reuters.com/reuters/businessNews"
        reuters_feed = feedparser.parse(reuters_url)
        for entry in reuters_feed.entries[:5]:
            texts.append(entry.get("title", ""))

        if not texts:
            return None

        scores = [TextBlob(t).sentiment.polarity for t in texts if t.strip()]
        return sum(scores) / len(scores) if scores else None

    except Exception:
        return None


def _get_sentiment_from_polygon(ticker, limit=10):
    """
    Polygon fallback — disabled by default due to rate limits.
    """
    if not USE_POLYGON_FALLBACK:
        return None
    try:
        news = _get_polygon_client().list_ticker_news(ticker, limit=limit)
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
    Tries RSS first (Yahoo Finance + Reuters), falls back to Polygon.
    Returns 0.0 (neutral) if both fail.
    """
    today = date.today()

    # Return cached result if from today
    if _cache_date.get(ticker) == today and ticker in _sentiment_cache:
        return _sentiment_cache[ticker]

    # Try RSS first
    score = _get_sentiment_from_rss(ticker)

    # Fall back to Polygon if RSS failed
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


def add_sentiment_to_df(df, ticker, score=None):
    """
    Adds a `sentiment` column (one constant value) to any OHLCV dataframe.

    CAVEAT (audit issue M-03): today's score is stamped onto every row. For
    live inference that is fine (one row matters). For TRAINING it means two
    years of history get today's sentiment, so the column carries no temporal
    information; models.train passes score=0.0 to keep the column neutral and
    the feature layout identical between training and inference.
    """
    if score is None:
        score = get_sentiment(ticker)
    df = df.copy()
    df['sentiment'] = score
    return df
