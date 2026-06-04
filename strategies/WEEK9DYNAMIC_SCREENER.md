# Week 9/10 — Dynamic Asset Selection

## Overview
Replace static asset lists with dynamic screeners for all three bots.
Each strategy has different criteria reflecting its risk profile.

## Current State
STABLE_ASSETS  = hardcoded list of 8 ETFs/blue chips
RISKY1_ASSETS  = hardcoded list of 5 momentum stocks
RISKY2_ASSETS  = hardcoded list of 8 crypto tickers


## Planned Screeners

### Stable — Strict Quality Screener
Only allows assets meeting ALL stability criteria:
- Market cap > $50B (large cap only)
- Average daily volume > 5M shares
- Beta < 1.2 (low volatility relative to market)
- Listed ETF OR S&P 500 component
- Examples that would pass: SPY, QQQ, AAPL, MSFT, V, JPM
- Examples that would fail: small caps, penny stocks, volatile sector ETFs

### Risky1 — Momentum Screener
Scans a universe of ~200 liquid stocks and picks top 5 by momentum:
- Price > 20-day high (breakout signal)
- Volume > 1.5x 20-day average (confirmation)
- RSI between 50-70 (trending, not overbought)
- Market cap > $5B (avoid illiquid stocks)
- Refreshes every cycle — always trading strongest momentum

### Risky2 — Crypto Volume/Momentum Screener
Scans top 50 crypto by market cap and picks top N by:
- 24-hour volume > threshold
- Price momentum over 7 days
- Available on both Polygon (X: prefix) and Alpaca
- Refreshes daily — follows where crypto volume is moving

## Files to Create/Modify
- data/stock_screener.py  — stable and risky1 screeners
- data/crypto_screener.py — risky2 screener
- core/config.py          — add universe lists, screener params
- strategies/stable.py    — call screener at start of each cycle
- strategies/risky1.py    — call screener at start of each cycle
- strategies/risky2.py    — call screener daily

## Implementation Order
1. risky1 screener first (most impactful, already has momentum logic)
2. risky2 screener second (crypto volume is easy to source)
3. stable screener last (most complex quality filters)

## Notes
- Screener results should be cached per cycle to avoid extra API calls
- Fallback to static lists if screener returns 0 results
- Log which assets were selected each cycle for audit trail
- Beta calculation requires benchmark data (SPY returns)