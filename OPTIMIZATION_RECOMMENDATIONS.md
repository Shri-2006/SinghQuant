# SinghQuant — Optimization & Infrastructure Recommendations

## 🔴 High Priority (Implement Now)

### 1. Normalize Input Data (Critical)
- Do NOT feed raw prices into models
- Convert price-based features into normalized forms:
  - Percentage returns: `returns = close.pct_change()`
  - Optional later: rolling z-score normalization
- Why: prevents scale bias across assets, ensures model learns
  patterns not price levels

### 2. Correct Slippage Modeling
- Apply slippage per side, not once per trade:
  - Buy:  `price * (1 + 0.0003)`
  - Sell: `price * (1 - 0.0003)`
- Result: 0.06% total round-trip cost
- Why: more realistic backtesting, prevents inflated metrics

### 3. Constrain RL Bot (Risky2)
- Limit action space to [BUY, SELL, HOLD] only
- Avoid continuous position sizing or leverage-based actions
- Use risk-adjusted reward (Sharpe-like or drawdown-aware)
- Why: prevents RL from exploiting unrealistic backtest conditions,
  improves generalization to live trading

### 4. System Heartbeat Monitoring via Discord/Telegram
- Daily status ping including:
  - System alive/dead
  - Current portfolio value
  - Last successful data sync timestamp
  - Last trade execution time
- Why: free-tier cloud systems can silently fail, enables passive
  monitoring without logging into server

---

## 🟡 Medium Priority (After System Stabilizes)

### 5. Log Volatility & Regime Signals
- Track rolling volatility (ATR, std dev of returns)
- Track regime indicators (trend vs sideways)
- Store alongside model inputs
- Why: prepares system for adaptive behavior, enables analysis
  of model performance by market condition

### 6. Improve Feature Engineering
- Add momentum indicators (returns over multiple windows)
- Add volatility measures
- Add volume-based signals
- Keep feature set simple and interpretable
- Why: strong features > complex models

---

## 🔵 Future Enhancements (Phase 2+)

### 7. Event-Driven Retraining
- Replace fixed 14-day retraining with triggers:
  - Volatility spikes (> 2 standard deviations)
  - Regime shifts
- Keep periodic fallback (every 2-4 weeks)
- Why: faster adaptation to market shocks

### 8. Advanced RL Improvements
- Expand action space cautiously after validation only
- Introduce transaction cost penalties
- Add position sizing constraints
- Perform strict out-of-sample testing
- Why: prevents overfitting and unrealistic strategies

---

## 🧠 Guiding Principles
- Reliability > Complexity
- Realism > Backtest performance
- Simplicity > Over-engineering

## System Goal (Current Phase)
Build a stable, interpretable, and realistically testable
trading system. Only introduce complexity AFTER:
- System runs consistently
- Metrics are trustworthy
- Risk controls are validated

---

## Implementation Order
1. Input normalization (affects all models — do first)
2. Slippage correction (backtesting realism)
3. Discord/Telegram heartbeat (passive monitoring)
4. RL reward function improvement (risky2 quality)
5. Volatility/regime logging (medium term)
6. Feature engineering upgrades (medium term)
7. Event-driven retraining (phase 2)
8. Advanced RL (phase 2)


## 🟡 Week 9/10 — Dynamic Asset Selection for Risky1

### Current state
RISKY1_ASSETS is a static list hardcoded in config.py.
This defeats the purpose of a momentum strategy — the best
momentum stocks change constantly.

### Planned approach
Replace static list with a dynamic screener that runs every cycle:

1. Maintain a universe of ~100 liquid stocks (S&P 500 subset)
2. Score each by momentum signal:
   - Price > 20-day high (breakout)
   - Volume > 1.5x 20-day average (confirmation)
   - RSI between 50-70 (trending but not overbought)
3. Pick top 5 by composite momentum score
4. Trade those 5 tickers that cycle

### Files to create/modify
- data/stock_screener.py — new file, runs momentum scan
- core/config.py — replace RISKY1_ASSETS with RISKY1_UNIVERSE
- strategies/risky1.py — call screener at start of each cycle

### Why this matters
A static list means risky1 trades the same stocks regardless
of market conditions. Dynamic selection means it always trades
whatever has the strongest momentum signal right now.