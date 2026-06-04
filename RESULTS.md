# SinghQuant — Live Trading Results

## Paper Trading Period: April 15 – June 2, 2026 (48 days)

### Stable Bot Performance
| Metric | Value |
|--------|-------|
| Total Trades | 91 |
| Buys | 69 |
| Sells | 22 |
| Realized PnL | +$17.22 |
| Return on Capital | +1.72% |
| Annualized Return | ~13% |
| Max Drawdown | 0% |
| Assets Traded | SPY, QQQ, AAPL, MSFT, DIA, IWM, SAP, AMZN |

### Risky1 Bot Performance
| Metric | Value |
|--------|-------|
| Total Trades | 1 (just started Jun 2) |
| First Trade | Buy 0.2229 NVDA @ $224.36 |
| Assets | NVDA, AMD, TSLA, META |

### Risky2 Bot Performance
| Metric | Value |
|--------|-------|
| Status | Live, cycling crypto 24/7 |
| Strategy | PPO Reinforcement Learning |
| Behavior | Conservative HOLD (retraining needed) |

---

## Backtest Results (2-Year Backtest, Jan 2024 – Jun 2026)

### Stable Bot — XGBoost
| Asset | Return | Sharpe | Max Drawdown | Trades |
|-------|--------|--------|--------------|--------|
| AAPL | +186% | 3.05 | -12.2% | 65 |
| QQQ | +134% | 3.15 | -10.0% | 56 |
| IWM | +99% | 2.80 | -11.7% | 78 |
| DIA | +77% | 3.27 | -9.9% | 67 |
| MSFT | +82% | 1.93 | -19.2% | 53 |
| SPY | +66% | 2.24 | -15.3% | 41 |
| SAP | -43% | -1.33 | -55.9% | 90 |

### Risky1 Bot — XGBoost Momentum
| Asset | Return | Sharpe | Max Drawdown | Trades |
|-------|--------|--------|--------------|--------|
| SPY | +138% | 4.86 | -4.4% | 75 |

---

## Context
- Market benchmark (S&P 500) returned +15.73% over same 48-day period
- Bot underperformed due to historic V-shaped bull rally Apr-Jun 2026
- Risk-adjusted annualized return of ~13% is solid baseline
- Risky1 assets were empty until June 2 — missed entire rally
- VIX/macro signals were overly restrictive (now fixed)

---

## Infrastructure
- Oracle Cloud ARM (4 CPU, 24GB RAM, free tier)
- 3 concurrent bots running 24/7
- SQLite WAL mode for concurrent reads/writes
- Automated nightly backups to ProDesk via rsync


# SinghQuant — Live Trading Results

## Paper Trading Period: April 15 – June 3, 2026 (49 days)

### Stable Bot Performance
| Metric | Value |
|--------|-------|
| Total Trades | 94+ |
| Buys | 71+ |
| Sells | 23+ |
| Realized PnL | +$20.39 |
| Return on Capital | +2.04% |
| Annualized Return | ~15% |
| Max Drawdown | 0% |
| Assets Traded | SPY, QQQ, AAPL, MSFT, DIA, IWM, SAP, AMZN |

### Risky1 Bot Performance
| Metric | Value |
|--------|-------|
| Status | Live as of June 2 |
| First Trade | Buy 0.2229 NVDA @ $224.36 |
| First Sell | TSLA @ $423.74 |
| Assets | NVDA, AMD, TSLA, META |
| Notes | Missed full rally — assets empty until June 2 |

### Risky2 Bot Performance
| Metric | Value |
|--------|-------|
| Status | Live, cycling crypto 24/7 |
| Strategy | PPO Reinforcement Learning |
| Assets | BTC, ETH, SOL, AVAX, LINK, ADA, XRP, DOGE |
| First Trades | BTC @ $66,704 · SOL @ $74.12 · AVAX @ $8.91 |
| Model | Retrained June 3 — per-ticker models in progress |

---

## Backtest Results (2-Year Backtest, Jan 2024 – Jun 2026)

### Stable Bot — XGBoost
| Asset | Return | Sharpe | Max Drawdown | Trades |
|-------|--------|--------|--------------|--------|
| AAPL  | +186%  | 3.05   | -12.2%       | 65     |
| QQQ   | +134%  | 3.15   | -10.0%       | 56     |
| IWM   | +99%   | 2.80   | -11.7%       | 78     |
| DIA   | +77%   | 3.27   | -9.9%        | 67     |
| MSFT  | +82%   | 1.93   | -19.2%       | 53     |
| SPY   | +66%   | 2.24   | -15.3%       | 41     |
| SAP   | -43%   | -1.33  | -55.9%       | 90     |

### Risky1 Bot — XGBoost Momentum
| Asset | Return | Sharpe | Max Drawdown | Trades |
|-------|--------|--------|--------------|--------|
| SPY   | +138%  | 4.86   | -4.4%        | 75     |

---

## System Fixes Applied June 2-3, 2026
- Separate Polygon API keys per strategy — rate limits eliminated
- SearXNG local sentiment (Yahoo Finance + Reuters RSS fallback)
- SAP AI Orchestration fixed — correct payload format
- Macro scorer improved — news sources only, negative keywords
- ADX threshold per strategy — risky1/risky2 use 20 vs stable 25
- Risky2 symbol format fixed — X:BTCUSD → BTCUSD for Alpaca
- RL reward function redesigned — agent now actively buys
- Per-ticker PPO models — eliminates price scale bias

---

## Context
- Market benchmark (S&P 500) returned +15.73% over same period
- Bot underperformed during historic V-shaped bull rally Apr-Jun 2026
- Risk-adjusted annualized return ~15% is solid conservative baseline
- Risky1 assets were empty until June 2 — missed entire rally
- VIX/macro signals were overly restrictive (now fixed)
- All 3 bots now actively trading as of June 3, 2026

---

## Infrastructure
- Oracle Cloud ARM (4 CPU, 24GB RAM, free tier forever)
- 3 concurrent bots running 24/7
- SQLite WAL mode for concurrent reads/writes
- Automated nightly backups to ProDesk via rsync over Tailscale
- Discord heartbeat notifications per cycle
- GitHub Actions CI — 68/68 tests passing