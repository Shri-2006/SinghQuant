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