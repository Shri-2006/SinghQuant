# SinghQuant: Multi-Strategy Paper-Trading System with ML and Risk Management

SinghQuant is an automated **paper-trading** platform that runs three strategies side by side against one Alpaca paper account, with Polygon market data, an SQLite audit trail, a Streamlit dashboard, and a multi-layer risk stack.

> **Paper trading only.** Every strategy targets `https://paper-api.alpaca.markets`. The broker wrapper refuses to construct a live client unless the environment variable `SINGHQUANT_ALLOW_LIVE_TRADING=I_UNDERSTAND_REAL_MONEY` is set *and* the strategy is flipped out of paper mode in `core/config.py`. `python run.py` prints the endpoint each strategy will use before anything starts.

> **Audit (September 2026).** The project was inactive for a while and was restored and audited, then the fixes were reviewed adversarially in a second pass. The code has been validated only against a fake broker; it is **not** yet proven against Alpaca's paper API and should be run supervised first (see audit section 11.5). The full engineering record, including root causes for the historical anomalies (the July 7 "-94.95%" kill switch, the 0.1446-vs-51.2 TSLA quantity mismatch, the buy/sell churn), is in [ENGINEERING_AUDIT.md](ENGINEERING_AUDIT.md). The decision history that led here is in [DECISION_LOG.md](DECISION_LOG.md).

## Strategies

| Strategy | Universe | Model | Timeframe | Notes |
|---|---|---|---|---|
| **stable** | SPY, QQQ, AAPL, MSFT, DIA, IWM, SAP, AMZN | XGBoost classifier | Daily bars, market hours | $1,000 notional budget, 20% per symbol, 5% stop, 15% kill switch |
| **risky1** | NVDA, AMD, TSLA, META | XGBoost classifier + momentum exit | Daily bars, market hours | $200 budget, 25% per symbol, 8% stop, 30% kill switch |
| **risky2** | BTC, ETH, SOL, AVAX, LINK, ADA, XRP, DOGE (via Alpaca crypto) | PPO (Stable-Baselines3), one model per ticker | Daily bars, 24/7 | $200 budget, 25% per symbol, 12% stop, 30% kill switch, 30-minute minimum hold. Disabled by default (`FEATURE_FLAGS`) until models are trained |

### How the ML works

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| **Used by** | stable, risky1 | risky2 |
| **How it learns** | Learns from labelled historical examples (next-day close up or down) | Learns by trial and error in a simulated environment |
| **Output** | Predicts BUY (1) or SELL (0) for the next bar | Chooses HOLD, BUY, or SELL from the current state |
| **Algorithm** | XGBoost | PPO via Stable Baselines3 |
| **Why** | Industry standard for tabular financial data | Stable RL algorithm for discrete action spaces |

Unsupervised learning has no notion of BUY or SELL, so it is not used as a primary model.

## Architecture

```
run.py
 ├─ APScheduler: retrain XGBoost models every RETRAIN_INTERVAL_DAYS (running bots reload on change)
 ├─ thread "stable"  ─┐
 ├─ thread "risky1"  ─┼─ strategies/common.py  (ONE shared engine)
 └─ thread "risky2"  ─┘        │
                                │  per cycle, per ticker:
   data/polygon_fetcher ──► core/features ──► data/sentiment_fetcher
        ──► models/regime_detector (ADX / ATR / VIX / SearXNG macro gate)
        ──► strategy decision (XGBoost or PPO)
        ──► core/portfolio   STRATEGY-level ledger: own positions, own cash budget, own peak equity
        ──► metrics/risk_manager  sizing, stop loss, ATR-scaled thresholds (as fraction of price)
        ──► metrics/equity_curve_filter  FULL / THROTTLE / HALT (HALT blocks entries, never exits)
        ──► anti-churn state machine (no entry while exit condition is true, one action per bar, min hold)
        ──► core/execution   own-quantity orders only, fills read back, partial/rejected handled
        ──► paper_trading/alpaca_paper (Alpaca REST, paper endpoint)
        ──► trades.db (trades, orders, positions, strategy_state, equity_snapshots, heartbeat)
        ──► dashboard/streamlit_app_v2.py, data/discord_notifier
```

Key design points established by the audit:

- **Strategy ownership.** Alpaca positions are account-wide, so the system keeps its own per-strategy ledger in SQLite. A strategy can only sell what its ledger says it owns; a kill switch closes only that strategy's positions; drawdown is measured on the strategy's own equity (cash budget + owned positions), not the shared account number.
- **Units.** Order quantities are shares/coins everywhere. Dollar budgets are converted exactly once, at sizing. Every `trades` row logs the signal price, the requested quantity, the broker order id, the status, and the actual fill quantity and price.
- **Kill switch.** Fires only after two consecutive critical readings, and implausible equity jumps (more than 50% in one cycle, NaN, zero) are treated as suspect rather than acted on. A halted strategy stays halted across restarts until `python tools/reset_kill_switch.py <strategy>` is run.
- **Signal price vs fill price.** On Polygon's free tier the latest daily bar is the previous close; that is the *signal* price and is logged as such. Fill prices come from Alpaca and are logged separately.

## Tech stack

Alpaca (paper trading and execution), Polygon.io (stock/ETF/crypto OHLCV), SQLite in WAL mode, pandas / ta, XGBoost, scikit-learn, Stable-Baselines3 + Gymnasium + PyTorch, vectorbt (backtesting), Streamlit + Plotly (dashboard), APScheduler (retraining), FRED (VIX), SearXNG + SAP AI Core / Google Gemini (optional macro classifier), Discord webhooks (heartbeats and alerts).

## Project structure

```
core/         config.py, features.py, logger.py (SQLite + schema migration), portfolio.py (strategy ledger), execution.py (broker adapter)
strategies/   common.py (shared engine), stable.py, risky1.py, risky2.py
metrics/      risk_manager.py, equity_curve_filter.py, risk.py
models/       train.py (XGBoost), retrain.py, rl_environment.py, rl_train.py, regime_detector.py
data/         polygon_fetcher.py, sentiment_fetcher.py, macro_fetcher.py, discord_notifier.py
paper_trading/alpaca_paper.py
backtesting/  engine.py, run_backtest.py
dashboard/    streamlit_app_v2.py (current), streamlit_app.py (legacy), compare.py
tools/        reset_kill_switch.py
tests/        68 original unit tests + 56 regression tests (test_execution.py, test_engine.py, test_migration_and_startup.py, test_second_pass.py) with a fake broker (fakes.py)
```

## Setup

**Prerequisites:** Python 3.11 or newer (3.11 to 3.14 verified), pip. Git is optional.

**1. Clone or unzip**
```bash
git clone https://github.com/Shri-2006/SinghQuant.git
cd SinghQuant
```

**2. Create a virtual environment and install**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
bash setup.sh            # compatible ranges (requirements.txt)
bash setup.sh --pinned   # original exact pins, Python 3.11 only
```
`setup.sh` installs `alpaca-trade-api` with `--no-deps` because its declared pins (`websockets<11`, `aiohttp==3.8.2`) conflict with `polygon-api-client` and do not build on Python 3.12+. On Windows without bash, run the four `pip install` lines from `setup.sh` by hand.

**3. Create `.env`** (never committed; values omitted on purpose)
```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
POLYGON_API_KEY=
# optional
POLYGON_API_KEY_RISKY1=
POLYGON_API_KEY_RISKY2=
DISCORD_WEBHOOK_URL=
SAP_AUTH_URL=
SAP_CLIENT_ID=
SAP_CLIENT_SECRET=
SAP_AI_API_URL=
SAP_ORCHESTRATION_DEPLOYMENT_ID=
RESOURCE_GROUP=default
GEMINI_API_KEY=
# optional per-strategy Alpaca paper accounts (strongest isolation)
ALPACA_API_KEY_STABLE= / ALPACA_SECRET_KEY_STABLE= (and _RISKY1, _RISKY2)
# operational switches
SINGHQUANT_DB_PATH=                 # alternative database file (tests use a temp file automatically)
SINGHQUANT_ADOPT_BROKER_POSITIONS=  # set to 1 once to adopt pre-existing broker positions into the ledger
```

**4. Train the models** (none are shipped; they are gitignored)
```bash
python models/train.py
python -c "from models.rl_train import train_all_tickers; train_all_tickers(timesteps=100000)"
```

**5. Run the tests**
```bash
python -m pytest tests/ -q
```

**6. Run the system**
```bash
python run.py
streamlit run dashboard/streamlit_app_v2.py
```
or with Docker: `docker compose up` (bots + dashboard on port 8501, sharing `trades.db`).

## Migrating an existing `trades.db`

The schema migration is additive: old rows are untouched, new columns and tables are added on first start. SELL rows written before the audit stored the position's **dollar market value** in the `quantity` column (recognisable by `status IS NULL`); rows written after store shares. If the broker already holds positions when the ledger is empty, the bot reports them and leaves them alone; set `SINGHQUANT_ADOPT_BROKER_POSITIONS=1` for one start to assign them to the strategy whose universe contains the symbol.

## Audit evidence

`2026-09-14T16-57_export.csv` (6,276 rows, April 15 to September 14, 2026) is the export of the live `trades` table that the audit was verified against. It is gitignored (`*_export.csv`). The analysis scripts used on it are reproduced in `ENGINEERING_AUDIT.md` section 3.

## Known limitations

- No models or historical database are included in the repository.
- Polygon free tier means the "latest" price is the previous daily close; intraday stops cannot trigger until the next bar.
- `feeds.reuters.com` is offline, so sentiment comes from the Yahoo RSS feed only; the SearXNG macro breaker requires a local SearXNG at `localhost:8080` and otherwise returns CLEAR.
- The XGBoost features are raw price levels (not returns), and the risky1 model was historically trained on SPY only; retraining with the current code uses the strategy's own assets and a chronological hold-out. Backtests in `RESULTS.md` were in-sample; `backtesting/run_backtest.py` now defaults to the hold-out tail (use `--full` for the old behaviour).
- risky2's PPO reward uses the next bar's price for the "missed opportunity" penalty (hindsight shaping during training only).
- Risk-free rate fetch always falls back to 4% (Polygon has no trades for the TNX index).

## Deployment history

- Oracle Cloud ARM free tier (4 CPU, 24 GB) with an rsync backup to a home server; an Azure VM was used earlier. Public addresses were removed from this README during the audit.

## Progress and history

The week-by-week build notes are preserved in `DECISION_LOG.md`, `WEEK6_RISK_PLAN.md`, `WEEK7_ADVANCED_RISK.md`, `WEEK8_ADVANCED_RISK.md`, `Week11_New_Bots.md`, `strategies/WEEK9DYNAMIC_SCREENER.md`, `OPTIMIZATION_RECOMMENDATIONS.md`, and `RESULTS.md` (whose backtest numbers should be read as in-sample; see the audit).

### Nice sources to read up on
https://www.investopedia.com/terms/b/bollingerbands.asp
