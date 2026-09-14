# SinghQuant Engineering Audit

Audit date: 2026-09-14
Audited artifact: `SinghQuant-main.zip` (GitHub download, not a git clone) plus `2026-09-14T16-57_export.csv`, an export of the live `trades` table (6,276 rows, 2026-04-15 to 2026-09-14) that the project owner placed in the folder during the audit.
Auditor: Claude (AI-assisted audit performed with the project owner). This document records what was found, why it happened, how it was fixed, and how each fix was validated. Nothing in it is a performance claim, and every historical number below comes from the export or from arithmetic on the configuration.

---

## 0. Executive summary

SinghQuant is a three-strategy paper-trading system (XGBoost "stable", XGBoost "risky1", PPO "risky2") sharing one Alpaca paper account, one SQLite trade log, and one Polygon daily-bar feed. The anomalies the owner remembered are explained by a small number of structural defects rather than by market losses:

1. **No strategy owns anything.** All three bots read, size, exit and kill-switch against account-wide Alpaca state. On 2026-06-02 risky1 sold stable's AMZN position (risky1 had never bought AMZN). On 2026-07-07 both bots' kill switches liquidated the whole account within 25 seconds of each other.
2. **The kill switch decided on one formula and logged another.** The decision compared account equity with a strategy's notional starting capital; the logged "-94.95%" was drawdown from an in-memory peak. Forty minutes before the switch fired, stable had sold $204 of AAPL at a profit; the reading that fired was not a market loss.
3. **BUY rows log shares; SELL rows log dollars.** Across 3,020 SELL rows, SELL `quantity` divided by the shares bought equals the logged price within 1% at the median. "0.1446 bought, 51.2 sold" is $50 of TSLA bought and $51 of TSLA sold.
4. **Logged prices are the previous session's close.** In all 408 (strategy, asset, day) groups there is exactly one distinct price.
5. **Risky1 oscillated because its momentum exit and its ML entry are independent predicates on the same static bar.** 248 of its 250 SELL rows are momentum exits; the median gap between consecutive trades is 2.36 minutes, one loop cycle. None of those 248 exits recorded a PnL, so the equity-curve throttle never saw them.
6. **Orders were logged before the broker confirmed them.** MSFT shows 13 BUY rows worth $584 in a period where only $191 was ever sold; SAP shows the same stop-loss "close" logged four times in 16 minutes.
7. **ATR scaling used dollars where a fraction of price was expected**, so every kill and warning threshold was silently 1.5x wider than configured.
8. **Risky2 (PPO) observed the oldest bar of its 200-day window, sampled actions randomly, had no risk controls, and was configured for the live Alpaca endpoint.**
9. **Scheduled retraining never ran** (job added paused), and running bots never reloaded models anyway.

The deployed instance was still running the old code during the audit: the export's last rows are a risky1 TSLA BUY at 16:47:34 UTC and SELL at 16:49:55 UTC on the audit day.

All of the above is fixed in this working copy, covered by 124 tests (68 original, 41 first-pass and 15 second-pass regression tests against a fake broker; 123 run in CI, one network-dependent test deselected), and documented below. A second, adversarial pass over the first-pass fixes found nine further defects in the new code (section 11), all fixed and tested. What could not be fixed from a ZIP (training models, the missing live database, the Alpaca-side cause of the July 7 reading) is stated in sections 9 and 11.5. **The system is not declared deployment-ready**: nothing has been exercised against the real broker.

---

## 1. Phase 0: Repository health and restoration audit

### 1.1 Inventory

| Area | Files | Notes |
|---|---|---|
| Entry point | `run.py` | Starts APScheduler retrain job, then one daemon thread per strategy |
| Strategies | `strategies/stable.py`, `strategies/risky1.py`, `strategies/risky2.py` | stable and risky1 were near-identical copies (about 230 lines each); risky2 a separate design |
| Core | `core/config.py`, `core/features.py`, `core/logger.py` | Config, indicator features, SQLite logging |
| Data | `data/polygon_fetcher.py`, `data/sentiment_fetcher.py`, `data/macro_fetcher.py`, `data/discord_notifier.py` | Polygon OHLCV, RSS+TextBlob sentiment, SearXNG/SAP/Gemini macro, Discord webhooks |
| Models | `models/train.py`, `models/retrain.py`, `models/rl_environment.py`, `models/rl_train.py`, `models/regime_detector.py` | XGBoost, retrain scheduler, Gym env, PPO, ADX/VIX regime |
| Risk | `metrics/risk_manager.py`, `metrics/equity_curve_filter.py`, `metrics/risk.py` | Kill-switch thresholds, FULL/THROTTLE/HALT, Sharpe/Sortino/MDD |
| Broker | `paper_trading/alpaca_paper.py` | Thin wrapper around `alpaca-trade-api` |
| Backtest | `backtesting/engine.py`, `backtesting/run_backtest.py` | vectorbt |
| Dashboard | `dashboard/streamlit_app.py`, `dashboard/streamlit_app_v2.py`, `dashboard/compare.py` | Two Streamlit apps; compose ran v1 |
| Tests | `tests/` (6 files, 68 tests) | Features, metrics, risk manager, regime, RL env, backtest engine |
| Ops | `Dockerfile`, `docker-compose.yml`, `setup.sh`, `.github/workflows/test.yml` | Python 3.11 |
| Docs | `README.md`, `DECISION_LOG.md`, `RESULTS.md`, `OPTIMIZATION_RECOMMENDATIONS.md`, `ORACLE_SETUP.md`, `WEEK*.md` | The decision log was used as evidence |

**Missing from the ZIP (gitignored):** `.env`, `trades.db`, `models/*.pkl`, `models/risky2_*.zip`, `backtesting/results/`, `*.log`. The trade-log export supplied by the owner substitutes for `trades.db` as evidence but is not a database the system can run on.

### 1.2 How the project is meant to run

1. `bash setup.sh` (installs `requirements.txt`, then reinstalls `alpaca-trade-api` and `websockets` to work around a conflict).
2. `.env` with Alpaca and Polygon keys (optional Discord, SAP AI Core, Gemini).
3. Train models (`models/train.py` had its `__main__` guard commented out, so the README's instruction did nothing; RL via `train_all_tickers()`).
4. `python run.py`; `streamlit run dashboard/streamlit_app.py`.

### 1.3 Environment findings on the audit machine

| Item | Finding |
|---|---|
| Python | 3.14.7 only (project targets 3.11; README said 3.10+) |
| git | Not installed; folder is not a repository. A pristine copy of the original tree was snapshotted before any change so every modification could be diffed (section 8.4) |
| Pinned requirements on 3.14 | `numpy==2.2.6` has no 3.14 wheel and fails to build without a C compiler, so the pinned set cannot install here |
| Unpinned install | pandas 3.0.5, numpy 2.5.3, xgboost 3.4.1, scikit-learn 1.9.1, gymnasium 1.3.0, APScheduler 3.11.3, stable-baselines3 2.9.0, torch 2.14 CPU, polygon-api-client 1.16.3, alpaca-trade-api 3.0.0 (`--no-deps`) all import and the original 56 collectable tests pass on them |
| vectorbt | 1.1.0 installs but fails to import with plotly 7 (removed `scattermapbox`); works with `plotly<7` |

### 1.4 Dependency and API-drift findings

- `alpaca-trade-api==3.0.0` is the deprecated Alpaca SDK. It pins `websockets<11`, `aiohttp==3.8.2`, `urllib3<2`, `msgpack==1.0.3`; `aiohttp 3.8.2` does not build on Python 3.12+. CI already installed it with `--no-deps`; `setup.sh` used a different, non-equivalent workaround.
- `polygon-api-client` needs `websockets>=10` (the conflict above).
- `XGBClassifier(use_label_encoder=False)`: parameter removed in XGBoost 2.x (warning only).
- `datetime.utcnow()` deprecated since 3.12 (logger, Discord notifier).
- `feeds.reuters.com` RSS is offline; `feedparser` returns an empty feed, silently.
- `SEARXNG_URL` hardcoded to `localhost:8080`; the macro breaker returns CLEAR anywhere without SearXNG.
- `get_risk_free_rate` calls Polygon `get_last_trade("I:TNX")`; indices have no trades, so it always fell back to 4% after a network call on every metric computation.
- `polygon_fetcher.get_latest_price` returned `.iloc[0]`, the oldest bar in the window (dead code, but wrong).

---

## 2. Phase 1: Architecture as actually implemented (before the fix)

### 2.1 Runtime topology

```
run.py (main thread: sleeps forever)
 |- APScheduler BackgroundScheduler  -> retrain_all()   [job added PAUSED: never fires; issue M-01]
 |- Thread "stable"  -> strategies.stable.run()
 |- Thread "risky1"  -> strategies.risky1.run()
 |- Thread "risky2"  -> strategies.risky2.run()   [only if FEATURE_FLAGS.risky2_enabled]
                              |
        all three use  get_api(strategy)  -> ONE Alpaca account (same key pair)
        all three write to   trades.db    -> ONE `trades` table, ONE `heartbeat` table
```

There was no execution, portfolio or order layer. Each strategy module contained its own copy of the kill switch, position lookup, sizing, order submission and logging.

### 2.2 The per-ticker lifecycle (stable and risky1, original code)

```
Alpaca account.portfolio_value  (account-wide, shared by all bots)
        |
Polygon get_latest_bar(ticker)  -> 200 DAILY bars (last bar = previous session close on free tier)
        |
build_features() + add_sentiment_to_df()  (one constant sentiment per ticker per day)
        |
current_price = last daily close        current_atr = last ATR in DOLLARS
        |
get_position_size(strategy, ACCOUNT equity, MAX_POSITION_SIZE * throttle, atr$)
        |     drawdown vs CAPITAL[strategy]; atr$ / ATR_BASELINE(fraction) -> always clamped 1.5x
        |
current_pos = Alpaca get_position(ticker).market_value   (ACCOUNT-wide, DOLLARS)
        |
[risky1] momentum_5 < 0 and current_pos > 0 -> close_position(ticker) (ACCOUNT-wide); log SELL qty=$
        |
stop loss: Alpaca avg_entry_price (account-wide) vs stale daily close
        |
regime gate (ADX / ATR / VIX / macro)  -> XGBoost predict on the last feature row
        |
BUY : qty = (max_pos$ - current_pos$) / stale_close; market order; log BUY qty=shares
SELL: close_position(ticker) (ACCOUNT-wide); log SELL qty=market_value$
        |
trades.db  (no order id, no fill price, no fill qty, no status)
```

Order status, fills, rejections and pending orders were never read back.

### 2.3 Risk layers (original)

| Layer | Where | Baseline | Scope | Effect |
|---|---|---|---|---|
| Kill switch | `check_kill_switch` in each strategy | `get_portfolio_risk_level` used `CAPITAL[strategy]` | Account equity | `api.close_all_positions()` (account-wide), then the thread exits |
| Warning | same | same | Account equity | halves `max_pos` for new buys only |
| Per-trade stop | `should_close_position` | Alpaca `avg_entry_price` vs stale close | Account position | `close_position(ticker)` |
| Equity-curve filter | `get_trading_state` | SQLite `pnl IS NOT NULL` rows | Per strategy by log rows | HALT skipped the whole cycle, including stop-loss checks |
| VIX / macro | `get_regime_for_strategy` | FRED VIX, SearXNG score | Global | Blocks new risky entries only |
| Risky2 | none of the above | | | No kill switch, no stop, no gate, no throttle |

### 2.4 Position ownership model (original)

Broker positions are account-wide; there was no local ledger; `CAPITAL` and `MAX_POSITION_SIZE` were notional and never enforced against the account; the symbol lists did not overlap at the time of the ZIP but had overlapped earlier (AMZN, see 4.3) and nothing prevented overlap.

### 2.5 Dead, duplicated and unfinished components

- Duplicated: `stable.py` and `risky1.py` shared about 200 identical lines.
- Dead: `get_latest_price` (wrong and unused), `is_crypto`, an unused `is_market_open` import, commented-out legacy blocks in `logger.py` and `regime_detector.py`, the live-URL path, dashboard v1 versus v2.
- Unfinished: dynamic screeners (WEEK9), confidence system (Week11), financial-health analyzer; `risky1` model trained on `["SPY"]` only ("temporary placeholder").
- Inconsistent: DECISION_LOG (2026-04-11) records "peak equity" as the drawdown baseline; the code used starting capital for the decision and peak only for the message.

---

## 3. Historical evidence and how it was analysed

Source: `2026-09-14T16-57_export.csv`, columns `timestamp, strategy, asset, action, price, quantity, pnl, reason` (the original 9-column `trades` table). Analysed with pandas; the scripts are reproduced in appendix A so the numbers can be regenerated.

### 3.1 Shape of the data

| strategy | BUY | SELL | KILL_SWITCH | SELL reasons |
|---|---|---|---|---|
| stable | 239 | 72 | 1 | 59 "ML signals SELL", 13 "Stop loss or portfolio risk triggered" |
| risky1 | 260 | 250 | 1 | 248 "Momentum turned negative", 2 "ML signals SELL" |
| risky2 | 2,753 | 2,700 | 0 | 2,700 "PPO agent chose SELL" |

Time span 2026-04-15 13:30 UTC to 2026-09-14 16:49 UTC. Realized PnL is present on 52 stable SELLs (sum +$78.56), 2 risky1 SELLs (+$2.53) and 0 risky2 SELLs.

### 3.2 Findings that are directly measurable in the export

| Finding | Measurement |
|---|---|
| SELL `quantity` is dollars | For every SELL, `quantity / (shares bought since the previous SELL) / logged price`: median 0.99 (risky1, n=249), 1.00 (risky2, n=2,700), 1.00 (stable, n=71) |
| Logged price is one value per day | Distinct prices per (strategy, asset, day): exactly 1 in all 408 groups |
| Risky1 trades once per loop cycle | Median gap between consecutive risky1 trades on the same symbol: 2.36 min; 10th to 75th percentile 2.355 to 2.53 min. Loop = 4 x 20 s + 60 s = 140 s |
| Momentum exits carry no PnL | 248 of 248 "Momentum turned negative" rows have `pnl` NULL; 5 of 13 stop-loss rows and 15 of 61 ML-SELL rows also NULL |
| Busiest days | risky1 AMD 2026-06-10: 103 trades; risky1 TSLA 2026-07-28: 90; risky2 436 to 463 trades per day 2026-06-19 to 06-25, up to 68 per coin per day |
| Risky2 hold time | Median 33.27 min (30-minute hold + one cycle); 111 of 2,700 sells inside 30 min (restarts lose the in-memory open time) |
| Dust top-ups | 143 of 239 stable BUY rows are under $5 (median stable BUY $1.86) |
| Phantom BUY rows | MSFT 2026-04-27 to 06-10: 13 BUY rows totalling 1.4477 sh (about $584) against a $200 per-symbol cap, then one SELL of $191.61; similar for IWM (9 buys, $628 vs $208 sold), SAP (14 buys, $384 vs $213), AAPL (4 buys, $566 vs $189) |
| Repeated "close" of the same position | SAP 2026-04-24: stop-loss SELL rows at 13:32:23 ($389), 13:39:46 ($213), 13:44:55 ($213), 13:48:17 ($213) |
| Cross-strategy liquidation | 2026-06-02 13:32:14 risky1 AMZN SELL $195.29 "Momentum turned negative"; risky1 has zero AMZN BUY rows in the whole history; stable had bought 0.7634 AMZN on 06-01 and re-bought 0.7655 at 13:32:55 |
| Kill switches | 2026-07-07 14:11:18.514 risky1 pnl=-0.949473; 14:11:43.111 stable pnl=-0.949497. Last trade before: 13:32:45 stable SAP BUY $200; 13:31:23 stable AAPL SELL $204.40 pnl +$12.49 |
| Recovery after the kill | risky2 resumed 07-08 09:17; stable 07-20 13:30; risky1 07-28 13:31 |

---

## 4. Investigations

### 4.1 Investigation 1: July 7 "-94.95%" kill switch on stable and risky1

**Code path.** Both `check_kill_switch` functions did:

```python
equity   = float(account._raw['portfolio_value'])          # ACCOUNT-wide
_peak_equity[strategy] = max(_peak_equity[strategy], equity)  # in memory, starts at CAPITAL
drawdown = (equity - peak) / peak                            # printed and logged
risk     = get_portfolio_risk_level(strategy, equity)        # DECIDES: (equity - CAPITAL) / CAPITAL
```

Two different drawdowns existed:

| Quantity | Formula | Used for |
|---|---|---|
| Logged | (account equity - in-memory peak) / peak | print, `pnl` column, Discord |
| Deciding | (account equity - CAPITAL[strategy]) / CAPITAL[strategy] | safe / warning / critical |

**What the export proves.**
- Both switches fired within 25 seconds on nearly identical values (-0.949473 and -0.949497). Their peaks were initialised differently (1000 and 200) so the values can only coincide if both peaks had been overwritten by the same observed account equity. The equity source was therefore account-wide and shared.
- The deciding condition needs account equity <= $850 for stable (-15% of 1000) and <= $140 for risky1 (-30% of 200). Both fired, so the reading was <= $140. With a logged -94.95%, the in-memory peak was about equity / 0.0505, i.e. at most about $2,770.
- Forty minutes earlier (13:31 to 13:32 UTC) stable sold $204 of AAPL for +$12.49 and bought $200 of SAP, normal activity for an account holding roughly $1,400 to $2,700 of assets. No trades occurred between 13:32:45 and 14:11:18. A real 95% loss in a long-only book of SPY/QQQ/AAPL/MSFT/DIA/IWM/SAP/AMZN plus small AMD/crypto positions cannot occur in 40 minutes.
- The trade record after the event is consistent with the account having been liquidated by `close_all_positions()` (called by both bots): risky2 restarted from empty on 07-08 (every coin BUY-first), stable resumed on 07-20, risky1 on 07-28. The staggered resume dates match the deciding formula: after a restart the peak resets to CAPITAL and the decision compares account equity with 850 (stable) and 140 (risky1), so each bot could only come back once the owner's restarts and the account level allowed it.

**What cannot be proven.** Why Alpaca returned a value <= $140 at 14:11 UTC on 2026-07-07 (a transient bad `portfolio_value`, a paper-account reset, or a mis-marked position) is not visible in the export or the code. The Alpaca account activity history for that day would settle it.

**Root cause (confirmed regardless of the trigger's origin):**
1. The kill switch protected the wrong quantity: account equity against one strategy's notional capital. With three strategies in one account, stable's "15%" switch actually meant "account down to $850" and risky1's "30%" meant "account down to $140". The protection was far weaker than configured on the way down, and when it finally fired it liquidated every strategy's positions, twice.
2. A single unvalidated reading was enough to fire it. There was no comparison with the previous reading, no NaN/zero check, no confirmation cycle.
3. The logged number was not the deciding number, so the audit trail could not explain the decision.
4. The peak lived in process memory, so restarts erased the high-water mark and made "drawdown from peak" meaningless as a historical record.

**Disproven:** genuine portfolio destruction (timeline in the export), position-sizing error as the trigger (no trades between the last normal trade and the switch), incorrect peak tracking as the trigger (the peak only affects the logged number, not the decision).

### 4.2 Investigation 2: TSLA BUY 0.1446 versus SELL about 51.2 (Aug 27)

| Step | Value |
|---|---|
| `MAX_POSITION_SIZE["risky1"]` | 200 x 0.25 = **$50.00** |
| BUY qty = `round((max_pos - current_pos) / current_price, 4)` | round(50 / 345.82, 4) = **0.1446** |
| SELL "quantity" logged = `current_pos` = `get_position(ticker).market_value` | **dollars** |
| Implied live price = 51.225 / 0.1446 | $354.25 (2.4% above the stale 345.82) |

The export shows 22 such round trips on 2026-08-27 alone (13:31 to 18:00 UTC), every BUY exactly 0.1446, every SELL between $50.2 and $51.24, every price 345.82, every SELL reason "Momentum turned negative", every gap 2 min 21 s to 2 min 24 s. Across all 3,020 SELL rows in the history the same dollars-versus-shares relationship holds (section 3.2).

**Conclusion (confirmed):** no 51-share sale happened. The order path was unit-consistent (dollars converted to shares once, on BUY; SELL used `close_position`, which sold the 0.1446 shares the account held). Only the log mixed units. Any analysis reading `quantity` as shares sees a 350x buy/sell ratio that did not exist.

**Disproven:** percentage-as-shares, dollars-as-shares in the order path, target-weight confusion, accidental shorting.

### 4.3 Investigation 3: cross-strategy position ownership

| Question | Answer (original code) |
|---|---|
| How are positions represented? | Only at the broker, account-wide |
| Strategy ownership? | None: no ledger, no tag, no per-strategy account |
| Can two strategies trade the same symbol? | Yes. AMZN was in both lists in early June 2026 |
| Can one strategy liquidate another's inventory? | Yes: `close_position(ticker)` on a shared symbol; `close_all_positions()` from any kill switch; stop-loss and momentum exits on the account's `avg_entry_price` and `market_value` |
| Reconciliation? | None |

**Confirmed incident, 2026-06-02 13:32:14 UTC.** risky1 logged `AMZN SELL 261.26 195.29 "Momentum turned negative"`. risky1 has no AMZN BUY anywhere in the 6,276 rows. Stable had bought 0.7634 AMZN the day before (0.739 + 0.0206 + 0.0038 shares at 270.64). The risky1 momentum exit saw `current_pos = market_value > 0` on the account, called `close_position("AMZN")`, sold stable's shares, and logged $195.29. Stable, seeing no position 41 seconds later, bought 0.7655 AMZN again at 13:32:55. This is exactly the hypothesised failure, executed in production.

**Confirmed incident, 2026-07-07.** Both kill switches called `close_all_positions()`, which includes risky2's crypto (risky2 has no kill switch of its own and restarted from empty the next day).

**Verdict:** the architecture was unsafe for multi-strategy operation in one account. Fixed in C-02.

### 4.4 Investigation 4: rapid BUY/SELL churn

**Timing.** Risky1's loop is `for 4 tickers: trade; sleep 20` then `sleep 60`: 140 s plus latency. The export's median gap between consecutive risky1 trades is 2.36 min. Each cycle produced one trade, alternating BUY and SELL.

**Mechanism (confirmed).** Within a trading day every input is constant: the last daily bar does not change, so `momentum_5`, the features, the regime and the prediction are identical every cycle. Entry and exit were evaluated independently:

```
cycle N   : no position -> momentum exit skipped (needs position) -> regime ok -> predict == 1 -> BUY
cycle N+1 : position    -> momentum_5 < 0 -> SELL "Momentum turned negative"   [ML not consulted]
cycle N+2 : no position -> predict == 1 -> BUY ...
```

When `momentum_5 < 0` and the model predicts 1 on the same bar, the system is a two-state oscillator with a one-cycle period. 248 of risky1's 250 SELLs are momentum exits, so this mechanism accounts for essentially all risky1 selling. No hysteresis, cooldown, minimum hold, pending-order check or last-signal memory existed. Each round trip paid the spread twice.

**Secondary contributors.**
- Momentum-exit SELLs recorded no `pnl` (computed but never passed to `log_trade`), so the equity-curve filter, whose job is to throttle a cold streak, saw none of these 248 exits.
- Stable's "top-up" logic bought the difference between `max_pos` and market value whenever it was at least $1: 143 of 239 stable BUYs are under $5.
- Risky2 sampled its policy (`deterministic=False`), so each cycle's HOLD/BUY/SELL was a random draw; 2,753 buys and 2,700 sells in five months, median hold 33 minutes (the 30-minute guard plus a cycle).

**Disproven:** polling frequency alone (repeating the same decision cannot alternate it), live-price noise (there were no live prices), threshold jitter.

**Root cause:** no state machine. Entry and exit were unrelated predicates on the same data and could both be true at once.

### 4.5 Investigation 5: identical prices

`current_price = float(df['close'].iloc[-1])` from `get_latest_bar(ticker, timespan="day")`. On Polygon's free tier the latest completed daily bar is the previous session's close, so the value is constant all day: exactly one distinct price per (strategy, asset, day) in all 408 groups. It is a signal price, not an order price and not a fill. Market orders filled at live prices (about $354 in the TSLA example) that were never captured. Consequences: the per-trade stop compared a live entry price with a stale close, so intraday moves could not trigger it until the next day; realized PnL was computed from the stale price.

### 4.6 Investigation 6: Risky2 PPO path

| Stage | Finding |
|---|---|
| Action space | `Discrete(3)`: 0 HOLD, 1 BUY, 2 SELL. Not continuous, not a weight, not a delta ("normalized [-1,1]" hypothesis disproven) |
| Observation at inference | `TradingEnvironment(df).reset()` set `current_step = 0`, so the observation was `df.iloc[0]`: the **oldest** of 200 daily bars, with position and unrealized-PnL slots always 0 |
| Sampling | `deterministic=False`: random draws |
| Sizing | `qty = round(max_pos / price, 4)` with `max_pos = $50`: unit-consistent; SELL logged dollars (same log bug) |
| Risk stack | None |
| Endpoint | `PAPER_MODE["risky2"] = False`: live URL |
| Train/inference consistency | Same feature functions; raw price levels (per-ticker models mitigate). The "missed opportunity" reward uses the next bar's price (hindsight shaping in training; not an observation leak) |
| Hold-time state | In-memory; lost on restart (111 of 2,700 sells inside the 30-minute hold) |
| Model path | `MODEL_DIR = "models"` relative to the working directory |

---

## 5. Additional findings from the general audit

Listed here; each is a numbered issue record in section 7.

Risk integrity: ATR unit mismatch (H-01); HALT skipped exits (H-05); starting-capital baseline contradicting the decision log (C-03); no equity validation (C-03); kill switch permanent-by-accident after restart (C-03).
Execution and accounting: BUY logged before fill (H-06); no pending-order tracking (H-06); SELL rows in dollars (H-03); PnL from stale price, momentum exits without PnL (H-03); KILL_SWITCH rows storing a fraction in the dollar `pnl` column (M-04).
ML validity: risky1 trained on SPY only (M-02); cross-ticker leakage in the split and a constant sentiment column (M-03); in-sample backtests in RESULTS.md (M-03); last-row label of 0 (M-03); retraining paused and never reloaded (M-01); import-time model load crashed startup when no model existed (M-01).
State and observability: heartbeat stale threshold (60 s) shorter than the cycle (M-05); metrics computed on dollar rows as returns (M-04); dashboard risk panel reproducing the account-versus-capital error (M-04).
Security and configuration: live endpoint for risky2 (C-01); no credentials in the repository; infrastructure identifiers in docs (L-02).

---

## 6. Severity ranking and remediation plan (as presented at the checkpoint)

| # | Issue | Severity | Status |
|---|---|---|---|
| C-01 | Risky2 configured for the live endpoint; live path unguarded | CRITICAL | Fixed |
| C-02 | Kill switch liquidates the shared account; no strategy ownership | CRITICAL | Fixed |
| C-03 | Kill switch decides on account equity vs strategy CAPITAL; logged number differs; no validation; in-memory peak | CRITICAL | Fixed |
| H-01 | ATR in dollars divided by a fractional baseline | HIGH | Fixed |
| H-02 | Entry/exit oscillation and dust top-ups | HIGH | Fixed |
| H-03 | SELL logs dollars; no fill/order data; missing PnL | HIGH | Fixed |
| H-04 | Risky2 observes the oldest bar, samples actions, has no risk stack | HIGH | Fixed |
| H-05 | HALT blocks stop-loss evaluation | HIGH | Fixed |
| H-06 | Phantom BUYs on rejected orders; no pending-order guard; repeated closes | HIGH | Fixed |
| M-01 | Retrain job paused; models never reloaded; import-time load crash | MEDIUM | Fixed |
| M-02 | Risky1 model trained on SPY only | MEDIUM | Fixed (code); models must be retrained |
| M-03 | Train/test leakage; constant sentiment; in-sample backtests; last-row label | MEDIUM | Fixed (code); documented in RESULTS.md |
| M-04 | Metrics on dollar rows; KILL_SWITCH fraction in `pnl`; dashboard risk panel | MEDIUM | Fixed |
| M-05 | Heartbeat stale threshold | MEDIUM | Fixed |
| M-06 | Dependency conflicts on modern Python | MEDIUM | Fixed |
| L-01 | Duplicated loops, dead code, docs drift | LOW | Fixed |
| L-02 | Infrastructure identifiers in docs | LOW | README cleaned; ORACLE_SETUP.md left for the owner |

Decisions that change trading behaviour were made explicitly with alternatives recorded: C-02 (ledger versus separate accounts), C-03 (two-reading confirmation), H-02 (which anti-churn mechanism), M-03 (training changes).

---

## 7. Issue records

Each record follows: Problem, Severity, Evidence, Root cause, Impact, Investigation, Fix, Why this fix, Validation, Remaining risk, Interview takeaway.

### C-01 Live endpoint reachable without opt-in

**Problem.** `core/config.py` had `PAPER_MODE["risky2"] = False`, and `get_api` mapped `False` straight to `https://api.alpaca.markets`. Nothing else guarded the live path.
**Severity.** CRITICAL (real-money path in a paper-only system).
**Evidence.** Source inspection of `core/config.py` and `paper_trading/alpaca_paper.py`. The export shows risky2 trading on the paper account through 2026-09, so the flag was flipped after risky2 was last enabled, or paper keys were rejected by the live endpoint; either way the configuration in the ZIP would route risky2 live if live keys were ever placed in `.env`.
**Root cause.** A per-bot boolean with an inverted, easily-misread comment ("flip to false to make risky2 live") and no second factor.
**Impact.** Real orders from an untested RL bot.
**Investigation.** Read the config, traced `get_api`, checked the dashboard and `run.py` for any endpoint reporting (none).
**Fix.** All strategies `PAPER_MODE = True`. `resolve_base_url` raises `LiveTradingBlocked` unless `SINGHQUANT_ALLOW_LIVE_TRADING=I_UNDERSTAND_REAL_MONEY` is set in the environment. `run.py` prints each strategy's endpoint in a preflight and refuses to start if misconfigured. Optional per-strategy paper keys (`ALPACA_API_KEY_<STRATEGY>`).
**Why this fix.** Two independent, explicit conditions are required for live money; the safe state is the default and is visible at startup.
**Validation.** `tests/test_engine.py::test_live_endpoint_is_blocked_without_opt_in`; `run.py` booted with dummy keys shows all three strategies on the paper URL (section 8.3).
**Remaining risk.** Someone editing the env var deliberately.
**Interview takeaway.** Safety switches need a default-safe state and an explicit, hard-to-misread opt-in, not a boolean with a comment.

### C-02 No strategy ownership of positions

**Problem.** All strategies read, sized, exited and kill-switched against account-wide broker state.
**Severity.** CRITICAL.
**Evidence.** 2026-06-02 risky1 AMZN SELL of $195.29 with zero risky1 AMZN buys ever; 2026-07-07 two `close_all_positions()` calls 25 seconds apart; source inspection of every `api.get_position` / `close_position` / `close_all_positions` call.
**Root cause.** Alpaca positions carry no owner. The design assumed one bot per symbol list and never persisted what each bot had bought; `CAPITAL` per strategy was a label, not an enforced budget.
**Impact.** One strategy can liquidate another's inventory; strategy P&L and drawdown are unmeasurable; a kill switch in one strategy destroys the others.
**Investigation.** Traced `get_current_position` and both SELL branches in stable/risky1/risky2; searched the export for symbols traded by more than one strategy (AMZN) and for SELLs without prior BUYs by the same strategy.
**Fix.** `core/portfolio.py`: per-strategy `positions` (qty, average entry, opened_at), `strategy_state` (cash budget, peak equity, kill-switch state, critical streak), `signal_state`, `orders` and `equity_snapshots` tables in the same SQLite file, added by an additive migration. `core/execution.py` sells only the ledger quantity, never calls `close_position`/`close_all_positions`, and reconciles drift when the broker holds less than the ledger (write-off with a `RECONCILE` row). `close_all_strategy_positions` replaces the account-wide emergency close. Startup reports unowned broker positions and adopts them only with `SINGHQUANT_ADOPT_BROKER_POSITIONS=1`.
**Why this fix.** It gives every quantity an owner without requiring new broker accounts, works with the existing database, and makes the invariant "sell only what you bought" enforceable in one place. Alternative considered: three Alpaca paper accounts (strongest isolation; supported via optional per-strategy keys but requires the owner to create accounts). Alternative rejected: tagging orders with `client_order_id` only (fills and manual trades would still be unattributed).
**Validation.** `test_execution.py`: `test_one_strategy_cannot_liquidate_anothers_position` (the June 2 scenario), `test_close_all_only_touches_own_positions`, `test_strategy_cannot_sell_more_than_it_owns`, `test_ledger_drift_written_off_when_broker_has_nothing`, `test_strategy_equity_is_independent_of_account_equity`; `test_engine.py::test_kill_switch_needs_two_confirmations_then_closes_only_own_positions`; `test_migration_and_startup.py::test_existing_broker_positions_are_reported_not_adopted_by_default`.
**Remaining risk.** Manual trades in the Alpaca UI are invisible until the next reconcile; the ledger trusts fills reported by the broker.
**Interview takeaway.** When a shared resource has no ownership concept, the application must keep its own ledger; "who owns this" cannot be derived after the fact.

### C-03 Kill switch on the wrong quantity, wrong baseline, unvalidated reading

**Problem.** Decision drawdown used `CAPITAL[strategy]` as the baseline against account equity; the logged drawdown used an in-memory peak; no validation of the reading; the thread exited on fire, and restarts re-fired.
**Severity.** CRITICAL.
**Evidence.** July 7 rows (section 4.1); DECISION_LOG 2026-04-11 stating peak equity was chosen; `get_portfolio_risk_level` using `start = CAPITAL[strategy]`.
**Root cause.** Two implementations of "drawdown" grew side by side: the peak version was added for the message, the capital version was left in the deciding function. Equity was the account's because there was no strategy equity to use (C-02).
**Impact.** Weak protection on the way down, account-wide liquidation when it fired, a log that could not explain the decision, no recovery path.
**Investigation.** Compared the printed formula with the deciding function; computed which account level each strategy's threshold implied ($850 and $140); checked the export for what preceded the switch (a normal $204 sale 40 minutes earlier) and how the bots came back (staggered, matching the capital baseline).
**Fix.** `strategies/common.py::evaluate_risk`: equity = strategy cash budget + own positions at the cycle's mark prices; missing marks value the position at entry (never zero); `validate_equity` rejects NaN/inf/non-positive readings and moves larger than 50% in one cycle as "suspect" (no entries, no kill) unless the next reading agrees; peak persisted in `strategy_state`; a single `risk_level_from_drawdown` decides and the same drawdown is written to `equity_snapshots`; the kill switch needs `KILL_SWITCH_CONFIRMATIONS = 2` consecutive critical readings; it closes only the strategy's positions and sets a persisted `halted` flag that survives restarts until `tools/reset_kill_switch.py` is run.
**Why this fix.** The deciding and logged numbers are now the same value from the same function; the baseline is the intended peak; bad data cannot fire the switch on its own; the switch is scoped to the strategy that tripped it. Alternative considered: keep single-reading firing (faster reaction) - rejected because the one documented firing was almost certainly a bad reading, and a two-to-four-minute delay is acceptable for a daily-bar paper system.
**Validation.** `test_engine.py`: `test_kill_switch_ignores_account_equity`, `test_kill_switch_needs_two_confirmations_then_closes_only_own_positions`, `test_logged_drawdown_equals_deciding_drawdown`, `test_implausible_equity_jump_is_suspect_and_does_not_kill`, `test_nan_or_inf_equity_is_rejected`, `test_missing_marks_do_not_create_drawdown`; `test_atr.py`/`test_risk.py` still pass on the backward-compatible `get_portfolio_risk_level`.
**Remaining risk.** The Alpaca-side cause of the July 7 reading is unknown; the confirmation delay is one cycle.
**Interview takeaway.** A safety control must log the exact value it decided on, and must never act on a single unvalidated input.

### H-01 ATR units

**Problem.** `get_atr_adjusted_thresholds` divided ATR in price units by `ATR_BASELINE` in fraction-of-price units.
**Severity.** HIGH (risk integrity).
**Evidence.** `build_features` on synthetic SPY/TSLA/BTC-like series gives ATR of $14.9 / $10.3 / $2,124, i.e. scale factors 995 / 685 / 141,569 before clamping; the existing tests passed fractional values (0.030) and therefore never exercised the real magnitudes.
**Root cause.** The config comment documented fractions; the caller passed the raw feature. The unit was implicit.
**Impact.** Every warning and kill threshold was 1.5x wider than configured (stable kill at -22.5%, risky1 at -45%), for every asset, always.
**Fix.** `atr_as_fraction(atr, price)` in `metrics/risk_manager.py`; the engine converts before any threshold or sizing call; config comment states the unit.
**Why this fix.** Normalising at the boundary between the feature layer and the risk layer is the only place both units are known.
**Validation.** `test_engine.py::test_atr_dollars_are_converted_to_fraction_before_scaling` (asserts the old input yields -0.225 and the converted input yields the intended tighter/wider values); original `test_atr.py` unchanged and passing.
**Remaining risk.** None known.
**Interview takeaway.** Unit mismatches between modules are invisible to tests that only exercise one module with hand-picked inputs.

### H-02 Oscillation and dust top-ups

**Problem.** Independent entry and exit predicates; no per-bar state; sub-$5 top-ups every cycle.
**Severity.** HIGH.
**Evidence.** Section 4.4; 143 of 239 stable BUYs under $5.
**Root cause.** No state machine; the exit rule was added "before the ML check" (DECISION_LOG 2026-04-03) without checking the entry rule against it.
**Impact.** Two market orders per cycle paying the spread; equity-curve filter blind because the exits logged no PnL.
**Fix.** In `strategies/common.py`: `can_enter` refuses a BUY while the strategy's own exit condition is already true on this bar and refuses re-entry on the bar just exited (`ONE_ACTION_PER_BAR`); `can_discretionary_exit` enforces `MIN_HOLD_SECONDS` (persisted `opened_at`) and no discretionary exit on the entry bar; emergency exits bypass both; top-ups only when the position is below 90% of target (`MIN_TOPUP_FRACTION`).
**Why this fix.** It removes the contradiction at its source (buying into a bar that would immediately sell) instead of adding a timer that would delay but not remove the oscillation. Alternatives considered: momentum hysteresis (changes the strategy's signal; category B), a fixed 15-minute cooldown (would still trade every 15 minutes on the same bar). Both are recorded as options, not implemented.
**Validation.** `test_engine.py`: `test_no_entry_when_momentum_exit_condition_already_true` (4 cycles, 0 orders), `test_oscillation_sequence_collapses_to_one_round_trip` (BUY, deferred, SELL on the new bar, no re-entry: exactly one round trip), `test_repeated_identical_buy_signals_do_not_multiply_exposure`, `test_no_dust_top_ups_when_price_drifts`, `test_stop_loss_bypasses_min_hold_but_discretionary_sell_waits`.
**Remaining risk.** Fewer momentum exits means positions are held through a negative-momentum bar they entered on; this is intended.
**Interview takeaway.** Trading logic needs an explicit state machine; two independent rules that can both be true produce oscillation, and no threshold tuning fixes that.

### H-03 Log units and missing execution data

**Problem.** SELL rows logged dollars in `quantity`; no order id, status, fill quantity or fill price; momentum exits logged no PnL; PnL computed from the stale price.
**Severity.** HIGH (data integrity).
**Evidence.** Section 3.2 ratio table; 248 momentum exits with NULL PnL.
**Root cause.** `current_pos` (market value) was reused as the "quantity" argument; `realized_pnl` was computed but omitted from one of the three `log_trade` calls.
**Fix.** `core/logger.py` adds nullable columns (`signal_price, order_id, status, filled_qty, filled_avg_price, strategy_position_before, account_position_before, risk_state, model_output`) by additive migration; `core/execution.py` writes exactly one row per order with the broker's fill data; `quantity` is always shares; realized PnL comes from the ledger's average entry and the fill price.
**Why this fix.** Only the execution layer knows the fill; logging there removes the three divergent call sites.
**Validation.** `test_execution.py::test_buy_fill_updates_ledger_and_logs_shares`, `test_sell_logs_shares_not_dollars`, `test_rejected_order_creates_no_position`; `test_migration_and_startup.py::test_legacy_database_is_migrated_in_place_and_history_preserved` (legacy rows preserved and recognisable by `status IS NULL`).
**Remaining risk.** Historical SELL rows remain in dollars; documented in README.
**Interview takeaway.** Log the outcome, not the intention; and never reuse a variable across units.

### H-04 Risky2 inference and risk

**Problem.** Observation from row 0; stochastic sampling; no risk controls; volatile hold-time state; CWD-relative model path.
**Severity.** HIGH.
**Evidence.** `env.reset()` sets `current_step = 0`; `_get_observation` reads `self.df.iloc[self.current_step]`; `deterministic=False`; risky2's 2,753/2,700 trade count and 111 sub-30-minute holds.
**Root cause.** The training environment was reused for inference without a "latest row" entry point.
**Fix.** `models/rl_environment.py::build_live_observation` (latest row, real position value and entry price, same layout as training via the shared `build_observation`); `strategies/risky2.py` uses it with `deterministic=True`; risky2 now runs in the shared engine and therefore has the stop loss (12%), strategy-level kill switch, throttle and a persisted 30-minute hold; `PAPER_MODE` true; `MODEL_DIR` absolute.
**Why this fix.** The agent finally sees the state it was trained to see (current features and its own position) and the strategy is subject to the same controls as the others.
**Validation.** `test_engine.py::test_ppo_live_observation_uses_latest_bar_and_real_position` (asserts the last row's feature value and the position slots; the env still starts at row 0 for training), `test_stop_loss_bypasses_min_hold_but_discretionary_sell_waits`, `test_equity_strategies_pause_when_market_closed_but_crypto_trades`; `test_r1.py` unchanged and passing.
**Remaining risk.** No PPO checkpoints exist in the ZIP, so a real model was not loaded; the reward design and raw-price features remain as they were (category C).
**Interview takeaway.** Reusing a simulator object for live inference is a classic train/serve skew; the observation builder must be shared and the inference path must inject live state.

### H-05 HALT blocked exits

**Problem.** `if state == "HALT": continue` skipped the ticker loop entirely, including stop-loss and momentum exits.
**Severity.** HIGH.
**Evidence.** Source inspection of both run loops.
**Root cause.** HALT was implemented as "skip the cycle" rather than "no new risk".
**Fix.** The engine always evaluates exits; HALT (and critical/suspect risk levels) only set `allow_entries = False`.
**Validation.** `test_engine.py::test_halt_blocks_entries_but_still_runs_exits` (HALT created from real closed-trade rows; BUY ignored, stop loss still fires).
**Interview takeaway.** "Stop trading" must be split into "stop adding risk" and "keep reducing risk".

### H-06 Phantom BUYs, repeated closes, no pending-order guard

**Problem.** `log_trade("BUY")` ran right after `submit_order` regardless of outcome; `close_position` was logged every cycle until the position disappeared; nothing tracked open orders.
**Severity.** HIGH.
**Evidence.** MSFT 13 BUY rows ($584) vs one $191 SELL; SAP stop-loss "close" logged four times in 16 minutes on 2026-04-24; risky2 on 2026-06-03 logging four to five $50 BUYs per coin but selling $47 to $50.
**Root cause.** Fire-and-forget order submission with no read-back.
**Fix.** `core/execution.py::submit_and_track`: records the order, submits, polls `get_order` until terminal or timeout, applies only the filled quantity to the ledger, writes the status; `has_open_order` (backed by `reconcile_open_orders`, which also applies fills that completed while the process was down) refuses a new order for a symbol with an open one; `MIN_ORDER_NOTIONAL` and finite-quantity checks.
**Why this fix.** Reconciling against the broker's own order record is the only source of truth for what happened.
**Validation.** `test_execution.py::test_partial_fill_applies_only_filled_quantity_then_reconciles`, `test_pending_order_blocks_duplicate_submission`, `test_restart_does_not_duplicate_and_late_fill_is_applied_once`, `test_rejected_order_creates_no_position`, `test_invalid_quantities_are_refused`; `test_engine.py::test_partial_fill_is_topped_up_not_duplicated`.
**Remaining risk.** Order status polling uses the deprecated SDK; a broker outage leaves orders "open" locally until reconciled.
**Interview takeaway.** Any side effect against an external system needs a read-back and an idempotency key.

### M-01 Retraining never ran; models never reloaded; import crashed without models

**Evidence.** `add_job(..., next_run_time=None)`: verified with APScheduler 3.11 that the job's `next_run_time` is `None` (paused). `model = load_model(...)` at module import raised `FileNotFoundError` when no `.pkl` existed, which prevented `run.py` from starting at all.
**Fix.** `first_run_time()` schedules the first run one interval after boot; `ModelHandle` loads lazily and reloads when the file's mtime changes; strategies skip entries (and still manage exits) when no model exists.
**Validation.** `test_engine.py::test_retrain_job_is_actually_scheduled`, `test_missing_model_skips_entries_without_crashing`; `run.py` boot test in 8.3.

### M-02 Risky1 model trained on SPY only

**Fix.** `train_and_save("risky1")` uses `RISKY1_ASSETS`. Engineering fix (the code called it a placeholder), not tuning. Models must be retrained by the owner.

### M-03 Training validity

**Evidence.** `train_test_split(shuffle=False)` over rows stacked ticker-by-ticker made the test set "the last ticker(s)", whose dates overlap the training tickers' dates; `add_sentiment_to_df` stamped today's score on two years of rows; `create_labels` kept a last row labelled 0; RESULTS.md backtests ran over the training window.
**Fix.** Per-ticker chronological split before stacking; sentiment 0.0 during training (feature layout unchanged so live inference still matches); last row dropped; explicit `FEATURE_COLUMNS` with `features_for_model` validating names, finiteness and model width; `run_backtest.py` evaluates the hold-out tail by default (`--full` for the old in-sample numbers); RESULTS.md annotated.
**Why this fix.** These are correctness fixes to the evaluation, not performance tuning; no parameters were changed against historical results.
**Validation.** `test_migration_and_startup.py::test_create_labels_and_time_split`, `test_engine.py::test_model_feature_width_mismatch_is_detected`, `test_nan_feature_row_never_reaches_the_broker`.
**Remaining risk.** Raw price-level features remain (category C: replacing them with returns needs a proper backtest).

### M-04 Metrics and dashboard accounting

**Fix.** `dashboard/compare.py` and the v2 leaderboard compute per-trade returns as `pnl / CAPITAL`; equity-curve filter ignores KILL_SWITCH rows; KILL_SWITCH rows no longer store a fraction in `pnl`; the v2 risk panel shows each strategy's ledger equity and drawdown from its persisted peak plus a "who owns what" table; docker-compose runs v2.
**Validation.** `test_engine.py::test_equity_curve_filter_ignores_kill_switch_fraction_rows`; dashboards byte-compile; not exercised in a browser (no credentials).

### M-05 Heartbeat threshold

**Fix.** `HEARTBEAT_STALE_SECONDS = 600`; heartbeat written after every ticker.

### M-06 Dependencies

**Fix.** `requirements.txt` with compatible ranges verified on 3.14; `requirements-py311-pinned.txt` with the original pins for Docker/3.11; `setup.sh` and CI use one recipe (alpaca `--no-deps` plus loose transitive deps); `plotly<7` for vectorbt; CI matrix 3.11 (pinned) and 3.12 (ranges); lazy Polygon clients so importing modules needs no key.

### L-01 Structure and dead code

**Fix.** One engine (`strategies/common.py`); strategy modules are declarative contexts; dead `get_latest_price` corrected; `utcnow` replaced; `tools/reset_kill_switch.py`; README rewritten with historical context preserved; DECISION_LOG entry added.

### L-02 Identifiers in documentation

**Finding.** No API keys, tokens or `.env` in the repository. `ORACLE_SETUP.md` contains a personal email, a Tailscale IP, OCI tenancy/subnet/image OCIDs and an SSH username; the README contained two public VM IPs (removed). OCIDs alone cannot authenticate, but they narrow an attacker's search. **Recommendation:** move `ORACLE_SETUP.md` out of the public repository or redact the identifiers; rotate the Alpaca and Polygon keys as a precaution given the age of the deployment (no evidence of exposure was found).

---

## 8. Validation summary

### 8.1 Test suite

```
python -m pytest tests/ -q --deselect tests/test_regime.py::TestVIXSignal::test_get_vix_returns_float_or_none
108 passed, 1 deselected
```
Environment: Python 3.14.7, pandas 3.0.5, numpy 2.5.3, xgboost 3.4.1, scikit-learn 1.9.1, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14 CPU, vectorbt 1.1.0, plotly 6.9.0, APScheduler 3.11.3, alpaca-trade-api 3.0.0, polygon-api-client 1.16.3. The deselected test fetches VIX from FRED over the network and is excluded from CI for determinism.

Original tests: 68 (all pass unchanged). New tests: 41 in `tests/test_execution.py` (13), `tests/test_engine.py` (23) and `tests/test_migration_and_startup.py` (5), driven by `tests/fakes.py::FakeAlpaca` (account-wide positions, configurable fill/partial/reject/pending behaviour, market clock).

Mapping to the required regression invariants:

| # | Invariant | Test |
|---|---|---|
| 1 | Cannot sell more than owned | `test_strategy_cannot_sell_more_than_it_owns` |
| 2 | One strategy cannot liquidate another's position | `test_one_strategy_cannot_liquidate_anothers_position`, `test_close_all_only_touches_own_positions`, `test_kill_switch_needs_two_confirmations_then_closes_only_own_positions` |
| 3 | Sizing units consistent | `test_buy_fill_updates_ledger_and_logs_shares`, `test_sell_logs_shares_not_dollars`, `test_oscillation_sequence_collapses_to_one_round_trip` (0.1446 reproduced) |
| 4 | Repeated identical signals no duplicate exposure | `test_repeated_identical_buy_signals_do_not_multiply_exposure`, `test_no_dust_top_ups_when_price_drifts` |
| 5 | Pending orders prevent duplicates | `test_pending_order_blocks_duplicate_submission`, `test_partial_fill_is_topped_up_not_duplicated` |
| 6 | Emergency exits bypass cooldowns | `test_stop_loss_bypasses_min_hold_but_discretionary_sell_waits`, `test_halt_blocks_entries_but_still_runs_exits` |
| 7 | Drawdown uses valid portfolio equity | `test_kill_switch_ignores_account_equity`, `test_logged_drawdown_equals_deciding_drawdown` |
| 8 | Missing/stale data cannot fake a drawdown | `test_missing_marks_do_not_create_drawdown`, `test_implausible_equity_jump_is_suspect_and_does_not_kill`, `test_nan_or_inf_equity_is_rejected` |
| 9 | Strategy vs account accounting not substituted | `test_strategy_equity_is_independent_of_account_equity` |
| 10 | Restart does not duplicate orders | `test_restart_does_not_duplicate_and_late_fill_is_applied_once` |
| 11 | Partial fills reconcile | `test_partial_fill_applies_only_filled_quantity_then_reconciles` |
| 12 | Rejected orders create no position | `test_rejected_order_creates_no_position` |
| 13 | ML inference receives expected features | `test_model_feature_width_mismatch_is_detected` |
| 14 | Crypto/equity timing separate | `test_equity_strategies_pause_when_market_closed_but_crypto_trades` |
| 15 | NaN/inf fail safely | `test_nan_feature_row_never_reaches_the_broker`, `test_invalid_quantities_are_refused`, `test_nan_or_inf_equity_is_rejected` |

### 8.2 Static checks

All 40 Python files byte-compile (`py_compile`). Every module imports with dummy credentials and no model files.

### 8.3 Boot test

`python run.py` with dummy keys, no models and a temporary database: preflight prints all three strategies on the paper URL, the scheduler starts, both threads start, Alpaca returns 401 for the dummy keys, each thread logs the error and retries in 60 s. The process no longer crashes at import when models are absent.

### 8.4 Not validated (stated plainly)

- No order was sent to Alpaca (no credentials); execution is validated against `FakeAlpaca`, whose surface mirrors the parts of `alpaca_trade_api.REST` the code uses.
- No XGBoost or PPO model was trained or loaded (no data key, no checkpoints); model paths are exercised with stubs.
- The Streamlit dashboards were not opened in a browser.
- The Docker image was not built (no Docker on the audit machine).
- The Alpaca-side cause of the July 7 reading remains unknown.

### 8.5 Files changed

Added: `core/execution.py`, `core/portfolio.py`, `strategies/common.py`, `tools/reset_kill_switch.py`, `tests/conftest.py`, `tests/fakes.py`, `tests/test_execution.py`, `tests/test_engine.py`, `tests/test_migration_and_startup.py`, `requirements-py311-pinned.txt`, `ENGINEERING_AUDIT.md`.
Modified: `core/config.py`, `core/logger.py`, `strategies/stable.py`, `strategies/risky1.py`, `strategies/risky2.py`, `run.py`, `paper_trading/alpaca_paper.py`, `metrics/risk_manager.py`, `metrics/equity_curve_filter.py`, `metrics/risk.py`, `models/train.py`, `models/retrain.py`, `models/rl_environment.py`, `models/rl_train.py`, `data/polygon_fetcher.py`, `data/sentiment_fetcher.py`, `data/discord_notifier.py`, `backtesting/engine.py`, `backtesting/run_backtest.py`, `dashboard/compare.py`, `dashboard/streamlit_app_v2.py`, `docker-compose.yml`, `Dockerfile`, `setup.sh`, `requirements.txt`, `.github/workflows/test.yml`, `.gitignore`, `README.md`, `RESULTS.md`, `DECISION_LOG.md`.
Unchanged: `core/features.py`, `models/regime_detector.py`, `data/macro_fetcher.py`, `dashboard/streamlit_app.py`, all six original test files, the WEEK/planning documents, `ORACLE_SETUP.md`.

---

## 9. Remaining risks, limitations and recommendations

**Unverified**
- Cause of the July 7 equity reading (needs Alpaca account history).
- Behaviour of `alpaca-trade-api 3.0.0` order polling against the current Alpaca API (the SDK is deprecated; `alpaca-py` is the supported successor and a migration is recommended).
- Live database migration on the Oracle host: the migration is additive and tested on a synthetic legacy schema, but the real file should be backed up first.

**Known limitations kept on purpose**
- Daily-bar signal price on Polygon's free tier: intraday stops still wait for the next bar. A quote endpoint would fix this and is a small, separate change.
- Raw price-level features and the risky2 reward design are unchanged (category C: require backtesting before changing).
- Momentum hysteresis was not added (category B: changes the strategy's signal).

**Recommended next steps for the owner**
1. Stop the deployed old code (it was churning TSLA during the audit), back up `trades.db`, deploy this version, start once with `SINGHQUANT_ADOPT_BROKER_POSITIONS=1` if the broker still holds positions.
2. Retrain models with the corrected pipeline; run `backtesting/run_backtest.py` (hold-out) and compare with the in-sample numbers in RESULTS.md to see the honest gap.
3. Rotate Alpaca/Polygon keys; redact `ORACLE_SETUP.md`.
4. Consider one paper account per strategy (`ALPACA_API_KEY_<STRATEGY>`), which makes broker-level statements per strategy possible.
5. Migrate to `alpaca-py`.
6. Add a live quote for stops and consider returns-based features (with a backtest).

---

## 10. Interview preparation

Five real engineering stories from this audit. The investigation was AI-assisted; the owner built the original system, wrote the decision log that made the reconstruction possible, and supplied the trade export that turned hypotheses into confirmed findings. The detail below is enough to explain and defend each story without overstating anyone's role.

### Story 1: The -94.95% that was not a loss

**Situation.** A paper-trading system with three bots sharing one Alpaca account. On July 7 both bots' kill switches fired at "-94.95%" and liquidated everything.
**Problem.** Was 95% of the money really lost?
**Evidence.** The trade log showed a normal, profitable $204 sale 40 minutes before the switch and no trades in between; both bots logged the same number within 25 seconds.
**Investigation.** Traced the kill-switch code and found two drawdown formulas: the deciding one compared account equity with each strategy's starting capital ($1,000 and $200), the logged one used an in-memory peak. Computing what the deciding formula needed showed the account had to read <= $140 for both to fire, from a book worth roughly $1,400 to $2,700 forty minutes earlier.
**Root cause.** The switch acted on a single, unvalidated, account-wide reading that was compared with the wrong baseline; the logged number was a different formula, so the audit trail could not explain the decision.
**Fix.** Strategy-level equity from a ledger, one drawdown function that both decides and logs, rejection of implausible readings, two-reading confirmation, strategy-scoped liquidation, persisted halt state with a manual reset tool.
**Why that fix.** Every element addresses one link in the failure chain; none of them weakens the control.
**Validation.** Six unit tests including "account reads 95% down but strategy is safe" and "one glitch reading is suspect, a persistent one is accepted".
**Result.** The switch now measures what it claims to measure and cannot be tripped by one bad number. The Alpaca-side origin of the reading is still unknown, and I say so.
**Concepts.** Debugging from logs, risk management, defensive programming, observability, state management.
**30-second version.** "Both bots reported a 95% drawdown and liquidated the account. The log showed a normal profitable trade 40 minutes earlier and nothing in between, so I traced the kill switch: it compared account-wide equity with each bot's notional starting capital and logged a different, peak-based number. One bad broker reading tripped both. I replaced it with per-strategy equity from a ledger, a single drawdown function that decides and logs the same value, plausibility checks, and a two-reading confirmation, all under tests."
**90-second STAR version.** Situation: a three-strategy paper-trading system I had built shared one broker account; in July both equity bots halted with a logged 94.95% drawdown. Task: determine whether the loss was real and make the control trustworthy. Action: I reconstructed the timeline from the trade export, which showed a routine profitable sale 40 minutes before and no trades until the switch. Reading the code, I found the decision used (account equity - strategy capital) / capital while the message used (equity - in-memory peak) / peak; both bots fired within 25 seconds on the same number, which is only possible if they were reading the same account-wide value. For both thresholds to trip, the account had to read at or below $140, which no long-only book of index ETFs does in 40 minutes with no trades. So the reading, not the market, was the trigger. I introduced a per-strategy ledger so each bot has its own cash budget and positions, computed drawdown from a persisted peak in one function that also writes the snapshot, rejected NaN/zero/implausible jumps as suspect, required two consecutive critical readings, and scoped the liquidation to the strategy that tripped. Result: 108 tests pass, including scenarios where the account reads 95% down while the strategy is unaffected, and where a glitch reading is ignored but a persistent one is honoured. I could not prove why Alpaca returned that value, and the document says so.

### Story 2: The 51-share sale that was $51

**Situation.** A prior analysis flagged TSLA buys of 0.1446 shares followed minutes later by sells of about 51 shares.
**Problem.** Was the system selling 350 times what it bought?
**Evidence.** The BUY quantity reproduces exactly from configuration: $50 cap / 345.82 = 0.1446. The SELL "quantity" divided by shares bought equals the live price within 1% across all 3,020 SELL rows.
**Root cause.** Three SELL branches passed the position's dollar market value as the `quantity` argument of the logger; the order path itself was unit-consistent.
**Fix.** A single execution layer that logs shares, the signal price, the fill price and quantity, the broker order id and the status; additive schema migration that preserves history and marks legacy rows.
**Validation.** Tests assert the BUY and SELL rows carry the same share quantity and distinct signal/fill prices; a migration test loads a synthetic old-schema database with the Aug-27 rows.
**Concepts.** Data integrity, unit discipline, reading evidence before believing a hypothesis.
**30-second version.** "The suspicious 51-share sale was $51 of market value logged in the quantity column: BUY rows logged shares, SELL rows logged dollars. I proved it arithmetically from the config, then across every sell in the history, and fixed the logging at the one place that knows the fill."

### Story 3: One bot sold another bot's stock

**Situation.** Suspected cross-strategy interference in a shared account.
**Evidence.** June 2: risky1 logged a $195 AMZN sale; risky1 never bought AMZN; stable had bought it the day before and re-bought it 41 seconds later.
**Root cause.** Positions had no owner; every exit used the account's position for the symbol; the kill switch called the account-wide `close_all_positions`.
**Fix.** Strategy-level ledger in SQLite, sells limited to owned quantity, strategy-scoped emergency close, drift reconciliation, opt-in adoption of pre-existing positions.
**Validation.** A test reproduces the June 2 scenario with a fake account holding 50.1446 TSLA (50 stable, 0.1446 risky1): risky1's sell removes 0.1446 and leaves stable's 50.
**Concepts.** System design, ownership and invariants, API integration limits.
**30-second version.** "Alpaca positions are account-wide, and the bots trusted them as their own. The log proved one bot sold another's shares. I added a per-strategy ledger and made the execution layer sell only what the ledger owns, with tests for the exact scenario."

### Story 4: The oscillator

**Situation.** Risky1 bought and sold TSLA every 2 minutes 21 seconds; 22 round trips in one afternoon.
**Investigation.** The gap equals one loop cycle; all inputs are a static daily bar; the momentum exit and the ML entry are evaluated independently, so with momentum negative and the model predicting 1 the system alternates deterministically. 248 of 250 risky1 sells were momentum exits and none logged a PnL, so the throttle meant to catch cold streaks was blind.
**Fix.** A state machine: no entry while the strategy's own exit condition is true on the bar, no re-entry on the bar just exited, minimum hold for discretionary exits, emergency exits exempt, no dust top-ups.
**Why not a cooldown.** A cooldown would still trade every N minutes on the same bar; hysteresis would change the strategy signal. The contradiction had to be removed where it was created.
**Validation.** Four-cycle regression test: the historical BUY-SELL-BUY-SELL sequence now yields one BUY and one SELL.
**Concepts.** State machines, root cause versus symptom, transaction-cost awareness.

### Story 5: Restoring a dormant ML system

**Situation.** A ZIP of an idle repository, no models, no database, Python 3.14 on the machine, pinned dependencies that no longer install.
**Actions.** Snapshotted the tree, probed the pins (numpy wheel missing, aiohttp pin unbuildable, vectorbt versus plotly 7), separated compatible ranges from the historical pins, made imports lazy so the system boots without models or keys, fixed the paused retrain job and the never-reloaded model, replaced the CWD-relative model path, found the PPO inference reading the oldest bar, added a fake broker and 40 regression tests, and wrote the audit as I went.
**Result.** The project starts, all 108 tests pass on the modern stack, CI covers 3.11 (pinned) and 3.12 (ranges), and every historical claim is either reproduced from data or marked unverified.
**Concepts.** Reproducibility, dependency management, test doubles, documentation as an engineering artifact.

---

## 11. Second-pass adversarial review of the first-pass fixes

Performed after the first pass, on the assumption that the first pass contained mistakes. Method: re-read every new or rewritten module (`core/portfolio.py`, `core/execution.py`, `strategies/common.py`, `core/logger.py`, `tests/fakes.py`) looking for crash windows, ordering hazards, unit and state mistakes, and behaviour the fake broker could not have exercised; then write a failing test for each suspicion before fixing it.

### 11.1 Classification

| Id | Defect | Introduced by | Severity | Status |
|---|---|---|---|---|
| S-01 | Order accepted by the broker but never recorded locally (crash or network timeout between `submit_order` and writing the broker id); `"submitting"` was not an "open" status so restart reconciliation ignored it | **First-pass fix** (new execution layer) | HIGH | Fixed + tested |
| S-02 | All frames fetched back to back; the original 20 s spacing that kept Polygon's free tier under its rate limit was removed | **First-pass fix** (regression) | HIGH (operational) | Fixed + tested |
| S-03 | Drift check compared only this strategy's ledger with the account; when the account held less than the sum of all ledgers a strategy could still sell another strategy's shares | **First-pass fix** (incomplete ownership guard) | HIGH | Fixed + tested |
| S-04 | `reconcile_open_orders` raised on an oversized sell fill and ran outside the cycle's exception handling, so one bad order would fail every cycle forever; an order the broker no longer knew (404 after a paper reset) blocked its symbol forever; a broker outage blocked forever | **First-pass fix** | MEDIUM | Fixed + tested |
| S-05 | Ledger fill and order-row update were two transactions (crash between them double-books on restart); a fill reported without an average price was recorded as applied without being applied | **First-pass fix** | MEDIUM | Fixed + tested; the fix itself had a bug caught by its test (see 11.3) |
| S-06 | Dust remainders below the broker's $1 minimum were resubmitted and rejected every cycle | **First-pass fix** (new sell-own-quantity path) | LOW | Fixed + tested |
| S-07 | A rejected BUY was retried every cycle on the same bar (correctly labelled now, but the same broker hammering as the original phantom-BUY pattern) | **First-pass fix** | LOW | Fixed + tested |
| S-08 | Positions in symbols removed from the configured asset list were never marked, stopped out or exited (original code had the same gap; the ledger made it visible) | Original design, surfaced by the ledger | MEDIUM | Fixed + tested |
| S-09 | After a kill switch whose sells were rejected or pending, the halted strategy never retried flattening; open risk with no management | **First-pass fix** (halt semantics) | MEDIUM | Fixed + tested |
| S-10 | Concurrency of three strategy threads on one SQLite file was asserted, not tested | Gap in first-pass validation | LOW | Tested (no code change needed) |

None of these existed as such in the original code, because the original had no ledger, no reconciliation and no order tracking at all; they are the failure modes of the new machinery. The original's equivalents were worse (no recovery at all), but that does not excuse them.

### 11.2 Fixes

- **S-01.** Every order gets a `client_order_id` written to the `orders` row before the order is sent (new nullable column, added by the additive migration). `"submitting"` and `"unresolved"`/`"awaiting_fill_price"` are open statuses. Reconciliation looks up an order by broker id and falls back to `get_order_by_client_order_id`. If `submit_order` raises, the code checks by client id before declaring the order rejected; if the broker has it, execution continues with it.
- **S-02.** The per-ticker sleep moved into the frame-fetch loop (no sleep after the last fetch). Marks remain consistent because the daily close does not change within a cycle.
- **S-03.** Before a sell, `available = account_qty - sum(other strategies' ledgers)`; this strategy may sell at most `available`, writes off the rest as `RECONCILE`, and refuses to submit when nothing is available. Every strategy applies the same rule, so the total sold can never exceed the account.
- **S-04.** Reconciliation never raises: a fill that cannot be booked is logged as `reconcile_error` and the order closed; a 404 closes the order as `unknown_at_broker`; a non-404 error keeps the order blocking (conservative) until it is older than 48 hours, when it is marked `unresolved_stale` with a log row telling the operator to check the broker. The reconcile call in `run_cycle` is also wrapped.
- **S-05.** `apply_fill(order_update=...)` writes the ledger, the cash budget and the order row in one SQLite transaction. A fill without a price leaves `filled_qty` unchanged and parks the order in `awaiting_fill_price` (open) so it is retried.
- **S-06.** A sell whose notional is below $1 is written off with a `RECONCILE` row instead of being submitted.
- **S-07.** A rejected entry sets signal state `BUY_REJECTED` for the bar; `can_enter` refuses until a new bar.
- **S-08.** `run_cycle` manages the union of configured assets and ledger symbols; entries are allowed only for configured assets.
- **S-09.** A halted strategy calls `close_all_strategy_positions` each cycle until its ledger is empty.
- **S-10.** Three threads performing 40 fills each on separate strategies in one database; totals exact, no errors.

### 11.3 Bugs in the second-pass fixes themselves

The first version of the S-05 fix wrote the broker's status (`filled`) onto an order whose fill could not be booked, which made the order terminal and lost the fill; `test_s05_fill_without_price_is_not_marked_applied` failed and the status was changed to a local open state. Two other initial test failures were test-construction mistakes (asserting on a state that the fix had already advanced past; buying at the wrong fake price), not product defects.

### 11.4 Validation

```
python -m pytest tests/ -q --deselect tests/test_regime.py::TestVIXSignal::test_get_vix_returns_float_or_none
123 passed, 1 deselected   (124 collected: 68 original, 41 first-pass, 15 second-pass)
```

### 11.5 What the second pass did not resolve

- **Deployment readiness: not established.** The evidence supports "the failure modes found so far are closed under a fake broker". It does not support "ready to deploy": no order has been sent to Alpaca's paper API with this code, the deprecated SDK's real behaviour for `get_order_by_client_order_id`, crypto symbol formats in `list_positions`, and post-close queued orders has not been observed, no model has been loaded, and the live database has not been migrated. A supervised paper run of at least several market sessions with the dashboard and `equity_snapshots` under review is the minimum before trusting it unattended.
- Realized P&L booked on a `RECONCILE` write-off uses the signal price, which is an estimate.
- The ledger starts every strategy at `CAPITAL` on migration; historical realized P&L stays in the `trades` table and is not carried into the new cash budget. This is a documented accounting reset, not a loss of data.
- Orders queued while the market is closed (DAY market orders submitted after 16:00 ET) are handled as pending and reconciled, but this path is untested against the real broker.

## Appendix A: analysis scripts used on the export

```python
import pandas as pd
df = pd.read_csv("2026-09-14T16-57_export.csv", index_col=0)
df["ts"] = pd.to_datetime(df["timestamp"]); df = df.sort_values("ts").reset_index(drop=True)
df["date"] = df.ts.dt.date

# counts, reasons, PnL completeness, kill switches, symbols shared by strategies
df.groupby(["strategy", "action"]).size().unstack(fill_value=0)
df[df.action == "SELL"].groupby(["strategy", "reason"]).size()
df[df.action == "SELL"].groupby("reason")["pnl"].apply(lambda x: x.isna().mean())
df[df.action == "KILL_SWITCH"]
m = df[df.action.isin(["BUY", "SELL"])].groupby("asset")["strategy"].nunique(); m[m > 1]

# unit check: SELL quantity / shares bought since the previous SELL / logged price ~ 1.0 means dollars
rows = []
for (s, a), g in df[df.action.isin(["BUY", "SELL"])].groupby(["strategy", "asset"]):
    held = 0.0
    for _, r in g.iterrows():
        if r.action == "BUY": held += r.quantity
        elif held > 0:
            rows.append(dict(strategy=s, asset=a, ratio=(r.quantity / held) / r.price)); held = 0.0
pd.DataFrame(rows).groupby("strategy").ratio.describe()

# churn: gaps between consecutive trades, holding time, distinct prices per day
t = df[df.action.isin(["BUY", "SELL"])].sort_values("ts")
t.groupby(["strategy", "asset"]).ts.diff().dt.total_seconds().div(60).groupby(t.strategy).describe()
t.groupby(["strategy", "asset", "date"]).price.nunique().describe()

# dust top-ups and phantom buys
b = df[df.action == "BUY"].assign(notional=lambda x: x.price * x.quantity)
((b.strategy == "stable") & (b.notional < 5)).sum()
```
