# Decision Log to keep track of what I was thinking while building

## March 28, 2026-Configuration of the system and allocation of the money and where to invest
**CONTEXT**: 

Began building the program. Created structure of the overall project, wrote config.py, readme, core folder, requirements.txt and gitignore and logger.py

Options 1: Don't put env in gitignore or put it in
Decision 1:
Put env variables into gitignore
Why: Security

Options 2: Give all bots equal capital , or split differently
Decision 2: Chose to split starter funds between 1000, 200, and 200 for stable,risky1,risky2
Why: Stable should have greater funds since less risk is always better and its better to have a safety net.

Option 3: Make stable assets dynamic like risky, or make it fixed
Decision 3: Chose to make stable assets fixed for now
Why: If it was ML like risky1, then it could get tricked by a stock that does well but is actually really risky. The stocks given are proven to be stable and "safe bets"

option 4: Choose another day amount or make it 14 days
Decision 4: 14 day trading interval
Why: 7 days is too volatile because one week can be very different from another week. Is 2 week also volatile? Yes, but it still gives more time for training and keeping the bot accurate.


Options 5: Make percentages for kill switch different or this
Decision 5: Choosing 15% and 30% kill switch for stable vs risky
Why: 15% fits stable bot profile and can be possibly regrown. 30% in a risky is because risky is more aggressive and more volatile, but also because if more than 30 is lost then its time to stop.

Options 6: Use SQLite or another alternative
Decision 6: Use SQLite
Why: There is no setup, uses a single file to transfer data, and no server process and since its only done locally and for myself, a larger database such as PostgreSQL or MySQL is not helpful at this stage or will be beneficial overall.

Option 7: run pnl and reason as opptional or make it mandatory
Decision 7: Make it optional 
Why: there might be no valid answer the system will give

Option 8: Create 3 bots or 4 bots
Decision 8: Make stable, risky1 and risky 2
Why:I didn't create a stable2 for crypto as the market does not have enough historical data for me to trust it 


**RESULT OF 3/28/2026**
Project pending, foundation complete, allocation of funds, kill switch settings, list of potential assets, trade logging complete, configuration of system done. Next step is providing the data pipeline.


## March 28, 2026 (midday)-Feature Engineering
**Context**
A way to transform the raw data of prices into understandable data that ML models can learn and train on was the work done at this time.

**Options Considered 1:**
- Separate features into multiple files per bot
-  Build features into a single file for all bots
**Decision 1:** Build a single file for all bots
**Why**: To reduce complexity of the overall program, and because these features are fairly similar across all strategies, it justs needs adjustment per strategy and that can be done in the input.

**Options Considered 2:**
- Copy the dataframe per features before adjusting the new copy
- Modify the original dataframe
**Decision 2:** Copy and modify the copy of the dataframe.
**Why:** Should the data get corrupted while doing the features, it would be best to have the original data still safe and secure

**Options Considered 3:** 
- Use normal averages of the entire time
- Use rolling averages 
**Decision 3:** Use rolling averages
**Why:** To keep the bot more accurate. Past data may no longer reflect current markets and may actually be harmful in future predictions

**Results of March 28th 2026 Midday session**
Feature engineering is now complete. Indicators that were implemented: rolling averages, RSI, momentum, Bollinger Bands, ATR, Z-score, and volume ratios. Tests have not yet happened because system is not yet ready for testing.

## March 28, 2026 Late Evening - Polygon Data Fetcher

**Context:**
Needed a source of real etf and stock price data for the bots

**Options Considered 1:** times to analyze
- Default to minute
- default to hours
- default to day
- default to other data

**Decision 1**
Stick to day data since we are sticking to free tier and minute/hour would require paid tier for Polygon.

**Options Considered 2:** how recent should the lookback of the bar be
- Fetch only today
- Fetch 5 days
- Fetch multiple days ago

**Decisions 2**: Choose 5 days since that is close enough to be related but not too small that it will be completely disregarded by outliers

**Options considered 3** Handling of empty data
- Crash
- Return with warning
- Ignore and move on

**Decision 3:** Return with a warning because the user should be warned there is no data, but the bot shouldn't simply crash. The bot also shouldn't move on without informing user because something might be wrong

**Result:** Still pending, we haven't tested yet

## March 28, 2026 (Early Night) — Switched from Kraken to Polygon for Crypto

**Context:**
I am in a location that is geoblocked by Kraken at the time of this bot creation. Kraken was supposed to be the crypto checker for risky2

**Options Considered1:**
- Lie about location to use Kraken
- Use Gemini (requires phone number which I don't want to give out for this)
- Use Coinbase Advanced
- Use Polygon for data and Alpaca for execution of the transactions

**Decision1:**
Decided to go with the final option 

**Why1:**
since I am already using polygon for the stocks and etfs. I just wanted to have more options in case polygon failed to work, but its not worth the effort of making new accounts and ensuring all locations work, and making a whole new plan of what libraries are needed, etc. Finally, using one data source for all bots will make it easier and I won't have to do any changes to build_features() for crypto as well. :D

**Result1:**
Pending system completion.

## March 28, 2026 (Early Night) — Added sentiment fetcher 

**Context1:**
Need something to calculate sentiment

**Options Considered1:**
- Use Textblob
- Use BERT
- Use other complex models

**Decision1:**
Decided to go with the first option 

**Why1:**
Textblob takes 2 lines to setup, its instant, its decent accuracy (not as good as more complex models, but works well enough), is beginner friendly, and it runs locally on the model, without requiring additional hardware. The only limitations is that it doesn't understand financial jargon, and fails to consider context.

**Result:**
Pending system completion.



**Context1:**
What to return when there is no news?
**Options Considered1:**
- Return error
- return positive
- return negative
- return neutral

**Decision1:**
Decided to go with the final option 

**Why1:**
Since there is no news, it is literally neutral, there is no feelings. It is possible there won't be no news so error is off the table, and its not necessarily positive or negative so I cannot do that.

**Result1:**
Pending system completion.


**Context1:**
What to apply in score column for rows

**Options Considered1:**
- Use different scores per row
- Use scores only in first row
- Use same score for all rows

**Decision1:**
Decided to go with the final option 

**Why1:**
Since the rows are talking about the same content, and we are inputting average score, we should use the same score for all rows.

**Result:**
Pending system completion.




## March 28, 2026 (Late Night) - Regime Detector

**Context1:**
Need to determine what to check first : volatile or trending
**Options Considered1:**
- Check trending first
- check volatile first

**Decision1:**
Decided to go with the final option 

**Why1:**
If you check trending first, then if its trending because its volatile, thats bad. If its volatile, then you aren't trading.

**Result:**
Pending system completion.


**Context2:**
Where to allow stable, either in RANGING and TRENDING and /or VOLATILE
**Options Considered2:**
- Allow in all
- Allow in first 2
- Choose something else

**Decision2:**
Decided to go with the second option 

**Why2:**
stable cannot be in volatile because it is opposing the exact meaning of volatile. Ranging and trending however could fit within the expectations
**Result:**
Pending system completion.




**Context2:**
I need to decide which bots to allow in VOLATILE
**Options Considered2:**
- Allow  all
- Allow in first 2
- Allow only risky2
- other choices
**Decision2:**
Decided to go with the third option 

**Why2:**
risky2 could be in volatile because crypto is inherently volatile, and doesn't have as much history as risky1. Additionally new coins could result in market change. However, stock markets are significantly more stable and as such should not be as volatile, thus risky1 is not allowed
**Result:**
Pending system completion.






## March 29, 2026 (Late Night) - Stable and Risky1 Supervised Learning Model Trainer creation

**Context1:**
Need to choose something to train model
**Options Considered1:**
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Spark MLib GBT
- AdaBooost
- LinearBoost

**Decision1:**
Decided to go with XGBoost

**Why1:**
XGboost is a industry standard and is used often. LightGBM was also considered greatly as it is comparable, and occasionally even better than XGBoost. This was a coin toss because LightGBM was essentially the same based on what my use would be (Its not going to have a huge difference in speed due to what we are testing on-years of daily bars of data). CatBoost is for categorical, I am using numbers. Random Forest is weaker than XGBoost on financial data and tends to be more overfitting. Spark MLib GBT would require more computers but I am using a single VPS
AdaBoost is slower, less accurate than XGBoost and older. I think XGBoost is considered the successor of AdaBoost? They both have Boost in their names
LinearBoost: Finance is not necessarily linear and might not be as good a fit.
**Result:**
Pending system completion.


**Context2:**
Need to decide how to split data into testing and training
**Options Considered2:**
- Standard split of 80 and 20 (First learned during ESE 188 understanding machine learning class in freshmen year, then learned this at Incture in India internship)
- 50 50
- other proportions

**Decision2:**
Decided to go with standard split
**Why2:**
Keeps enough test data (20 for test) while giving 80 for training. Theres enough to train without giving away the answers of the 20 testing. 
**Result:**
Pending system completion.


## March 29, 2026 (Late Night)— Per-Bot PAPER_MODE Flags

**Context:**
Bots need to be chosen when to be live or be paper traded because if its still be trained you don't want to use real money. 

**Options Considered:**
- One global PAPER_MODE flag for all bots
- Per-bot PAPER_MODE dictionary

**Decision:**
- Per bot paper mode

**Why:**
Stable and risky1 use supervised ML which is easier to test and make it go live earlier. risky2 is RL and would need a lot more paper trading time to prove itself. A single change would make them all switch at the same time. Its best to avoid risking money when not needed. Also, risky1 should also take more time than stable due to its inherent nature of being risky.

**Result:**
Pending system completetion.

## March 29, 2026 (Late Night)— Retraining of supervised learning bots creation

**Context:**
What data to train on? 
**Options Considered:**
- Once a year
- Once every two weeks
- Some other time
- 2 years

**Decision:**
- per 2 years

**Why:**
Train on 2 years training data because it is recent enough to be effective but not so recent that it will be entirely affected by outliers

**Result:**
Pending system completetion.


**Context:**
How to schedule retrain?
**Options Considered:**
- background scheduler
- blocking scheduler

**Decision:**
- background

**Why:**
We don't really need to see the scheduling reset every single time we look because its not the main point of the program

**Result:**
Pending system completetion.


## March 30, 2026 — Building the backtesters

**Context:**
need to decide how much fees and slippage will happen
**Options Considered:**
- 0
- 0.1%
- some other percentage

**Decision:**
- .1%

**Why:**
.1% is about the amount of fees that Alpaca will do during the trades. Slippage of .1% is also in the reasonable range
**Result:**
Pending system completetion.


**Context:**
there might be errors when running a ticker
**Options Considered:**
- ignore error
- try catch error
- throw error
- crash

**Decision:**
- try catch

**Why:**
Even if one ticker fails, some of the other tickers might still be fine. At least this way we know which ticker is broken and the other tickers still go through

**Result:**
Pending system completetion.



**Context:**
how much initial cash in test
**Options Considered:**
-  same amount as actual models
- 1000
- some other amount

**Decision:**
- 1000

**Why:**
1000 is a nice number to easily see how well the model is doing and to compare it with other options
**Result:**
Pending system completetion.



## March 30, 2026 — Hosting Decision

**Context:**
Need a place to host the bots 24/7

**Options Considered:**
- Oracle Cloud Free Tier (always free)
- Microsoft Azure (free via student account, ~2+ years)
- DigitalOcean ($6/mo)
- EliteBook laptop (unreliable)
- Google Cloud (free tier very low specs though)
- AWS Free for 6 months. Also same problem regarding student plan as Azure

**Decision:**
Oracle Cloud Free Tier (ARM Ampere A1 instance)

**Why:**
Oracle's ARM instance gives 4 CPUs and 24GB RAM completely free forever which is significantly better specs than Azure's free student tier for a compute-heavy workload like this. Azure student credits expire, Oracle's always-free tier does not. Azure kept as a backup option if Oracle signup fails. This project I intend to keep running for the foreseable future after all.

**Result:**
Pending — deployment happens later



## April 3, 2026 - Treasury 10 year metric choice

**Context:**
Needed to choose a rate for comparison

**Options Considered:**
- risky rate
- non_risky rate
- 10 year treasury safe

**Decision:**
10 year treasury safe rate
**Why:**
The 10 year treasury rate is considered a safe investment that will not fail.
**Result:**
Pending — deployment happens later


## April 3, 2026 - Fallback

**Context:**
Needed to choose a rate for fallback in case live fetch fails

**Options Considered:**
- 0%
- 4%
- 10%

**Decision:**
4%
**Why:**
The current rate is about 4%, so it will fit the current climate.

**Result:**
Pending — deployment happens later


## April 3, 2026 - Amount of trading days

**Context:**
Needed to choose amount of days to trade

**Options Considered:**
- 365
- 252
- 10
- other

**Decision:**
252 
**Why:**
252 days is considered the industry standard of business days in the United States
**Result:**
Pending — deployment happens later



## April 3, 2026 - How to get treasury rate

**Context:**
needed to have a rate for metrics
**Options Considered:**
- live fetch
- fixed hardcoded fetch
- live fetch, then if that fails do hardcoded

**Decision:**
Last option
**Why:**
Live fetch would be more accurate to what is being chosen, and the fallback is reasonable and in case live fails.
**Result:**
Pending — deployment happens later

## April 3, 2026 - time to sleep between cycles

**Context:**
deciding how logn to wait per cycle
**Options Considered:**
- no sleep
- one min
- one hour
- some other time

**Decision:**
one min
**Why:**
avoid rate limiting while not taking too long
**Result:**
Pending — deployment happens later

## April 3, 2026 - when to place kill swithc

**Context:**
need to decide when to check kill switch
**Options Considered:**
- before trading
- after trading
- during trading

**Decision:**
before trading
**Why:**
if the kill switch is active before the model runs, then the market is extradonarily bad and shouldn't buy right now
**Result:**
Pending — deployment happens later


## April 3, 2026 - try catch in the run logic of stable

**Context:**
how many try in run()?

**Options Considered:**
- one
- 0
- 2
- more

**Decision:**
2
**Why:**
each ticker may have its own error, if so it will simply go to the rest.. however, the user might want to manually stop, thus resulting in keyboardinterrup also, if another error happened for the entire thing it owuld be stored as e and mentioned.
**Result:**
Pending — deployment happens later





## April 3, 2026 - When to put momemntum check in risky1.py

**Context:**
need to decide when to check for momentum exit
**Options Considered:**
- before ml signal check
- during ml signal check
- after check

**Decision:**
before ml signal check
**Why:**
momentum exit doesn't rely on ml signal check, and it would be faster and save more resources
**Result:**
Pending — deployment happens later




## April 3, 2026 - when to return

**Context:**
momentum sell in riskyy1 was just completed, if it says to momentum exit what to do
**Options Considered:**
- return now
- ignore
- something else

**Decision:**
return
**Why:**
after momentum exit it won't do anything for this run. 
**Result:**
Pending — deployment happens later




## April 3, 2026 - how many times to call alpaca

**Context:**
need to call alpaca, how many times?
**Options Considered:**
- once for all strategies
- directly each strategy
- something else

**Decision:**
once
**Why:**
simplifies the project and will still work just fine
**Result:**
Pending — deployment happens later



## April 3, 2026 - why does risky2 ignore market check?

**Context:**
stable and risky1 must check for if the market is open or close. what about risky2
**Options Considered:**
- check
- ignore
- something else

**Decision:**
ignore
**Why:**
the market is open 24/7 365 days a year. it never closes unless the market has failed
**Result:**
Pending — deployment happens later



## April 3, 2026 - Testing Suite

**Context:**
Need to check the core functions without using the actual api keys
**Options Considered:**
- dont test
- use real api keys
- use magicmock and dummy env var

**Decision:**
use dummy env vars and magicmock since magicmock fakes the external dependencies 
**Why:**
(was necessary cause polygon clietn would crash at import without a key during the tests, and i can't really use the real env key there)**Result:**
Pending — deployment happens later
Some failed attempts actually was usng patch() to mock the restclient, which failed because polygon would initialize the client when importing, then patch would have come, making patch unable to stop it. Thus i set a dummy env var instead


**Options Considered 2:** What to test
- Skip edge cases
- Test only happy path
- Test edge cases too (empty returns, flat returns, etc.)

**Decision 2:** Test edge cases
**Why:** The flat returns test caught a real floating point bug in sharpe_ratio() — excess.std() was never exactly 0.0 due to floating point precision. Fixed by checking < 1e-10 instead of if it is 0. Tests proved their value there by providing a bug.

**Result**
4/4,8/8, and 16/16 tests passed


## April 3, 2026 — run.py System Launcher

**Context:**
Need a single entry point to start the entire system

**Options Considered 1:** How to run multiple bots
- Run one after another
- Run in separate threads at same time
- Run in separate processes

**Decision 1:** Threads
**Why:** one after another would mean stable bot runs forever in its while True loop and risky1 never starts. Threads let both run simultaneously sharing the same memory and timing. Separate processes would be overkill for this use case and waste resources.

**Result:** Pending — not yet deployed

---

## April 3, 2026 — Dashboard Design

**Context:**
Need a way to visualize strategy performance

**Options Considered:**
- Terminal only (compare.py)
- Web dashboard (Streamlit)
- Both

**Decision:** Both
**Why:** Terminal is fast for debugging during development in case someone who doesn't know the code wants to attempt editing it. Streamlit gives a live URL for the resume and shows users a real deployed product that they can understand, even without being programmer geniuses. Capital allocation pulled directly from config.py so dashboard always reflects actual settings without hardcoding the details each time.
 
**Result:** Pending — not yet deployed




## April 3, 2026 — DevOps Time!

**Context:**
Need to make system reproducible, testable, and deployable for other people without significant modifications and bug testing

**Options Considered 1:** Containerization
- No Docker, just run Python directly on server
- Docker
- something else

**Decision 1:** Docker
**Why:** Guarantees same environment on Oracle Cloud as locally. No "works on my machine" problems. Anyone can clone and run docker-compose up to start the full system.

**Options Considered 2:** CI/CD
- Manually run tests before pushing
- GitHub Actions auto-run on every push
- other

**Decision 2:** GitHub Actions
**Why:** Automated tests catch bugs before they reach the live system.Easier to check. Tests now run on every push to main with dummy API keys so no credentials are exposed.

**Result:** Docker setup complete, CI passing 16/16 on push


## April 10, 2026 - RL Bot Architecture for risky2

**Context** 
A third strategy for crypto taht goes beyond supervised ML needs to be chosen. XGBoost does predict direction but it cannot manage positions over time

**options considered 1:** which RL algorithm to choose?
- PPO (proximal policy optimization)
- DQN( deep q-network)
- A3C (asynchrnous advantage critic)
- SAC (soft actor critic)

**Decision 1: **PPO
**Why:** PPO is a industry standard in trading bots, it is more stable than DQN, easier and simpler than A3C and it works well in discrete actions. Stable Baselines3 also has a clean implementation of PPO that integrates directly with our gymnasium environment

**options considered 2** how many actions to do?
- BUY/SELL (like xgboost)
- HOLD/BUY/SELL

**decision 2:** last option
**why** XGBoost only predicts direction but the RL also manages the positions over time, thus needing o learn when holding is a right move, not just buy/sell. If no HOLD it would over trade.


**Options Considered 3:** Training data for lookback
- 2 years (same as XGBoost)
- 1 year

**Decision 3:** 1 year
**Why:** XGBoost is supervised so more data helps. RL learns by trial and error — too much historical data risks overfitting to old and outdated regimes like the 2022 bear market patterns that don't reflect current conditions.

**Options Considered 4:** gamma value
- 0.0 (instant profit only)
- 0.99 (value future rewards almost equally)

**Decision 4:** 0.99
**Why:** At gamma=0 the agent only cares about instant profit and would chase tiny gains constantly. At 0.99 it's willing to hold a positon patiently for bigger future rewards which is better behavior for a crypto trading bot.

**Results:**
Stable and risky1 are running on old elitebook. risky2 has not been trained yet, will train once cloud opens up since elitebook is too old to handle.




## April 11, 2026 - Smart Sleep Duration

**Context:** Bots were pinging every 60 seconds even when market was 12+ hours from opening, wasting Azure credits.

**Decision:** Dynamic sleep based on time until market open.
- >24h until open : sleep 12h
- >12h until open : sleep 8h  
- >8h until open : sleep 3h
- >1h until open: sleep 30min
- <1h until open: sleep 1min
- Market open: trade normally

**Why:** Alpaca clock API tells us exact time until next open-works for any timezone and handles weekends/holidays automatically. Reduces API calls by ~99% during closed hours.
**Result:** Live on Azure — sleeping 12 hours on Friday night 






## April 11, 2026 — Week 6 Risk Management Upgrade

**Context:**
System was running with only a basic kill switch. I felt like a multi-layer risk management to protect capital and profits would be better.

**Decision 1: Peak equity vs starting capital for drawdown**
- Starting capital: doesn't protect profits made
- Peak equity: protects profits — if bot makes $200 then loses it all back, peak equity catches it, starting capital wouldn't
- Decision: Peak equity 

**Decision 2: Stop loss percentage per strategy**
- Too tight (2%) gets triggered by normal daily noise
- Too loose (15%) doesn't protect against real losses
- Decision: stable 5%, risky1 8%, risky2 12%
- Why: ETFs move 1-2% daily so 5% is a real signal. Crypto is extremely volatile so a 12% breathing room was better.

**Decision 3: Warning threshold at ~50% of kill switch**
- stable: warn at 7% (kill at 15%)
- risky1: warn at 15% (kill at 30%)
- Why: gives time to reduce position sizes before full shutdown. Two layers of protection instead of one sudden stop.

**Decision 4: Sortino over Sharpe for primary metric**
- Sharpe penalizes ALL volatility including upside
- Sortino only penalizes downside volatility
- For a trading bot upside volatility is desirable
- Decision: keep both, add Sortino as additional metric 

**Decision 5: Smart sleep duration to save Azure credits**
- Bots were pinging every 60 seconds even when market was
  12+ hours from opening — wasteful
- Use Alpaca clock API to calculate exact time until open
- 24h: sleep 12h, >12h: sleep 8h, >8h: sleep 3h,
  1h: sleep 30min, <1h: sleep 1min
- Works for any timezone, handles weekends/holidays automatically
- Reduces API calls by ~99% during closed hours 

**Result:** All changes live on Azure, 
tests passing in github actions







## April 11, 2026 — Equity Curve Filter (equity_curve_filter.py)

**Context:**
The existing risk_manager.py handles per-trade risk: stop loss thresholds, position sizing, and the peak equity kill switch. However, there was no mechanism to detect sustained cold streaks — periods where the bot is consistently losing across multiple trades. A single bad trade is noise. A downward-trending equity curve is a signal.

**Decision 1: Separate file from risk_manager.py**
-  Add equity curve logic directly into risk_manager.py
- Create a dedicated equity_curve_filter.py

- Decision: create new file
- Why: risk_manager.py operates at the per-trade level. Equity curve logicoperates at the macro level — it determines whether the bot should be trading at all, not how large a single trade should be. Mixing these two responsibilities into one file makes both harder to test and harder to explain. The separation also reflects how the system runs: equity_curve_filter  is called once per cycle, risk_manager is called once per ticker.

**Decision 2: Three-state system instead of binary on/off**
- Binary -either trade or don't
- Three states: FULL, THROTTLE, HALT

- Decision: three states
- Why: A binary system is too abrupt. If equity dips slightly below its moving average, cutting all trading is an overreaction. Throttling to 50% position size first gives the bot room to recover before a full halt is triggered. THROTTLE acts as an early warning layer; HALT is the hard stop.

**Decision 3: Equity reconstructed from closed trades, not a snapshot table**
- Add a dedicated equity_snapshots table to SQLite
- Reconstruct equity as starting capital + cumulative profit and loss from the existing trades table

- Decision: Reconstruct from closed trades
- Why: A new table would require schema changes and additional write logic in every bot. The trades table already contains all the data needed. Only closed trades (pnl IS NOT NULL) are included — open positions are excluded because unrealized PnL is not settled capital.

**Decision 4: Thresholds**
- Moving average period: 10 closed trades
- Throttle trigger: drawdown > 5% OR equity below its 10-trade MA
- Halt trigger: drawdown > 12% (complements the existing peak equity kill switch in risk_manager.py — that one is per-strategy, this one is window-based)
- History window: last 50 closed trades

- Why: 10-trade MA is responsive without being noisy. 5% throttle and 12% halt mirror the warning/critical structure already established in risk_manager.py, keeping the risk philosophy consistent across the system.

**Result:** File complete. Will be wired into stable.py and risky1.py
trading loops when those files are revisited - called once per cycle,before the per-ticker loop begins.



## April 11, 2026 — ATR-Based Dynamic Kill Switch

**Context:**
Static drawdown thresholds (15% stable, 30% risky1) don't account for  volatility. During high-volatility regimes, normal price swings can trip the kill switch when not needed



**Decision:** Scale kill switch thresholds dynamically using current ATR relative to a per-strategy baseline.

- scale = current_atr / baseline_atr, clamped to [0.5x, 1.5x]
- High ATR → thresholds widen (avoid noise triggers)
- Low ATR → thresholds tighten (stricter protection in calm markets)
- Falls back to static thresholds if ATR unavailable

**Why:** ATR was already computed in features.py — no new data needed. current_atr is optional everywhere so nothing breaks on startup.

**Result:** Wired into get_portfolio_risk_level, get_position_size, and should_close_position. Tested and passing.



## April 12, 2026 — Week 8: ADX Regime Detection

**Context:**
Regime detector was using SMA crossover to detect trends, which gives false signals in choppy markets where price keeps crossing the average for no real reason.

**Decision:** Replace SMA crossover with ADX (Average Directional Index), threshold 25.
- ADX measures trend strength regardless of direction
- SMA crossover fallback kept for when fewer than 14 bars are available
- ADX > 25 = real trend, below = ranging

**Why:** ADX is the industry standard for regime detection. SMA crossover can show "trending" just because price drifted slightly, ADX requires sustained directional movement to confirm.

**Result:** the tests passed, system pending actual results during trading.


## April 12, 2026 — Week 8: VIX Macro Signal

**Context:**
Regime detection was purely per-asset (ADX, ATR). No macro-level fear signal existed. A bot could be trading normally into a market crash because the individual stock ADX looked fine.

**Decision:** Fetch live VIX from FRED API once per cycle, pass into regime check.
- VIX < 20 then normal trading
- VIX 20-30 then elevated, risky bots still trade but equity curve filter handles sizing
- VIX > 30 not good, risky1 and risky2 sit out, stable continues

**Why:** FRED is free with no API key. VIX covers macro fear that per-asset indicators miss entirely. Stable keeps trading because it's designed for rough markets.

**Result:** the tests passed, system pending actual results during trading again

## April 12, 2026 - Week 8: SearXNG Macro Circuit Breaker

**Context:**
VIX is a lagging indicator -it reacts after price moves. Needed something that could catch macro risk events from headlines before price action hits.

**Decision:** Use self-hosted SearXNG instance (HuggingFace Space) to search macro keywords each cycle. Score results, use AI to classify ambiguous cases.

**Signal flow:**
- keyword score ≤ 1 then CLEAR
- keyword score ≥ 8 -> DANGER
- ambiguous (2-7) use SAP AI Orchestration if that fails go to Gemini fallback then failsafe is keyword fallback

**Why SearXNG over paid APIs:** Already had a live HuggingFace instance running fir the ai knowledge system (check out my github for it!). Free, private, no rate limits that matter at our call frequency.

**Why SAP + Gemini for ambiguous cases:** Keyword matching alone is too blunt — "federal reserve interest rate" appears in normal news daily. AI classification handles nuance. SAP AI Core was available from a previous project and I have permission to use it, and it wont cost any additional money so why not. Gemini Flash-Lite as fallback because it's free and fast.

**Why CLEAR on failure:** SearXNG going down should never stop the bot. Macro signal is advisory, not mandatory.

**Wiring:** DANGER raises effective VIX to 35, CAUTION raises to 22. Plugs into existing VIX logic — no new kill switch code needed.

**Result:** Live, tested, returning correct signals.




## [DATE] — Discord Webhook Heartbeat Monitoring

Decision: Add per-cycle Discord heartbeat pings from stable.py and risky1.py
Why: Need real-time visibility into bot health without SSH.
         Kill switch alerts must be instant.
Results: Pending check


## May 30, 2026 - Changing Retraining Time

**Context:** Oracle Cloud Instance needs more compute use otherwise it will be reclaimed by oracle and thats not good
**Decision:** Increase training time from 14 days to 3 days
**WHY** because it will increase compute use
**result** unknown

## June 1st, 2026 - Polygon API key amount

**Context:** polygon keeps rate limiting
**Decision** increase amount of api keys
**WHY** it doesn't break any TOS and it reduces compltexity
**result** implementing


## June 2, 2026 — Major System Fixes
**SearXNG sentiment replacing Polygon news**
Problem: Polygon news API rate limiting sentiment fetcher
Solution: SearXNG local instance as primary, Polygon as fallback
Result: sentiment 429 errors eliminated, 24hr cache added

**SAP AI Orchestration payload format fix**
Problem: SAP API changed format, classification returning None
Solution: orchestration_config wrapping module_configurations
Result: SAP classifying correctly, Gemini fallback rarely needed

**Macro scorer false positives**
Problem: Reddit doom posts triggering DANGER when market at ATH
Solution: news-only sources, higher thresholds (DANGER=12), 
          negative keywords for bullish signals
Result: score dropped from 10 to 0 during bull market

**ADX threshold lowered for risky1**
Problem: ADX=25 too strict, risky1 never trading
Solution: strategy-specific thresholds, risky1/risky2 use ADX=20
Result: risky1 placed first trade (NVDA) on June 2

**Risky1 assets added**
Decision: NVDA, AMD, TSLA, META for momentum strategy
Why: high volume, strong trends, volatile enough for signals


## June 3, 2026 — Per-Ticker PPO Models for Risky2

**Context:**
Single combined model trained on all 8 crypto tickers produced
bad results — ep_rew_mean inflated to 4e+05, entropy near 0,
value loss 2e+07. Root cause: BTC at $66k dominates reward
signal vs DOGE at $0.09, model learned to only care about BTC.

**Decision:** Train one PPO model per ticker (Option A)
**Why:**
- Each ticker has unique price patterns and volatility
- Per-ticker models eliminate price scale bias in rewards
- 100k timesteps per ticker × 8 tickers ≈ same total compute
- Falls back to legacy model if ticker model missing

**Files changed:**
- models/rl_train.py — get_model_path(), train_all_tickers()
- strategies/risky2.py — MODELS dict, loads per-ticker at startup

**Result:** Training in progress on Oracle ARM

---

## June 3, 2026 — RSS Feeds for Sentiment

**Context:**
Polygon news API hitting 429 rate limits constantly for sentiment.
SearXNG returning 0 results for individual stock tickers.

**Decision:** Switch to Yahoo Finance + Reuters RSS feeds
**Why:**
- Yahoo Finance has per-ticker RSS (free, no API key)
- Reuters covers general market context
- feedparser library handles parsing cleanly
- No rate limits, no API keys, updates every few minutes
- 24-hour cache per ticker keeps API calls near zero

**Result:** 429 errors eliminated on sentiment fetching

---

## June 3, 2026 — Risky2 Symbol Format Fix

**Context:**
Alpaca stores crypto as BTCUSD (no slash) but code was
converting X:BTCUSD → BTC/USD (with slash). Position check
always returned 0, bot kept buying same ticker repeatedly.
Cash went negative (-$91.49).

**Decision:** Convert X:BTCUSD → BTCUSD (just remove X: prefix)
**Why:** Match Alpaca's actual storage format exactly.
**Result:** Duplicate buys eliminated, position checks working

---

## June 3, 2026 — RL Reward Function Fix

**Context:**
Risky2 was always HOLDing (action 0). Root cause: BUY reward
was -0.001 (negative!) while HOLD reward was 0.0. Agent
rationally learned HOLD = always better than BUY.

**Decision:** Redesign reward function to incentivize trading:
- BUY: +0.001 (small positive to encourage exploration)
- SELL: normalized PnL × 1.5 bonus for profitable trades
- HOLD with winner: small positive reward
- HOLD with loser: stronger penalty
- HOLD with no position when price rising: small penalty

**Result:** Agent now buys crypto — BTC, SOL, AVAX, DOGE all
trading on first cycle after retraining