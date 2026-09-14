import os
from dotenv import load_dotenv

load_dotenv()

#discord notification setup
DISCORD_WEBHOOK_URL=os.getenv("DISCORD_WEBHOOK_URL","")

# ---------------------------------------------------------------------------
# PAPER TRADING SAFETY
#
# SinghQuant is a paper-trading system. Every strategy is paper by default and
# the broker wrapper (paper_trading/alpaca_paper.py) refuses to build a live
# client unless BOTH of the following are true:
#   1. PAPER_MODE[strategy] is False, and
#   2. the environment variable SINGHQUANT_ALLOW_LIVE_TRADING is set to the
#      exact string "I_UNDERSTAND_REAL_MONEY".
# The audit found risky2 had been left pointed at the live endpoint
# (PAPER_MODE["risky2"] = False); see ENGINEERING_AUDIT.md issue C-01.
# ---------------------------------------------------------------------------
PAPER_MODE={
    "stable":True,
    "risky1":True,
    "risky2":True,
}
LIVE_TRADING_OPT_IN_VALUE = "I_UNDERSTAND_REAL_MONEY"
ALLOW_LIVE_TRADING = os.getenv("SINGHQUANT_ALLOW_LIVE_TRADING", "") == LIVE_TRADING_OPT_IN_VALUE

#Alpaca Details
ALPACA_API_KEY=os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY=os.getenv("ALPACA_SECRET_KEY")
#if PAPER_MODE=true then this is active, else its to live url
ALPACA_PAPER_URL ="https://paper-api.alpaca.markets"
ALPACA_LIVE_URL="https://api.alpaca.markets"

# Optional per-strategy Alpaca credentials. If set, that strategy uses its own
# (paper) account, which is the strongest possible isolation between bots.
# When not set, all strategies share ALPACA_API_KEY / ALPACA_SECRET_KEY and
# isolation is provided by the strategy-level ledger in core/portfolio.py.
ALPACA_KEYS_PER_STRATEGY = {
    s: (os.getenv(f"ALPACA_API_KEY_{s.upper()}") or ALPACA_API_KEY,
        os.getenv(f"ALPACA_SECRET_KEY_{s.upper()}") or ALPACA_SECRET_KEY)
    for s in ("stable", "risky1", "risky2")
}

#Polygon details for stocks and etf
POLYGON_API_KEY=os.getenv("POLYGON_API_KEY") #now will also be for crypto
POLYGON_API_KEY_RISKY1 =os.getenv("POLYGON_API_KEY_RISKY1", POLYGON_API_KEY)
POLYGON_API_KEY_RISKY2=os.getenv("POLYGON_API_KEY_RISKY2", POLYGON_API_KEY)

#KRAKEN for Crypto bot- Invalid due to geolocation restrictions.
#KRAKEN_API_KEY=os.getenv("KRAKEN_API_KEY")
#KRAKEN_SECRET_KEY=os.getenv("KRAKEN_SECRET_KEY")

#Allocation of capital, basically starting funds.
# These are NOTIONAL budgets per strategy inside one shared paper account. The
# strategy-level ledger (core/portfolio.py) enforces them: each strategy's
# equity = its own cash budget + its own positions marked to market.

CAPITAL ={
    "stable":1000.00,
    "risky1": 200.00,
    "risky2": 200.00,
}

#Max position size of each bot (dollars per symbol)
MAX_POSITION_SIZE ={
    "stable": (CAPITAL["stable"]*.2),
    "risky1": (CAPITAL["risky1"]*.25),
    "risky2":(CAPITAL["risky2"]*.25)
}

#Emergency Kill switches (drawdown from STRATEGY peak equity)
MAX_DRAWDOWN={
    "stable":(-.15), #15 percent loss= stop
    #30 percent loss = stop
    "risky1":(-.3),
    "risky2":(-.3)
}

# The kill switch only fires after this many CONSECUTIVE critical readings.
# A single bad equity value from the broker (or a paper-account reset) must not
# liquidate everything on its own. See ENGINEERING_AUDIT.md issue C-03.
KILL_SWITCH_CONFIRMATIONS = 2

# An equity reading that moves more than this fraction from the previous
# reading in one cycle is treated as suspect and not acted on until confirmed.
EQUITY_MAX_STEP_CHANGE = 0.5


#assets to invest in
#adjust stable assets to include whichever you want to invest in, it can't be populated like the risky1 because its supposed to be stable
STABLE_ASSETS = ["SPY", "QQQ", "AAPL", "MSFT", "DIA", "IWM","SAP","AMZN"]
#Risky1 will be modified by ML model so its dynamically changing
RISKY1_ASSETS=["NVDA","AMD","TSLA","META"]
#risky2 is in crypto and i don't reallyy know what to do for making it "risky", the X is for polygon prefix of crypto
RISKY2_ASSETS = ["X:BTCUSD","X:ETHUSD","X:SOLUSD","X:AVAXUSD","X:LINKUSD","X:ADAUSD","X:XRPUSD","X:DOGEUSD"
]

STRATEGY_ASSETS = {
    "stable": STABLE_ASSETS,
    "risky1": RISKY1_ASSETS,
    "risky2": RISKY2_ASSETS,
}

#Retraining of model occurances in days
RETRAIN_INTERVAL_DAYS=3


FEATURE_FLAGS={
    "risky2_enabled":False #when false, risky2 model simply doesn't run. when true, risky2 model runs. meant because its still being built as of this moment and is wasting resources.
}


#adding per trade stop loss to prevent individual stocks from tanking the overall performance of the strategy.

STOP_LOSS_PER_TRADE={
    "stable":.05,#if a stock moves 5%...thats a problem
    "risky1": .08,#somewwhat volatile, should get some room
    "risky2":.12#crypto gets a heck of alot more room because it is well...insane
}

WARNING_DRAWDOWN={
    "stable":-.07,"risky1":-.15,"risky2":-.15 #half of the killswitch should be good for a warning (where it will start reducing how much of the position is held).
}
#ATR is the normal volatility for threshold scaling. if live atr exceeds limit the kill switch threshold wides to avoid being triggered by regular noise while if below threshold will tighten to protect capital and profit.
#ATR is average true range, which measures how the asset price changes on average per day. it checksvolatility by looking at the biggest high minus low, high minus previous close, low minus previous close, then gets average over 14 days.
# UNITS: these baselines are FRACTIONS OF PRICE (0.015 = 1.5% of price per day).
# The ATR produced by core/features.py is in price units (dollars), so callers
# must divide by price first (metrics.risk_manager.atr_as_fraction). See
# ENGINEERING_AUDIT.md issue H-01.
ATR_BASELINE={
    "stable":0.015,"risky1":.025,"risky2":.045#-1.5 daily atr can be normal for etfs, 2.5 is normal for regular stocks and 4.5 for crypto should be fine.
}

# Anti-churn (see ENGINEERING_AUDIT.md issue H-02).
# Discretionary (non-emergency) exits are only allowed once at least this many
# seconds have passed since the position was opened. Emergency exits (stop
# loss, portfolio critical, kill switch) ignore this.
MIN_HOLD_SECONDS = {
    "stable": 0,
    "risky1": 0,
    "risky2": 1800,
}
# Stable/risky1 trade on daily bars; the system may act at most once per
# symbol per bar, and never re-enter on the bar it just exited.
ONE_ACTION_PER_BAR = {
    "stable": True,
    "risky1": True,
    "risky2": False,
}

# Seconds to wait for a market order to fill before treating it as pending.
ORDER_FILL_TIMEOUT_SECONDS = 20

# Only top up an existing position when it is below this fraction of its
# target size. The exported trade log showed 143 of stable's 239 BUY rows were
# sub-$5 "dust" top-ups fired every cycle as prices drifted (audit issue H-02).
MIN_TOPUP_FRACTION = 0.10


#working on new dashboard below Week 7
STRATEGY_COLORS={ ## gained from LLM, wasn't sure how to implement this on my own
    "stable": "#4A90D9",
    "risky1": "#E67E22",
    "risky2": "#9B59B6"
}

# A full cycle takes 4 tickers x 20 s + 60 s (risky1) up to 8 x 20 s + 60 s
# (stable), so the old value of 60 s marked healthy bots as DISCONNECTED.
HEARTBEAT_STALE_SECONDS = 600
