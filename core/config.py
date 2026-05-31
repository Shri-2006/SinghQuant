import os
from dotenv import load_dotenv

load_dotenv()

#discord notification setup
DISCORD_WEBHOOK_URL=os.getenv("DISCORD_WEBHOOK_URL","")

#Paper trading mode, false means its actual money, true when paper trading
PAPER_MODE={
    "stable":True,#flip to false to make stable live
    "risky1":True,#flip to false to make risky1live
    "risky2":True#flip to false to make risky2 live, like will take more time since its RL experimental
}

#Alpaca Details
ALPACA_API_KEY=os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY=os.getenv("ALPACA_SECRET_KEY")
#if PAPER_MODE=true then this is active, else its to live url
ALPACA_PAPER_URL ="https://paper-api.alpaca.markets"
ALPACA_LIVE_URL="https://api.alpaca.markets"




#Polygon details for stocks and etf
POLYGON_API_KEY=os.getenv("POLYGON_API_KEY") #now will also be for crypto


#KRAKEN for Crypto bot- Invalid due to geolocation restrictions.
#KRAKEN_API_KEY=os.getenv("KRAKEN_API_KEY")
#KRAKEN_SECRET_KEY=os.getenv("KRAKEN_SECRET_KEY")

#Allocation of capital, basically starting funds

CAPITAL ={
    "stable":1000.00,
    "risky1": 200.00,
    "risky2": 200.00,
}

#Max position size of each bot
MAX_POSITION_SIZE ={
    "stable": (CAPITAL["stable"]*.2),
    "risky1": (CAPITAL["risky1"]*.25),
    "risky2":(CAPITAL["risky2"]*.25)
}

#Emergency Kill switches
MAX_DRAWDOWN={
    "stable":(-.15), #15 percent loss= stop
    #30 percent loss = stop
    "risky1":(-.3), 
    "risky2":(-.3)
}





#assets to invest in
#adjust stable assets to include whichever you want to invest in, it can't be populated like the risky1 because its supposed to be stable
STABLE_ASSETS = ["SPY", "QQQ", "AAPL", "MSFT", "DIA", "IWM","SAP"]
#Risky1 will be modified by ML model so its dynamically changing
RISKY1_ASSETS=[]
#risky2 is in crypto and i don't reallyy know what to do for making it "risky", the X is for polygon prefix of crypto
RISKY2_ASSETS = ["X:BTCUSD","X:ETHUSD","X:SOLUSD","X:AVAXUSD","X:LINKUSD","X:ADAUSD","X:XRPUSD","X:DOGEUSD"
]

#Retraining of model occurances in days
RETRAIN_INTERVAL_DAYS=3


FEATURE_FLAGS={
    "risky2_enabled":True #when false, risky2 model simply doesn't run. when true, risky2 model runs. meant because its still being built as of this moment and is wasting resources.
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
ATR_BASELINE={
    "stable":0.015,"risky1":.025,"risky2":.045#-1.5 daily atr can be normal for etfs, 2.5 is normal for regular stocks and 4.5 for crypto should be fine.
}


#working on new dashboard below Week 7
STRATEGY_COLORS={ ## gained from LLM, wasn't sure how to implement this on my own
    "stable": "#4A90D9",
    "risky1": "#E67E22",
    "risky2": "#9B59B6"
}

HEARTBEAT_STALE_SECONDS = 60
