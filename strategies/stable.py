## This will fetch latest price data from polygon, build_features(), check regime (if volatile it will not do it), load the ml model to predict buy/sell/hold. If buy, check to avoid MAX_Position_size, place grid of buy orders around current price. If sell, it will close the position, and it will check the kill switch where if drawdown supersedes 15% it will stop everything to prevent extreme loss. It will also log every trade to SQLite.
#stable.py
#import os
import time
#import pandas as pd
from datetime import datetime
#from core.config import (STABLE_ASSETS,MAX_POSITION_SIZE,MAX_DRAWDOWN,CAPITAL,PAPER_MODE,ALPACA_PAPER_URL,ALPACA_LIVE_URL )
from core.config import (STABLE_ASSETS, MAX_POSITION_SIZE, CAPITAL)
#from core.logger import log_trade
from core.features import build_features
from data.polygon_fetcher import get_latest_bar
from data.sentiment_fetcher import add_sentiment_to_df
from models.regime_detector import get_regime_for_strategy
from models.train import load_model
#from metrics.risk import compute_all_metrics
#import alpaca_trade_api as tradeapi
from paper_trading.alpaca_paper import get_api, get_sleep_duration
from metrics.risk_manager import get_position_size, should_close_position, get_portfolio_risk_level
from core.logger import log_trade, log_heartbeat
from metrics.equity_curve_filter import get_trading_state
from models.regime_detector import get_vix
from data.macro_fetcher import get_macro_signal
from data.discord_notifier import send_heartbeat


strategy="stable"
_peak_equity = {strategy: CAPITAL[strategy]} #get the highest equity
#vix = get_vix()
#loading model only once to keep same model for each trade in single session, save memory (analysis of algorithms thank you) and to improve speed to avoid reinitializing model from disk each trade
model=load_model("stable_model.pkl")



def check_kill_switch(api):
    """
    Checks portfolio risk level using the peak equity. If critical it will trip kill switch, if warning reduce position size by about half
    """
    global _peak_equity
    account=api.get_account()
    equity=float(account._raw['portfolio_value'])

    #update peak equity for the profit
    _peak_equity[strategy]=max(_peak_equity[strategy],equity)
    peak=_peak_equity[strategy]

    #drawdown using peak value
    drawdown=(equity-peak)/peak
    risk_level=get_portfolio_risk_level(strategy,equity)

    if risk_level=="critical":
        print(f"KILL SWITCH HAS BEEN FIRED:     DRAWDOWN{drawdown:.2%} from the peak of ${peak:.2f}")
        api.close_all_positions()
        log_trade(strategy,"ALL","KILL_SWITCH",0,0,pnl=drawdown,reason=f"Drawdown {drawdown:.2%} from the peak")
        return True
    elif risk_level=="warning":
        print(f"WARNINGG, drawdown {drawdown:.2%}, bot to reduce position size")
    return False






def get_current_position(api,ticker):
    """"
    this wil return the current posisition size in dollars for ticker, if doesn't have one, it will return 0
    """
    try:
        position=api.get_position(ticker)
        return float(position.market_value)
    except:
        return 0.0#no position held
    



def trade_ticker(api, ticker,multiplier=1.0,vix=None):

    """
    Main trading code that will run once per ticker in trading cycle
    """
    account = api.get_account()
    equity = float(account._raw['portfolio_value'])

    # Get price data first — needed for stop loss and ML
    df = get_latest_bar(ticker)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping")
        return

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker)
    if len(df) == 0:
        return

    current_price = float(df['close'].iloc[-1])
    current_atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else None
  

    # Risk-adjusted position size
    base_size = MAX_POSITION_SIZE[strategy]*multiplier
    max_pos = get_position_size(strategy, equity, base_size,current_atr)
    if max_pos == 0:
        print(f"Portfolio critical — skipping {ticker}")
        return

    # Check stop loss on existing position
    current_pos = get_current_position(api, ticker)
    if current_pos > 0:
        try:
            position = api.get_position(ticker)
            entry_price = float(position.avg_entry_price)
            if should_close_position(strategy, entry_price, current_price, equity, current_atr):
                api.close_position(ticker)
                realized_pnl = (current_price - entry_price) / entry_price * current_pos
                log_trade(strategy, ticker, "SELL", current_price, current_pos,
                    pnl=realized_pnl, reason="Stop loss or portfolio risk triggered")
                print(f"Closed {ticker} — risk management triggered")
                return
        except:
            pass

    # Check regime
    if not get_regime_for_strategy(df, strategy,vix):
        print(f"{ticker} regime unfavorable, sitting out")
        return

    # ML prediction
    feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
    X = df[feature_cols].values
    prediction = model.predict(X[-1].reshape(1,-1))[0]

    # Execute trade
    if prediction == 1 and current_pos < max_pos:
        qty = round((max_pos - current_pos) / current_price, 4)
        if qty > 0:
            order_value = qty * current_price
            if order_value < 1.0:
                print(f"Skipping {ticker} — order ${order_value:.2f} below minimum")
                return
            api.submit_order(symbol=ticker, qty=qty, side='buy',
                           type='market', time_in_force='day')
            log_trade(strategy, ticker, "BUY", current_price, qty,
                     reason="ML signals BUY, regime favorable")
            print(f"Buy {qty} {ticker} @ ${current_price}")
    elif prediction == 0 and current_pos > 0:
        try:
            position = api.get_position(ticker)
            entry_price = float(position.avg_entry_price)
            realized_pnl = (current_price - entry_price) / entry_price * current_pos
        except:
            realized_pnl = None
        api.close_position(ticker)
        log_trade(strategy, ticker, "SELL", current_price, current_pos,
            pnl=realized_pnl, reason="ML signals SELL")
        print(f"SELL {ticker} @ ${current_price}")
    else:
        print(f"HOLD {ticker}")




def run():
    """ 
    This is the main loop for the stable strategy, and it runs continusly. it will check each ticker every 60 seconds, and runs continuosly. It iwll be called from run.py when the system boots up
    """
    api = get_api("stable")
    print("Stable bot started now...")
    last_trade_time = datetime.now()
    last_sync_time = datetime.now()   

    while True:
        try:
            if check_kill_switch(api):
                print("Kill switch has stopped stable bot")
                from data.discord_notifier import send_alert

                account = api.get_account()
                send_alert(
                    bot_name="stable",           
                    alert_type="KILL SWITCH",
                    message="Kill switch triggered — all positions closed. Bot halted.",
                    portfolio_value=float(account.portfolio_value)
                )
                break

            # Check if market is open before trading
            sleep_secs=get_sleep_duration(strategy)
            if sleep_secs>0:
                print(f"Market is closed - slepeing for {sleep_secs//3600} hours {(sleep_secs%3600//60)} minutes")
                time.sleep(sleep_secs)
                continue
            
            state,multiplier=get_trading_state(strategy)



            if state =="HALT":
                print(f"Equity curve halt is currently active for {strategy} - skipping cycle")
                log_heartbeat(strategy,"PAUSED")
                time.sleep(60)
                continue
            vix = get_vix()
            # Macro signal from SearXNG — raises effective VIX if alarming headlines detected
            macro  = get_macro_signal()
            if macro == "DANGER":
                effective_vix = max(vix or 0, 35)
            elif macro == "CAUTION":
                effective_vix = max(vix or 0, 22)
            else:
               effective_vix = vix or 0
            # Pass effective_vix instead of vix going forward
            vix = effective_vix if effective_vix > 0 else None

            for ticker in STABLE_ASSETS:
                try:
                    trade_ticker(api,ticker,multiplier,vix)
                    last_trade_time = datetime.now() 
                except Exception as e:
                    print(f"error trading this asset: {ticker}: {e}")
                    continue
                time.sleep(20)
            last_sync_time = datetime.now() 
            print(f"cycle has complteded at {datetime.now()}-time to sleep for 1 min")
            log_heartbeat(strategy, "RUNNING")
            account=api.get_account()
            send_heartbeat(
                bot_name="stable",
                is_alive=True,
                portfolio_value=float(account.portfolio_value),
                last_trade_time=str(last_trade_time),
                last_sync_time=str(last_sync_time),
                extra_info="Cycle complete."
            )

            time.sleep(60)
        except KeyboardInterrupt:
            print("Stable bot was manually stopped by user")
            break
        except Exception as e:
            print(f"there was an unexpected error : {e}, will restart cycle in 60 seconds")
            time.sleep(60)
        # At top of file




        
    
if __name__=="__main__":
    run();
























































#Alpaca connection (paper vs live)
# if PAPER_MODE["stable"]:
#     BASE_URL=ALPACA_PAPER_URL
# else:
#     BASE_URL=ALPACA_LIVE_URL

# def get_api():
#     """
#     will confirm and return the alpaca api conenctions.
#     """
#     from core.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
#     return tradeapi.REST(ALPACA_API_KEY,ALPACA_SECRET_KEY,BASE_URL)

# def check_kill_switch(api):
#     """
#     This will checkk if the loss has exceeded 15% overall from the original capital. if it has, it closes alll positions and stops trading immediately. It will also return true if kill switchh is fired, else false if safe
#     """
#     account=api.get_account()
#     equity=float(account._raw['portfolio_value'])
#     start_cash=CAPITAL["stable"]
#     drawdown=(equity-start_cash)/(start_cash);

#     if drawdown<= MAX_DRAWDOWN["stable"]:
#         print(f"KILL SWITCH IS BEING FIRED STOP NOW - drawdown is {drawdown:.2%} and has exceeded the limit")
#         api.close_all_positions()
#         log_trade("stable","ALL","KILL_SWITCH",0,0,pnl=drawdown,reason=(f"Drawdown is {drawdown:.2%}"))
#         return True
#     return False



# #Main trading logic btw, take a look :D
# def trade_ticker(api,ticker):
#     """
#     This will run a full trading cycle per ticker, and is called only once per ticker per trading cycle in the function run()
    
#     """
#     #get price data
#     df=get_latest_bar(ticker)
#     if df is None or df.empty:
#         print(f"There is no data for {ticker}, will skip")
#         return
    
#     #now the features
#     df=build_features(df)
#     df=add_sentiment_to_df(df,ticker)

#     #now checks regime
#     if not get_regime_for_strategy(df,"stable"):
#         print(f"{ticker}  regime is in a unfavorable spot, sitting out this time")
#         return
    
#     #now gets the ML prediction
#     feature_cols=[c for c in df.columns if c not in ['open','high','low','close','volume']]
#     X=df[feature_cols].values
#     prediction=model.predict(X[-1].reshape(1,-1))[0]
#     price=float(df['close'].iloc[-1])
    
#     #now based on the predicition it will choose what trade to do
#     current_pos=get_current_position(api,ticker)
#     max_pos=MAX_POSITION_SIZE["stable"]
#     if prediction==1 and current_pos<max_pos:
#         #if predicition is 1 and pos is less than max, buy
#         qty = round((max_pos - current_pos) / price, 4)
#         if qty > 0:
#             order_value = qty * price
#             if order_value < 1.0:
#                 print(f"Skipping {ticker} — order value ${order_value:.2f} below $1 minimum")
#                 return
#             api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='day')
#             log_trade("stable", ticker, "BUY", price, qty, reason="ML signals to Buy and the regime is favorable")
#             print(f"Buy {qty}{ticker} @ ${price}")
#     elif prediction ==0 and current_pos>0:
#         #if prediciton is 0 and pos is greater than 0, SELL NOW
#         api.close_position(ticker)
#         log_trade("stable",ticker,"SELL",price,current_pos,reason="ML signals to SELL now")
#         print(f"SELL{ticker}  @  ${price}")
#     else:
#         #in other cases simply hold the pos
#         print(f"HOLD{ticker}, no action needed rn")

