import time
import numpy as np
from datetime import datetime
#from core.config import CAPITAL, MAX_POSITION_SIZE, RISKY2_ASSETS
from core.config import RISKY2_ASSETS, MAX_POSITION_SIZE, CAPITAL, POLYGON_API_KEY_RISKY2

from core.logger import log_trade
from core.features import build_features
from data.polygon_fetcher import get_latest_bar
from data.sentiment_fetcher import add_sentiment_to_df
from models.rl_train import load_rl_model
from models.rl_environment import TradingEnvironment
from data.discord_notifier import send_heartbeat, send_alert
from core.logger import log_heartbeat
#from datetime import datetime
from paper_trading.alpaca_paper import get_api#, is_market_open
strategy="risky2"
model=load_rl_model()


def get_current_position(api,ticker):
    """" This will return the current position size of a ticker in dollars and if tehre is no position it will return a 0"""
    try:
        position=api.get_position(ticker)
        positions_value=float(position.market_value)
        return positions_value
    except:
        return 0.0
    

def trade_ticker(api,ticker):
    """"
    The PPO model will decide whether to hold or buy or sell for each crypto ticker. This will specificilaly build a live observation from the most recent price data and feed it to the agent
    """
    if model is None:
        print(f" No RL model has been loaded and {ticker} will be skipped")
        return;
    df = get_latest_bar(ticker, api_key=POLYGON_API_KEY_RISKY2)
    if df is None or df.empty:
        print(f"No data is in {ticker}, skipping")
        return
    # build the features in df
    df=build_features(df)
    df=add_sentiment_to_df(df,ticker)
    df=df.dropna()
    if len(df)==0:
        print(f"There are no featured rows for {ticker} after dropna occured, skipping")
        return;
    env=TradingEnvironment(df=df,initial_capital=CAPITAL[strategy],max_position_pct=MAX_POSITION_SIZE[strategy]/CAPITAL[strategy])
    obs,  _=env.reset()

    #nwo to get the action from model
    action, _ = model.predict(obs, deterministic=False)
    price=float(df['close'].iloc[-1])
    current_pos=get_current_position(api,ticker)
    max_pos=MAX_POSITION_SIZE[strategy]
    #actions to be executed

    #buy
    if action==1 and current_pos==0:
        qty=round(max_pos/price,4)
        if qty*price>= 1.0:
            api.submit_order(symbol=ticker,qty=qty,side='buy',type='market',time_in_force='gtc')
            log_trade(strategy,ticker,"BUY",price,qty,reason="The PPO Agent chose to BUY")
            print(f"BUY {qty} of {ticker} @ ${price}")
        
    #sell
    elif action==2 and current_pos>0:
        api.close_position(ticker)
        log_trade(strategy,ticker,"SELL",price,current_pos,reason="PPO agent is choosing to sell")
        print(f"SELL {ticker} @ ${price}")

    else:
        print(f"HOLD {ticker} due to agent action {action}")

def run():
    """
    This is the main loop for risky2 RL bot. Since crypto is 24/7 there is no need for market hours check
    """
    api=get_api(strategy)
    print("Risky2 RL bot has started now.....")
    last_trade_time=datetime.now()
    last_sync_time=datetime.now()
    while True:
        try:
            for ticker in RISKY2_ASSETS:
                try:
                    trade_ticker(api,ticker)
                except Exception as e:
                    print(f"There was an error in trading {ticker}: {e}")
                time.sleep(20)
            print(f"Cycle was completed at {datetime.now()}, sleeping for 60 seconds (1min)")
            last_sync_time = datetime.now()
            log_heartbeat(strategy, "RUNNING")
            account = api.get_account()
            send_heartbeat(
                bot_name="risky2",
                is_alive=True,
                portfolio_value=float(account.portfolio_value),
                last_trade_time=str(last_trade_time),
                last_sync_time=str(last_sync_time),
                extra_info="Crypto cycle complete."
            )
            time.sleep(60)
        except KeyboardInterrupt:
            print("Risky2 bot was manually stopped by user")
            break;

        except Exception as e:
            print(f"Unexpected error occured: {e}\n restarting in 60 seconds")
            time.sleep(60)

if __name__=="__main__":
    run()
