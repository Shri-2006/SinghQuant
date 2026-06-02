import time
import numpy as np
from datetime import datetime
from core.config import RISKY2_ASSETS, MAX_POSITION_SIZE, CAPITAL, POLYGON_API_KEY_RISKY2
from core.logger import log_trade, log_heartbeat
from core.features import build_features
from data.polygon_fetcher import get_latest_bar
from data.sentiment_fetcher import add_sentiment_to_df
from models.rl_train import load_rl_model
from models.rl_environment import TradingEnvironment
from data.discord_notifier import send_heartbeat, send_alert
from paper_trading.alpaca_paper import get_api

strategy = "risky2"
model = load_rl_model()

def _to_alpaca_symbol(ticker):
    """
    Converts Polygon crypto format to Alpaca format.
    X:BTCUSD → BTC/USD
    Regular stocks pass through unchanged.
    """
    if ticker.startswith("X:"):
        base = ticker[2:]  # remove X:
        # insert slash before USD e.g. BTCUSD → BTC/USD
        if base.endswith("USD"):
            return base[:-3] + "/USD"
    return ticker

def get_current_position(api, ticker):
    """
    Returns current position size in dollars.
    Returns 0.0 if no position held.
    """
    try:
        alpaca_symbol = _to_alpaca_symbol(ticker)
        position = api.get_position(alpaca_symbol)
        return float(position.market_value)
    except:
        return 0.0

def trade_ticker(api, ticker):
    """
    PPO model decides HOLD/BUY/SELL for each crypto ticker.
    Builds live observation from latest price data and feeds to agent.
    """
    if model is None:
        print(f"No RL model loaded — skipping {ticker}")
        return

    df = get_latest_bar(ticker, api_key=POLYGON_API_KEY_RISKY2)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping")
        return

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker)
    df = df.dropna()
    if len(df) == 0:
        print(f"No feature rows for {ticker} after dropna, skipping")
        return

    env = TradingEnvironment(
        df=df,
        initial_capital=CAPITAL[strategy],
        max_position_pct=MAX_POSITION_SIZE[strategy] / CAPITAL[strategy]
    )
    obs, _ = env.reset()

    action, _ = model.predict(obs, deterministic=False)
    price = float(df['close'].iloc[-1])
    current_pos = get_current_position(api, ticker)
    max_pos = MAX_POSITION_SIZE[strategy]
    alpaca_symbol = _to_alpaca_symbol(ticker)

    if action == 1 and current_pos == 0:
        qty = round(max_pos / price, 4)
        if qty * price >= 1.0:
            api.submit_order(
                symbol=alpaca_symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            log_trade(strategy, ticker, "BUY", price, qty,
                     reason="PPO agent chose BUY")
            print(f"BUY {qty} {ticker} @ ${price}")

    elif action == 2 and current_pos > 0:
        api.close_position(alpaca_symbol)
        log_trade(strategy, ticker, "SELL", price, current_pos,
                 reason="PPO agent chose SELL")
        print(f"SELL {ticker} @ ${price}")

    else:
        print(f"HOLD {ticker} due to agent action {action}")


def run():
    """
    Main loop for risky2 RL bot.
    Crypto trades 24/7 so no market hours check needed.
    """
    api = get_api(strategy)
    print("Risky2 RL bot has started now.....")
    last_trade_time = datetime.now()
    last_sync_time = datetime.now()

    while True:
        try:
            for ticker in RISKY2_ASSETS:
                try:
                    trade_ticker(api, ticker)
                    last_trade_time = datetime.now()
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
            break

        except Exception as e:
            print(f"Unexpected error: {e}\nRestarting in 60 seconds")
            time.sleep(60)


if __name__ == "__main__":
    run()