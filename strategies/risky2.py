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

strategy         = "risky2"
MIN_HOLD_SECONDS = 1800  # 30 min minimum hold to prevent churn

# Load one model per ticker at startup
print("Loading per-ticker RL models...")
MODELS = {}
for ticker in RISKY2_ASSETS:
    MODELS[ticker] = load_rl_model(ticker)

# Track when each position was opened
_position_open_time = {}


def _to_alpaca_symbol(ticker):
    """
    Converts Polygon crypto format to Alpaca format.
    X:BTCUSD → BTCUSD (Alpaca stores without slash)
    """
    if ticker.startswith("X:"):
        return ticker[2:]
    return ticker


def get_current_position(api, ticker):
    """Returns current position size in dollars, 0 if none."""
    try:
        return float(api.get_position(_to_alpaca_symbol(ticker)).market_value)
    except:
        return 0.0


def trade_ticker(api, ticker):
    """
    PPO model decides HOLD/BUY/SELL for a crypto ticker.
    Uses ticker-specific model for better accuracy.
    Enforces minimum hold time and cash check to prevent churn.
    """
    model = MODELS.get(ticker)
    if model is None:
        print(f"No model for {ticker} — skipping")
        return

    df = get_latest_bar(ticker, api_key=POLYGON_API_KEY_RISKY2)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping")
        return

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker)
    df = df.dropna()
    if len(df) == 0:
        print(f"No feature rows for {ticker}, skipping")
        return

    env = TradingEnvironment(
        df=df,
        initial_capital=CAPITAL[strategy],
        max_position_pct=MAX_POSITION_SIZE[strategy] / CAPITAL[strategy]
    )
    obs, _ = env.reset()

    action, _   = model.predict(obs, deterministic=False)
    price       = float(df['close'].iloc[-1])
    current_pos = get_current_position(api, ticker)
    max_pos     = MAX_POSITION_SIZE[strategy]
    alpaca_sym  = _to_alpaca_symbol(ticker)

    if action == 1 and current_pos == 0:
        qty         = round(max_pos / price, 4)
        order_value = qty * price

        # Guard against negative cash — never buy without sufficient funds
        try:
            available_cash = float(api.get_account().cash)
            if available_cash < order_value:
                print(f"Skipping {ticker} — insufficient cash (${available_cash:.2f} < ${order_value:.2f})")
                return
        except:
            pass

        if order_value >= 1.0:
            api.submit_order(symbol=alpaca_sym, qty=qty, side='buy',
                             type='market', time_in_force='gtc')
            log_trade(strategy, ticker, "BUY", price, qty,
                      reason="PPO agent chose BUY")
            _position_open_time[ticker] = datetime.now()
            print(f"BUY {qty} {ticker} @ ${price}")

    elif action == 2 and current_pos > 0:
        # Enforce minimum hold time to prevent churn
        open_time = _position_open_time.get(ticker)
        if open_time:
            held_seconds = (datetime.now() - open_time).total_seconds()
            if held_seconds < MIN_HOLD_SECONDS:
                print(f"HOLD {ticker} — min hold not met ({held_seconds:.0f}s/{MIN_HOLD_SECONDS}s)")
                return
        api.close_position(alpaca_sym)
        log_trade(strategy, ticker, "SELL", price, current_pos,
                  reason="PPO agent chose SELL")
        _position_open_time.pop(ticker, None)
        print(f"SELL {ticker} @ ${price}")

    else:
        print(f"HOLD {ticker} due to agent action {action}")


def run():
    """
    Main loop for risky2 RL bot.
    Crypto trades 24/7 so no market hours check needed.
    """
    api             = get_api(strategy)
    last_trade_time = datetime.now()
    last_sync_time  = datetime.now()
    print("Risky2 RL bot has started now.....")

    while True:
        try:
            for ticker in RISKY2_ASSETS:
                try:
                    trade_ticker(api, ticker)
                    last_trade_time = datetime.now()
                except Exception as e:
                    print(f"Error trading {ticker}: {e}")
                time.sleep(20)

            print(f"Cycle completed at {datetime.now()}, sleeping 60 seconds")
            last_sync_time = datetime.now()
            log_heartbeat(strategy, "RUNNING")

            account = api.get_account()
            send_heartbeat(
                bot_name="risky2", is_alive=True,
                portfolio_value=float(account.portfolio_value),
                last_trade_time=str(last_trade_time),
                last_sync_time=str(last_sync_time),
                extra_info="Crypto cycle complete."
            )
            time.sleep(60)

        except KeyboardInterrupt:
            print("Risky2 bot manually stopped")
            break
        except Exception as e:
            print(f"Unexpected error: {e}\nRestarting in 60 seconds")
            time.sleep(60)


if __name__ == "__main__":
    run()