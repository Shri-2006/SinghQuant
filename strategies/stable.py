import time
from datetime import datetime
from core.config import STABLE_ASSETS, MAX_POSITION_SIZE, CAPITAL, POLYGON_API_KEY
from core.features import build_features
from data.polygon_fetcher import get_latest_bar
from data.sentiment_fetcher import add_sentiment_to_df
from models.regime_detector import get_regime_for_strategy, get_vix
from models.train import load_model
from paper_trading.alpaca_paper import get_api, get_sleep_duration
from metrics.risk_manager import get_position_size, should_close_position, get_portfolio_risk_level
from core.logger import log_trade, log_heartbeat
from metrics.equity_curve_filter import get_trading_state
from data.macro_fetcher import get_macro_signal
from data.discord_notifier import send_heartbeat, send_alert

strategy     = "stable"
_peak_equity = {strategy: CAPITAL[strategy]}
model        = load_model("stable_model.pkl")


def check_kill_switch(api):
    """
    Checks portfolio risk using peak equity.
    Fires kill switch if critical, warns if approaching threshold.
    """
    global _peak_equity
    account  = api.get_account()
    equity   = float(account._raw['portfolio_value'])

    _peak_equity[strategy] = max(_peak_equity[strategy], equity)
    peak     = _peak_equity[strategy]
    drawdown = (equity - peak) / peak
    risk     = get_portfolio_risk_level(strategy, equity)

    if risk == "critical":
        print(f"KILL SWITCH FIRED — drawdown {drawdown:.2%} from peak ${peak:.2f}")
        api.close_all_positions()
        log_trade(strategy, "ALL", "KILL_SWITCH", 0, 0,
                  pnl=drawdown, reason=f"Drawdown {drawdown:.2%} from peak")
        send_alert(
            bot_name=strategy,
            alert_type="KILL SWITCH",
            message="Kill switch triggered — all positions closed. Bot halted.",
            portfolio_value=equity
        )
        return True
    elif risk == "warning":
        print(f"WARNING — drawdown {drawdown:.2%}, reducing position size")
    return False


def get_current_position(api, ticker):
    """Returns current position size in dollars, 0 if none."""
    try:
        return float(api.get_position(ticker).market_value)
    except:
        return 0.0


def trade_ticker(api, ticker, multiplier=1.0, vix=None):
    """
    Full trading cycle for one ticker.
    Checks stop loss, regime, ML prediction then executes BUY/SELL/HOLD.
    """
    account = api.get_account()
    equity  = float(account._raw['portfolio_value'])

    df = get_latest_bar(ticker, api_key=POLYGON_API_KEY)
    if df is None or df.empty:
        print(f"No data for {ticker}, skipping")
        return

    df = build_features(df)
    df = add_sentiment_to_df(df, ticker)
    if len(df) == 0:
        return

    current_price = float(df['close'].iloc[-1])
    current_atr   = float(df['atr'].iloc[-1]) if 'atr' in df.columns else None

    # Risk-adjusted position size
    base_size = MAX_POSITION_SIZE[strategy] * multiplier
    max_pos   = get_position_size(strategy, equity, base_size, current_atr)
    if max_pos == 0:
        print(f"Portfolio critical — skipping {ticker}")
        return

    # Stop loss check
    current_pos = get_current_position(api, ticker)
    if current_pos > 0:
        try:
            entry_price = float(api.get_position(ticker).avg_entry_price)
            if should_close_position(strategy, entry_price, current_price, equity, current_atr):
                realized_pnl = (current_price - entry_price) / entry_price * current_pos
                api.close_position(ticker)
                log_trade(strategy, ticker, "SELL", current_price, current_pos,
                          pnl=realized_pnl, reason="Stop loss or portfolio risk triggered")
                print(f"Closed {ticker} — risk management triggered")
                return
        except:
            pass

    # Regime check
    if not get_regime_for_strategy(df, strategy, vix):
        print(f"{ticker} regime unfavorable, sitting out")
        return

    # ML prediction
    feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
    prediction   = model.predict(df[feature_cols].values[-1].reshape(1, -1))[0]

    if prediction == 1 and current_pos < max_pos:
        qty         = round((max_pos - current_pos) / current_price, 4)
        order_value = qty * current_price
        if qty > 0 and order_value >= 1.0:
            api.submit_order(symbol=ticker, qty=qty, side='buy',
                             type='market', time_in_force='day')
            log_trade(strategy, ticker, "BUY", current_price, qty,
                      reason="ML signals BUY, regime favorable")
            print(f"Buy {qty} {ticker} @ ${current_price}")
        else:
            print(f"Skipping {ticker} — order ${order_value:.2f} below minimum")

    elif prediction == 0 and current_pos > 0:
        try:
            entry_price  = float(api.get_position(ticker).avg_entry_price)
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
    Main loop for stable grid+ML strategy.
    Runs continuously, checks market hours, trades each cycle.
    """
    api             = get_api(strategy)
    last_trade_time = datetime.now()
    last_sync_time  = datetime.now()
    print("Stable bot started now...")

    while True:
        try:
            if check_kill_switch(api):
                print("Kill switch has stopped stable bot")
                break

            sleep_secs = get_sleep_duration(strategy)
            if sleep_secs > 0:
                print(f"Market is closed - sleeping for {sleep_secs//3600}h {(sleep_secs%3600)//60}m")
                log_heartbeat(strategy, "PAUSED")
                try:
                    account = api.get_account()
                    send_heartbeat(
                        bot_name=strategy, is_alive=True,
                        portfolio_value=float(account.portfolio_value),
                        last_trade_time="Market closed",
                        last_sync_time=str(datetime.now()),
                        extra_info=f"Sleeping {sleep_secs//3600}h — market closed"
                    )
                except:
                    pass
                time.sleep(sleep_secs)
                continue

            state, multiplier = get_trading_state(strategy)
            if state == "HALT":
                print(f"Equity curve halt active for {strategy} — skipping cycle")
                log_heartbeat(strategy, "PAUSED")
                time.sleep(60)
                continue

            vix   = get_vix()
            macro = get_macro_signal()
            if macro == "DANGER":
                vix = max(vix or 0, 35)
            elif macro == "CAUTION":
                vix = max(vix or 0, 22)
            else:
                vix = vix or 0
            vix = vix if vix > 0 else None

            for ticker in STABLE_ASSETS:
                try:
                    trade_ticker(api, ticker, multiplier, vix)
                    last_trade_time = datetime.now()
                except Exception as e:
                    print(f"Error trading {ticker}: {e}")
                time.sleep(20)

            last_sync_time = datetime.now()
            print(f"Cycle completed at {datetime.now()} — sleeping 1 min")
            log_heartbeat(strategy, "RUNNING")

            account = api.get_account()
            send_heartbeat(
                bot_name=strategy, is_alive=True,
                portfolio_value=float(account.portfolio_value),
                last_trade_time=str(last_trade_time),
                last_sync_time=str(last_sync_time),
                extra_info="Cycle complete."
            )
            time.sleep(60)

        except KeyboardInterrupt:
            print("Stable bot manually stopped")
            break
        except Exception as e:
            print(f"Unexpected error: {e} — restarting in 60 seconds")
            time.sleep(60)


if __name__ == "__main__":
    run()