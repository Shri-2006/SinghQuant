"""
Risky2 strategy: crypto with one PPO model per ticker, 24/7.

Audit changes (ENGINEERING_AUDIT.md issue H-04):
  * the observation is built from the LATEST bar with the strategy's real
    position state (was: a freshly reset env observing the OLDEST bar),
  * actions are taken deterministically (was: sampled),
  * the shared engine now applies the stop loss, strategy-level kill switch,
    throttle and minimum hold (30 min, persisted) to risky2 as well,
  * orders go through the ledger-aware execution layer with GTC time in force.
"""
from core.config import RISKY2_ASSETS, CAPITAL, POLYGON_API_KEY_RISKY2
from data.discord_notifier import send_heartbeat, send_alert
from strategies.common import StrategyContext, Decision, make_frame_fetcher, run_forever, artifact_status

strategy = "risky2"
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

_handles = {}


def get_handle(ticker):
    """Verified per-ticker PPO handle, created lazily so importing this module needs no torch."""
    if ticker not in _handles:
        from models.rl_train import rl_model_handle
        _handles[ticker] = rl_model_handle(ticker)
    return _handles[ticker]


def get_model(ticker):
    """The verified PPO for `ticker`, or None while its artifact does not validate (Phase 2 I-3)."""
    return get_handle(ticker).model


def ppo_decide(ctx, ticker, df, owned_qty, avg_entry, price):
    from models.rl_environment import build_live_observation
    model = get_model(ticker)
    if model is None:
        print(f"[{strategy}] No model for {ticker} — skipping")
        return None
    obs = build_live_observation(df, owned_qty * price, avg_entry, CAPITAL[strategy])
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    name = ACTION_NAMES.get(action, "HOLD")
    if name == "BUY" and owned_qty > 0:
        return Decision("HOLD", action, "PPO chose BUY but already in position")
    return Decision(name, action, f"PPO agent chose {name}")


def build_context(**overrides):
    ctx = StrategyContext(
        name=strategy,
        assets=list(RISKY2_ASSETS),
        decide=ppo_decide,
        fetch_frame=make_frame_fetcher(POLYGON_API_KEY_RISKY2),
        uses_market_hours=False,      # crypto trades 24/7
        momentum_exit=False,
        regime_gate=False,            # the original risky2 had no regime gate
        time_in_force="gtc",          # Alpaca crypto orders must be GTC
        check_account_cash=True,      # crypto needs settled account cash (restored from main, R-3)
        notify=send_heartbeat,
        alert=send_alert,
        model_status=lambda: artifact_status(*(get_handle(t) for t in RISKY2_ASSETS))(),
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def run():
    """Main loop for the risky2 RL strategy."""
    run_forever(build_context())


if __name__ == "__main__":
    run()
