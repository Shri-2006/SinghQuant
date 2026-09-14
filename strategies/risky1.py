"""
Risky1 strategy: momentum stocks with XGBoost plus a momentum exit.

The trading loop lives in strategies/common.py. The momentum exit
(momentum_5 < 0) is now part of the shared state machine: it cannot fire on
the bar the position was opened on, and a BUY is refused while the exit
condition is already true (this was the 2026-08-27 TSLA oscillation).
"""
from core.config import RISKY1_ASSETS, POLYGON_API_KEY_RISKY1
from models.train import ModelHandle
from models.regime_detector import get_regime_for_strategy
from data.discord_notifier import send_heartbeat, send_alert
from strategies.common import StrategyContext, xgb_decider, make_frame_fetcher, effective_vix, run_forever

strategy = "risky1"
model_handle = ModelHandle("risky1_model.pkl")


def build_context(**overrides):
    ctx = StrategyContext(
        name=strategy,
        assets=list(RISKY1_ASSETS),
        decide=xgb_decider(model_handle),
        fetch_frame=make_frame_fetcher(POLYGON_API_KEY_RISKY1),
        uses_market_hours=True,
        momentum_exit=True,
        regime_gate=True,
        time_in_force="day",
        macro_context=effective_vix,
        regime_ok=get_regime_for_strategy,
        notify=send_heartbeat,
        alert=send_alert,
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def run():
    """Main loop for the risky1 strategy."""
    run_forever(build_context())


if __name__ == "__main__":
    run()
