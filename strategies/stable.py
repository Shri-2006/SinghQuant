"""
Stable strategy: ETF / blue-chip XGBoost, daily bars, market hours only.

The trading loop lives in strategies/common.py (shared with risky1/risky2).
This module only declares what is specific to "stable".
"""
from core.config import STABLE_ASSETS, POLYGON_API_KEY
from models.train import ModelHandle
from models.regime_detector import get_regime_for_strategy
from data.discord_notifier import send_heartbeat, send_alert
from strategies.common import StrategyContext, xgb_decider, make_frame_fetcher, effective_vix, run_forever

strategy = "stable"
model_handle = ModelHandle("stable_model.pkl")


def build_context(**overrides):
    ctx = StrategyContext(
        name=strategy,
        assets=list(STABLE_ASSETS),
        decide=xgb_decider(model_handle),
        fetch_frame=make_frame_fetcher(POLYGON_API_KEY),
        uses_market_hours=True,
        momentum_exit=False,
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
    """Main loop for the stable strategy."""
    run_forever(build_context())


if __name__ == "__main__":
    run()
