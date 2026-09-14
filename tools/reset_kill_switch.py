"""
Clears a strategy's kill-switch halt after a human has reviewed why it fired.

    python tools/reset_kill_switch.py stable
    python tools/reset_kill_switch.py risky1 --reset-peak

--reset-peak also sets the peak equity to the current strategy equity so the
strategy does not immediately re-trip on the same drawdown. Without it the
peak is kept (the safer default: the drawdown must actually recover).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import portfolio  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("strategy", choices=["stable", "risky1", "risky2"])
    p.add_argument("--reset-peak", action="store_true")
    a = p.parse_args()
    state = portfolio.get_strategy_state(a.strategy)
    print(f"{a.strategy}: halted={state['halted']} reason={state['halted_reason']} peak={state['peak_equity']:.2f} last_equity={state['last_equity']}")
    fields = {"halted": False, "halted_reason": None, "critical_streak": 0}
    if a.reset_peak and state["last_equity"]:
        fields["peak_equity"] = state["last_equity"]
    portfolio.update_strategy_state(a.strategy, **fields)
    print(f"{a.strategy}: halt cleared" + (" and peak reset" if a.reset_peak else ""))


if __name__ == "__main__":
    main()
