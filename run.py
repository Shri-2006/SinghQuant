import time
import threading  # bots run side by side; each strategy loops forever in its own thread

from core.config import FEATURE_FLAGS, PAPER_MODE, ALLOW_LIVE_TRADING
from models.retrain import start_scheduler
from paper_trading.alpaca_paper import resolve_base_url


def preflight():
    """Refuses to start if any strategy would reach a live endpoint without opt-in."""
    for s in ("stable", "risky1", "risky2"):
        url = resolve_base_url(s)  # raises LiveTradingBlocked if misconfigured
        mode = "PAPER" if PAPER_MODE[s] else "LIVE (explicit opt-in)"
        print(f"  {s:7s} -> {mode}: {url}")
    if ALLOW_LIVE_TRADING:
        print("  WARNING: live-trading opt-in is set in the environment")


def main():
    """
    Starts the trading system:
      * preflight check that every strategy targets the paper endpoint
      * background retraining scheduler
      * one daemon thread per enabled strategy
      * keeps the main process alive
    """
    print("="*50)
    print(" SinghQuant Trading System Starting....")
    print("="*50)
    preflight()

    scheduler = start_scheduler()

    from strategies.stable import run as run_stable
    from strategies.risky1 import run as run_risky1

    threads = [
        threading.Thread(target=run_stable, daemon=True, name="stable"),
        threading.Thread(target=run_risky1, daemon=True, name="risky1"),
    ]
    if FEATURE_FLAGS["risky2_enabled"]:
        from strategies.risky2 import run as run_risky2
        threads.append(threading.Thread(target=run_risky2, daemon=True, name="risky2"))
    else:
        print("risky2 RL bot is disabled in config.py (FEATURE_FLAGS); enable it once models are trained")

    for t in threads:
        t.start()
        print(f"{t.name} strategy thread started")

    try:
        while True:
            time.sleep(60)
            dead = [t.name for t in threads if not t.is_alive()]
            if dead:
                print(f"WARNING: strategy thread(s) exited: {dead}")
    except KeyboardInterrupt:
        print("\nShutting down SinghQuant due to user input...")
        scheduler.shutdown()
        print("Scheduler has stopped. Goodbye for now!")


if __name__ == "__main__":
    main()
