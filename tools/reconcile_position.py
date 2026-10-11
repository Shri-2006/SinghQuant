"""
Manual reconciliation of ledger discrepancies (remediation R-1, owner decision D1-B).

The trading engine never writes off a position that has vanished from the
broker. It FLAGS the missing quantity, keeps the ledger as it was, values the
flagged quantity conservatively and alerts. A human resolves the flag here
after checking the broker (Alpaca dashboard / account activities):

    python tools/reconcile_position.py
        list every flagged discrepancy

    python tools/reconcile_position.py write-off STRATEGY SYMBOL --mark PRICE --note "why" --confirm
        the shares really are gone: write off exactly the flagged quantity
        (cash credited at min(PRICE, average entry), never a gain; journaled
        as WRITE_OFF_DRIFT in ledger_events plus a RECONCILE row in trades)

    python tools/reconcile_position.py clear STRATEGY SYMBOL --note "why" --confirm
        the broker does hold the shares (the flag was wrong): remove the flag
        only. The next cycle re-flags it if the broker still disagrees.

Without --confirm nothing is changed (dry run).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import portfolio  # noqa: E402
from core.execution import manual_write_off  # noqa: E402
from core.logger import log_trade  # noqa: E402

STRATEGIES = ["stable", "risky1", "risky2"]


def _print_flags(db_path=None):
    flags = portfolio.get_discrepancies(db_path=db_path)
    if not flags:
        print("No flagged discrepancies.")
        return flags
    for (strategy, symbol), d in sorted(flags.items()):
        print(f"{strategy:7s} {symbol:10s} ledger qty {d['qty']:.6f}  missing at broker {d['missing_qty']:.6f}  "
              f"avg entry {d['avg_entry_price']:.4f}  last mark {d['last_mark']}  since {d['since']}\n"
              f"        reason: {d['reason']}")
    return flags


def main(argv=None, db_path=None):
    p = argparse.ArgumentParser(description="Resolve ledger discrepancies flagged by the trading engine.")
    sub = p.add_subparsers(dest="cmd")
    for name in ("write-off", "clear"):
        sp = sub.add_parser(name)
        sp.add_argument("strategy", choices=STRATEGIES)
        sp.add_argument("symbol")
        sp.add_argument("--note", required=True, help="who checked what at the broker, and why")
        sp.add_argument("--confirm", action="store_true", help="actually change the ledger (default: dry run)")
        if name == "write-off":
            sp.add_argument("--mark", type=float, required=True, help="price used for the conservative credit")
    a = p.parse_args(argv)

    if a.cmd is None:
        _print_flags(db_path)
        return 0

    missing = portfolio.get_discrepancy_qty(a.strategy, a.symbol, db_path=db_path)
    if missing <= 0:
        print(f"{a.strategy} {a.symbol} has no flagged discrepancy; nothing to do.")
        return 1
    owned = portfolio.get_position_qty(a.strategy, a.symbol, db_path=db_path)
    print(f"{a.strategy} {a.symbol}: ledger qty {owned}, flagged missing {missing}")

    if a.cmd == "write-off":
        if not a.confirm:
            print(f"DRY RUN: would write off {missing} {a.symbol} at min({a.mark}, average entry). Add --confirm.")
            return 0
        credited, realized = manual_write_off(a.strategy, a.symbol, a.mark, a.note, db_path=db_path)
        print(f"Written off {missing} {a.symbol}: cash credited {credited:.2f}, realized {realized:.2f}")
        return 0

    if not a.confirm:
        print("DRY RUN: would clear the flag (ledger unchanged). Add --confirm.")
        return 0
    portfolio.clear_discrepancy(a.strategy, a.symbol, db_path=db_path)
    log_trade(a.strategy, a.symbol, "RECONCILE", 0.0, missing, pnl=None, status="discrepancy_cleared_manual",
              reason=f"flag cleared by operator: {a.note}", db_path=db_path)
    print("Flag cleared; ledger unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
