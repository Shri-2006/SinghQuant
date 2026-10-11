"""
SAFE MODE control (Phase 2 I-4).

SAFE MODE is entered automatically after repeated unclean exits (a crash
loop). It blocks NEW entries for every strategy; exits, stop losses and the
kill switch keep running. It is stored in the database, so restarts cannot
clear it. Only a human can, after finding out why the process kept dying:

    python -m tools.safe_mode status
    python -m tools.safe_mode clear --note "who / why it is safe now" --confirm
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SINGHQUANT_NO_AUTO_MIGRATE", "1")

from core import lifecycle  # noqa: E402


def main(argv=None, db_path=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("clear")
    c.add_argument("--note", required=True)
    c.add_argument("--confirm", action="store_true")
    a = p.parse_args(argv)
    state = lifecycle.get_safe_mode(db_path)
    if a.cmd == "status":
        print(f"SAFE MODE active since {state['since']}: {state['reason']}" if state else "SAFE MODE is off")
        for r in lifecycle.recent_runs(10, db_path):
            print(f"  run {r['id']}: started {r['started_at']} ended {r['ended_at']} ({r['end_reason']})")
        return 0
    if not state:
        print("SAFE MODE is off; nothing to clear.")
        return 1
    if not a.confirm:
        print(f"DRY RUN: would clear SAFE MODE ({state['reason']}). Add --confirm.")
        return 0
    lifecycle.clear_safe_mode(a.note, db_path)
    print("SAFE MODE cleared. New entries resume on each strategy's next cycle after startup reconciliation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
