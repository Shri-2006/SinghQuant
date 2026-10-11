"""
Container health checks (Phase 2 I-4). Reads the database READ-ONLY.

    python -m tools.healthcheck liveness     exit 0 if every enabled strategy's heartbeat is fresh
    python -m tools.healthcheck readiness    exit 0 if additionally every strategy may open new entries
    python -m tools.healthcheck report       JSON: liveness, readiness and degraded reasons per strategy
    python -m tools.healthcheck dashboard    exit 0 if Streamlit answers /_stcore/health

Liveness is what Docker's HEALTHCHECK uses: a DEGRADED strategy (model
missing, suspect broker data, halted by the kill switch, safe mode) is still
alive and still managing its positions, so it must not look dead. Readiness
and the degraded reasons are reported separately for people and dashboards.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SINGHQUANT_NO_AUTO_MIGRATE", "1")

from core.config import HEARTBEAT_STALE_SECONDS, FEATURE_FLAGS  # noqa: E402

NOT_ALIVE_STATUSES = {"STOPPED"}


def enabled_strategies():
    return ["stable", "risky1"] + (["risky2"] if FEATURE_FLAGS.get("risky2_enabled") else [])


def collect(db_path=None, now=None, stale_seconds=HEARTBEAT_STALE_SECONDS):
    from core.logger import get_heartbeat_full
    from core import lifecycle
    now = now or datetime.now(timezone.utc)
    out = {"strategies": {}, "safe_mode": None, "error": None}
    try:
        out["safe_mode"] = lifecycle.get_safe_mode(db_path, readonly=True)
    except Exception as e:
        out["error"] = f"cannot read system state: {e}"
    for s in enabled_strategies():
        try:
            hb = get_heartbeat_full(s, db_path, readonly=True)
        except Exception as e:
            out["strategies"][s] = {"alive": False, "ready": False, "status": None, "age_seconds": None,
                                    "degraded_reason": f"heartbeat unreadable: {e}"}
            continue
        if hb is None:
            out["strategies"][s] = {"alive": False, "ready": False, "status": None, "age_seconds": None,
                                    "degraded_reason": "no heartbeat yet"}
            continue
        seen = datetime.fromisoformat(hb["last_seen"]).replace(tzinfo=timezone.utc)
        age = (now - seen).total_seconds()
        alive = age <= stale_seconds and hb["status"] not in NOT_ALIVE_STATUSES
        out["strategies"][s] = {"alive": alive, "ready": bool(alive and hb["ready"]), "status": hb["status"],
                                "age_seconds": round(age, 1), "degraded_reason": hb["degraded_reason"]}
    out["alive"] = bool(out["strategies"]) and all(v["alive"] for v in out["strategies"].values())
    out["ready"] = out["alive"] and all(v["ready"] for v in out["strategies"].values()) and not out["safe_mode"]
    return out


def dashboard_ok(url="http://127.0.0.1:8501/_stcore/health", timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def main(argv=None, db_path=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "report"
    if mode == "dashboard":
        ok = dashboard_ok()
        print("dashboard ok" if ok else "dashboard not responding")
        return 0 if ok else 1
    state = collect(db_path)
    if mode == "report":
        print(json.dumps(state, indent=2, default=str))
        return 0
    if mode == "liveness":
        for s, v in state["strategies"].items():
            if not v["alive"]:
                print(f"NOT ALIVE: {s} status={v['status']} age={v['age_seconds']}s ({v['degraded_reason']})")
        return 0 if state["alive"] else 1
    if mode == "readiness":
        if state["safe_mode"]:
            print(f"NOT READY: SAFE MODE ({state['safe_mode'].get('reason')})")
        for s, v in state["strategies"].items():
            if not v["ready"]:
                print(f"NOT READY: {s} ({v['degraded_reason']})")
        return 0 if state["ready"] else 1
    print(f"unknown mode {mode!r}; use liveness, readiness, report or dashboard")
    return 2


if __name__ == "__main__":
    sys.exit(main())
