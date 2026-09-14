"""
Startup, migration and training-helper regressions.

* an old-schema trades.db (9 columns, no ledger tables) is migrated in place and
  its historical rows remain readable,
* pre-existing broker positions are NOT adopted unless explicitly requested,
* run_forever executes a cycle with a fake broker without network or models,
* create_labels drops the last (unlabelable) row; time_split is chronological.
"""
import os
import sqlite3

import numpy as np
import pytest

from core import portfolio
from core.logger import init_db, get_trades, get_trades_full, log_trade
from strategies.common import StrategyContext, Decision, run_forever, adopt_unowned_broker_positions
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca


def _make_legacy_db(path):
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL, asset TEXT NOT NULL, action TEXT NOT NULL, price REAL NOT NULL,
                    quantity REAL NOT NULL, pnl REAL, reason TEXT)''')
    conn.execute('''CREATE TABLE heartbeat (strategy TEXT PRIMARY KEY, last_seen TEXT NOT NULL, status TEXT NOT NULL)''')
    # the historical Aug-27 pattern: BUY in shares, SELL in dollars
    rows = [("2026-08-27T17:52:54", "risky1", "TSLA", "BUY", 345.82, 0.1446, None, "ML signals BUY, regime favorable"),
            ("2026-08-27T17:55:16", "risky1", "TSLA", "SELL", 345.82, 51.225, None, "Momentum turned negative"),
            ("2026-07-07T14:30:00", "stable", "ALL", "KILL_SWITCH", 0, 0, -0.9495, "Drawdown -94.95% from peak")]
    conn.executemany("INSERT INTO trades (timestamp,strategy,asset,action,price,quantity,pnl,reason) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_legacy_database_is_migrated_in_place_and_history_preserved(tmp_path):
    path = str(tmp_path / "legacy.db")
    _make_legacy_db(path)
    init_db(path)                                   # additive migration
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"status", "filled_qty", "filled_avg_price", "order_id", "signal_price"} <= cols
    assert {"positions", "orders", "strategy_state", "signal_state", "equity_snapshots"} <= tables
    conn = sqlite3.connect(path)
    assert "client_order_id" in {r[1] for r in conn.execute("PRAGMA table_info(orders)")}
    conn.close()
    init_db(path)                                   # running the migration twice is harmless
    old = get_trades("risky1", path)
    assert len(old) == 2 and old[1][6] == pytest.approx(51.225)        # historical dollar-quantity row untouched
    full = get_trades_full("risky1", path)
    assert all(r["status"] is None for r in full)                       # legacy rows recognisable
    log_trade("risky1", "TSLA", "SELL", 345.82, 0.1446, pnl=1.2, status="filled", filled_qty=0.1446, db_path=path)
    assert get_trades_full("risky1", path)[-1]["status"] == "filled"
    assert portfolio.get_strategy_state("risky1", path)["cash_budget"] == 200.0


def test_existing_broker_positions_are_reported_not_adopted_by_default(db_path, monkeypatch):
    monkeypatch.delenv("SINGHQUANT_ADOPT_BROKER_POSITIONS", raising=False)
    api = FakeAlpaca(prices={"SPY": 500.0, "TSLA": 350.0})
    api.seed_position("SPY", 0.4, 480.0)
    api.seed_position("TSLA", 50.0, 300.0)
    ctx = StrategyContext(name="stable", assets=["SPY", "QQQ"], decide=lambda *a: Decision("HOLD"),
                          fetch_frame=lambda t: None, db_path=db_path)
    assert adopt_unowned_broker_positions(ctx, api) == []
    assert portfolio.list_positions("stable", db_path) == {}
    monkeypatch.setenv("SINGHQUANT_ADOPT_BROKER_POSITIONS", "1")
    adopted = adopt_unowned_broker_positions(ctx, api)
    assert adopted == [("SPY", 0.4)]                                    # TSLA is not in stable's universe
    qty, avg, _ = portfolio.get_position("stable", "SPY", db_path)
    assert qty == pytest.approx(0.4) and avg == pytest.approx(480.0)
    assert portfolio.get_strategy_state("stable", db_path)["cash_budget"] == pytest.approx(1000 - 0.4 * 480)


def test_run_forever_completes_a_cycle_with_fake_broker(db_path):
    api = FakeAlpaca(prices={"SPY": 500.0})
    calls = {"notify": 0}
    def notify(**kw):
        calls["notify"] += 1
    ctx = StrategyContext(name="stable", assets=["SPY"], decide=lambda *a: Decision("BUY", 1, "t"),
                          fetch_frame=lambda t: make_featured_df(price=500.0), uses_market_hours=True,
                          per_ticker_sleep=0, cycle_sleep=0, regime_ok=lambda df, s, v: True,
                          notify=notify, db_path=db_path, exec_kw=dict(sleep_fn=lambda s: None))
    run_forever(ctx, api=api, sleep_fn=lambda s: None, max_cycles=2)
    assert len(api.submitted) == 1 and calls["notify"] == 2
    assert portfolio.get_position_qty("stable", "SPY", db_path) == pytest.approx(0.4)


def test_run_forever_survives_broker_exceptions(db_path):
    class Broken(FakeAlpaca):
        def get_clock(self):
            raise RuntimeError("401 unauthorized")
    api = Broken(prices={"SPY": 500.0})
    ctx = StrategyContext(name="stable", assets=["SPY"], decide=lambda *a: Decision("BUY"),
                          fetch_frame=lambda t: make_featured_df(price=500.0), per_ticker_sleep=0, cycle_sleep=0,
                          db_path=db_path)
    run_forever(ctx, api=api, sleep_fn=lambda s: None, max_cycles=3)   # must not raise
    assert api.submitted == []


def test_create_labels_and_time_split():
    from models.train import create_labels, time_split
    df = make_featured_df(rows=20, price=100.0)
    labelled = create_labels(df)
    assert len(labelled) == 19                                          # last row has no "tomorrow"
    assert set(labelled["label"].unique()) <= {0, 1}
    X = np.arange(10).reshape(-1, 1); Y = np.arange(10)
    a, b, c, d = time_split(X, Y, 0.2)
    assert list(c) == list(range(8)) and list(d) == [8, 9]              # chronological, no shuffle
