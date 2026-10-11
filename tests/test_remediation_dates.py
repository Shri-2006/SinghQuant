"""
Phase 1 remediation R-2: the test suite must not depend on the calendar.

The audit-hardening fixtures hard-coded start="2026-08-01"; once those bars
were older than STALE_BAR_MAX_DAYS (2026-10-04) every BUY in the engine tests
became "stale-bar" and 16 tests failed. Fixture bars are now anchored to
today (or to an explicit frozen clock).
"""
import pathlib
import re
from datetime import timedelta

import pandas as pd

from core.config import STALE_BAR_MAX_DAYS
from strategies.common import run_cycle, utcnow
from tests.conftest import make_featured_df, fixture_today
from tests.fakes import FakeAlpaca
from tests.test_engine import make_ctx, always

FAST = dict(sleep_fn=lambda s: None)
TESTS_DIR = pathlib.Path(__file__).parent


def test_default_fixture_bars_end_today_and_are_never_stale():
    df = make_featured_df()
    age_days = (pd.Timestamp.now(tz="UTC").tz_localize(None) - df.index[-1]).total_seconds() / 86400
    assert 0 <= age_days < 1
    assert age_days < STALE_BAR_MAX_DAYS


def test_end_offsets_give_distinct_bar_ids_and_equal_offsets_the_same_bar():
    a1, a2, b = make_featured_df(end_offset_days=1), make_featured_df(end_offset_days=1), make_featured_df()
    assert a1.index[-1] == a2.index[-1]
    assert b.index[-1] - a1.index[-1] == pd.Timedelta(days=1)
    assert a1.index[-1] == fixture_today() - pd.Timedelta(days=1)


def test_no_test_hard_codes_a_fixture_start_date():
    """Guard: a literal start date in make_featured_df(...) is a time bomb."""
    pattern = re.compile(r"make_featured_df\([^)]*\bstart\s*=")
    offenders = []
    for path in TESTS_DIR.glob("test_*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{n}")
    assert offenders == []


def test_stale_bar_logic_under_a_frozen_clock(db_path):
    """The engine's staleness decision follows the injected clock, not the wall clock."""
    frozen = utcnow().replace(year=2031, month=3, day=14, hour=15, minute=0, second=0, microsecond=0)
    df = make_featured_df(price=500.0, now=frozen)
    api = FakeAlpaca(prices={"SPY": 500.0})
    ctx = make_ctx("stable", api, {"SPY": df}, decide=always("HOLD"), db_path=db_path)

    fresh = run_cycle(ctx, api, now=frozen, **FAST)["risk"]
    assert fresh.stale_symbols == []
    later = run_cycle(ctx, api, now=frozen + timedelta(days=STALE_BAR_MAX_DAYS + 1), **FAST)["risk"]
    assert later.stale_symbols == ["SPY"]
