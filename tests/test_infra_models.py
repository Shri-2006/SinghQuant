"""
Phase 2 I-3: model artifacts are persistent, manifest-tracked and verified.

* .dockerignore excludes *.pkl / *.zip, and compose did not mount the model
  directory, so models lived only in a container's private filesystem.
* A missing model used to be a print() once and silent no-trading.
Now a strategy only opens new entries while every artifact it needs is in
manifest.json with a matching SHA-256 (checked BEFORE unpickling); anything
else is an explicit status, a heartbeat readiness flag and an alert, while
existing positions keep their stops and kill switch.
"""
import itertools
import json
import os
import pickle
import threading
from datetime import timedelta

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from core import portfolio
from core.execution import submit_and_track
from core.logger import get_heartbeat_full
from models import artifacts
from models.train import FEATURE_COLUMNS, ModelHandle, save_model, load_model
from strategies.common import run_cycle, xgb_decider, artifact_status, utcnow
from tests.conftest import make_featured_df
from tests.fakes import FakeAlpaca
from tests.test_engine import make_ctx

FAST = dict(sleep_fn=lambda s: None)
_clock = itertools.count(0, 30)
EXEC = dict(sleep_fn=lambda s: None, clock_fn=lambda: next(_clock))


def _model(n_features=len(FEATURE_COLUMNS), predict=1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features))
    y = np.array([0, 1] * 20)
    m = LogisticRegression().fit(X, y)
    m.coef_[:] = 0.0
    m.intercept_[:] = 50.0 if predict == 1 else -50.0          # deterministic prediction
    return m


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "model_artifacts"
    d.mkdir()
    monkeypatch.setenv("SINGHQUANT_MODEL_DIR", str(d))
    return str(d)


# ---------------------------------------------------------------------------
# Publishing and the manifest
# ---------------------------------------------------------------------------

def test_publish_writes_a_content_addressed_file_and_a_manifest_entry(model_dir):
    entry = save_model(_model(), "stable_model.pkl", version="v1", meta={"note": "t"})
    files = sorted(os.listdir(model_dir))
    assert entry["file"] in files and entry["file"].startswith("stable_model.") and entry["file"].endswith(".pkl")
    assert entry["file"] == f"stable_model.{entry['sha256'][:12]}.pkl"
    assert entry["sha256"] == artifacts.sha256_file(os.path.join(model_dir, entry["file"]))
    manifest = json.load(open(os.path.join(model_dir, "manifest.json")))
    assert manifest["schema"] == 1
    got = manifest["artifacts"]["stable_model.pkl"]
    assert got["version"] == "v1" and got["strategy"] == "stable" and got["meta"] == {"note": "t"}
    assert not any(f.startswith(".tmp-") for f in files)


def test_republishing_keeps_the_previous_entry_and_file_for_rollback(model_dir):
    first = save_model(_model(predict=1), "stable_model.pkl", version="v1")
    second = save_model(_model(predict=0), "stable_model.pkl", version="v2")
    entry = artifacts.load_manifest(model_dir)["artifacts"]["stable_model.pkl"]
    assert entry["version"] == "v2" and entry["previous"]["version"] == "v1"
    assert os.path.exists(os.path.join(model_dir, first["file"]))      # old file kept
    assert first["file"] != second["file"]


# ---------------------------------------------------------------------------
# Verification statuses
# ---------------------------------------------------------------------------

def _handle(model_dir, calls=None):
    from models.train import _load_pickle, check_feature_width
    def loader(path):
        if calls is not None:
            calls.append(path)
        return _load_pickle(path)
    return artifacts.ArtifactHandle("stable_model.pkl", loader, check_feature_width, model_dir)


def test_unregistered_artifact_is_not_loaded(model_dir):
    with open(os.path.join(model_dir, "stable_model.pkl"), "wb") as f:      # a legacy, unregistered file
        pickle.dump(_model(), f)
    h = ModelHandle("stable_model.pkl")
    assert h.model is None and h.status == artifacts.UNREGISTERED and "tools/model_manifest.py" in h.detail


def test_missing_file_is_reported(model_dir):
    entry = save_model(_model(), "stable_model.pkl", version="v1")
    os.remove(os.path.join(model_dir, entry["file"]))
    h = ModelHandle("stable_model.pkl")
    assert h.model is None and h.status == artifacts.MISSING


def test_tampered_file_is_never_unpickled(model_dir):
    entry = save_model(_model(), "stable_model.pkl", version="v1")
    with open(os.path.join(model_dir, entry["file"]), "ab") as f:
        f.write(b"tampered")
    calls = []
    h = _handle(model_dir, calls)
    assert h.model is None and h.status == artifacts.HASH_MISMATCH
    assert calls == []                                               # verification happened BEFORE loading


def test_invalid_manifest_is_reported(model_dir):
    save_model(_model(), "stable_model.pkl", version="v1")
    with open(os.path.join(model_dir, "manifest.json"), "w") as f:
        f.write("{not json")
    h = ModelHandle("stable_model.pkl")
    assert h.model is None and h.status == artifacts.MANIFEST_INVALID


def test_wrong_feature_width_is_incompatible(model_dir):
    save_model(_model(n_features=5), "stable_model.pkl", version="v1")
    h = ModelHandle("stable_model.pkl")
    assert h.model is None and h.status == artifacts.INCOMPATIBLE and "expects 5 features" in h.detail


def test_unloadable_file_is_a_load_error(model_dir):
    artifacts.publish("stable_model.pkl", lambda tmp: open(tmp, "wb").close() or None, "stable", "v1")
    h = ModelHandle("stable_model.pkl")
    assert h.model is None and h.status == artifacts.LOAD_ERROR


def test_a_model_that_appears_later_is_picked_up(model_dir):
    h = ModelHandle("stable_model.pkl")
    assert h.model is None
    save_model(_model(), "stable_model.pkl", version="v1")
    assert h.model is not None and h.status == artifacts.OK
    assert h.entry["version"] == "v1"
    save_model(_model(predict=0), "stable_model.pkl", version="v2")          # retrain: hot reload
    assert h.model.predict(np.zeros((1, len(FEATURE_COLUMNS))))[0] == 0 and h.entry["version"] == "v2"


def test_reader_never_sees_a_broken_artifact_while_retraining_publishes(model_dir):
    save_model(_model(), "stable_model.pkl", version="v0")
    h = ModelHandle("stable_model.pkl")
    seen, stop = set(), threading.Event()

    def reader():
        while not stop.is_set():
            seen.add(h.refresh())

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(15):
            save_model(_model(predict=i % 2), "stable_model.pkl", version=f"v{i + 1}")
    finally:
        stop.set(); t.join()
    assert seen == {artifacts.OK}


# ---------------------------------------------------------------------------
# Engine: entries blocked, risk management intact, visible status and alerts
# ---------------------------------------------------------------------------

def _xgb_ctx(api, frames, db_path, model_dir):
    handle = ModelHandle("stable_model.pkl")
    ctx = make_ctx("stable", api, frames, decide=xgb_decider(handle), db_path=db_path,
                   model_status=artifact_status(handle))
    sent = []
    ctx.alert = lambda **kw: sent.append(kw)
    return ctx, sent


def test_missing_model_blocks_entries_but_stop_loss_still_fires(db_path, model_dir):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0, "QQQ": 100.0})
    submit_and_track(api, "stable", "SPY", "buy", 1.0, 100.0, "x", db_path=db_path, **EXEC)
    api.set_price("SPY", 90.0)                                      # -10%: below the 5% stop
    frames = {"SPY": make_featured_df(price=90.0, atr=1.35), "QQQ": make_featured_df(price=100.0, atr=1.5)}
    ctx, sent = _xgb_ctx(api, frames, db_path, model_dir)
    out = run_cycle(ctx, api, **FAST)
    assert out["tickers"]["SPY"] == "emergency-exit"               # risk management unaffected
    assert out["tickers"]["QQQ"] == "no-model"                     # no new entry without a verified model
    assert not any(o["side"] == "buy" for o in api.submitted[1:])
    hb = get_heartbeat_full("stable", db_path)
    assert hb["status"] == "RUNNING" and hb["ready"] is False      # alive, not ready
    assert "unregistered" in hb["degraded_reason"]
    assert [a["alert_type"] for a in sent] == ["MODEL UNAVAILABLE"]


def test_model_alert_repeats_every_six_hours_and_recovery_is_announced(db_path, model_dir):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0})
    ctx, sent = _xgb_ctx(api, {"SPY": make_featured_df(price=100.0, atr=1.5)}, db_path, model_dir)
    t0 = utcnow()
    for minutes in (0, 60, 300, 359):
        run_cycle(ctx, api, now=t0 + timedelta(minutes=minutes), **FAST)
    assert [a["alert_type"] for a in sent] == ["MODEL UNAVAILABLE"]
    run_cycle(ctx, api, now=t0 + timedelta(minutes=361), **FAST)
    assert [a["alert_type"] for a in sent] == ["MODEL UNAVAILABLE"] * 2
    save_model(_model(), "stable_model.pkl", version="v1")
    out = run_cycle(ctx, api, now=t0 + timedelta(minutes=362), **FAST)
    assert [a["alert_type"] for a in sent][-1] == "MODEL RESTORED"
    assert out["ready"] is True and get_heartbeat_full("stable", db_path)["ready"] is True
    assert out["tickers"]["SPY"] == "buy-filled"                   # entries resume once verified


def test_readiness_is_reported_while_the_market_is_closed(db_path, model_dir):
    api = FakeAlpaca(cash=100000, prices={"SPY": 100.0}, market_open=False)
    ctx, sent = _xgb_ctx(api, {"SPY": make_featured_df(price=100.0)}, db_path, model_dir)
    out = run_cycle(ctx, api, **FAST)
    assert out["paused"] and out["ready"] is False
    hb = get_heartbeat_full("stable", db_path)
    assert hb["status"] == "PAUSED" and hb["ready"] is False
    assert [a["alert_type"] for a in sent] == ["MODEL UNAVAILABLE"]  # known before the open, not at it


def test_artifact_status_names_every_unusable_artifact(model_dir):
    from models.train import _load_pickle
    save_model(_model(), "stable_model.pkl", version="v1")
    ok = artifacts.ArtifactHandle("stable_model.pkl", _load_pickle)
    gone = artifacts.ArtifactHandle("risky2_BTCUSD.zip", _load_pickle)
    good, reason = artifact_status(ok, gone)()
    assert good is False and "1/2" in reason and "risky2_BTCUSD.zip: unregistered" in reason
    assert artifact_status(ok)() == (True, "")


# ---------------------------------------------------------------------------
# Training, offline loading and the register tool
# ---------------------------------------------------------------------------

def test_train_and_save_publishes_with_metadata(model_dir, monkeypatch):
    import pandas as pd
    import data.polygon_fetcher as pf
    from models import train
    def fake_fetch(tickers, start, end, delay=12, **kw):
        idx = pd.date_range("2024-01-01", periods=260, freq="D")
        rng = np.random.default_rng(1)
        out = {}
        for t in tickers[:2]:
            close = 100 + np.cumsum(rng.normal(size=len(idx)))
            out[t] = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                                   "volume": rng.integers(1e5, 2e5, len(idx)).astype(float)}, index=idx)
        return out
    monkeypatch.setattr(pf, "get_multiple_tickers", fake_fetch)
    train.train_and_save("stable", "2024-01-01", "2024-12-31", delay=0)
    entry = artifacts.load_manifest()["artifacts"]["stable_model.pkl"]
    assert entry["meta"]["n_features"] == len(FEATURE_COLUMNS) and entry["meta"]["train_end"] == "2024-12-31"
    assert ModelHandle("stable_model.pkl").model is not None             # verified and usable by the bot


def test_offline_load_model_verifies_registered_and_warns_on_legacy(model_dir, capsys):
    save_model(_model(), "stable_model.pkl", version="v1")
    assert load_model("stable_model.pkl") is not None
    with open(os.path.join(model_dir, "risky1_model.pkl"), "wb") as f:
        pickle.dump(_model(), f)
    assert load_model("risky1_model.pkl") is not None
    assert "not in the manifest" in capsys.readouterr().out
    entry = artifacts.load_manifest()["artifacts"]["stable_model.pkl"]
    with open(os.path.join(model_dir, entry["file"]), "ab") as f:
        f.write(b"x")
    with pytest.raises(artifacts.ArtifactError):
        load_model("stable_model.pkl")


def test_register_tool_dry_run_confirm_and_rejection(model_dir, tmp_path, capsys):
    from tools.model_manifest import main
    legacy = tmp_path / "old_container_stable_model.pkl"
    with open(legacy, "wb") as f:
        pickle.dump(_model(), f)
    before = artifacts.sha256_file(str(legacy))
    args = ["register", "stable_model.pkl", "--file", str(legacy), "--version", "prod-legacy", "--note", "copied from old container"]
    assert main(args) == 0 and "DRY RUN" in capsys.readouterr().out
    assert artifacts.load_manifest()["artifacts"] == {}
    assert main(args + ["--confirm"]) == 0
    assert ModelHandle("stable_model.pkl").model is not None
    assert artifacts.sha256_file(str(legacy)) == before                       # source untouched
    wrong = tmp_path / "wrong.pkl"
    with open(wrong, "wb") as f:
        pickle.dump(_model(n_features=3), f)
    assert main(["register", "risky1_model.pkl", "--file", str(wrong), "--version", "x", "--note", "n", "--confirm"]) == 2
    assert "expects 3 features" in capsys.readouterr().out
    assert main(["register", "nonsense.pkl", "--file", str(legacy), "--version", "x", "--note", "n"]) == 2


def test_status_tool_exit_code_reflects_readiness(model_dir, capsys):
    from tools.model_manifest import main
    assert main(["status"]) == 1                                   # nothing registered
    save_model(_model(), "stable_model.pkl", version="v1")
    save_model(_model(), "risky1_model.pkl", version="v1")
    assert main(["status"]) == 0
