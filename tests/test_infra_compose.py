"""
Phase 2: docker-compose.yml invariants (I-1 WAL persistence, I-2 migration
ordering, I-3 model persistence, I-4 startup and recovery). Docker itself is
not needed: these guard the file against regressions.
"""
import os

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def compose():
    with open(os.path.join(ROOT, "docker-compose.yml")) as f:
        return yaml.safe_load(f)


def _volumes(service):
    return [v.split(":") for v in service.get("volumes", [])]


def test_database_is_on_a_directory_volume_never_a_single_file_mount(compose):
    for name, svc in compose["services"].items():
        for host, container, *_ in _volumes(svc):
            assert not host.endswith(".db") and not container.endswith(".db"), f"{name}: single-file DB mount"
        if "SINGHQUANT_DB_PATH" in svc.get("environment", {}):
            assert svc["environment"]["SINGHQUANT_DB_PATH"] == "/app/data/trades.db"
            assert ["./data", "/app/data"] in _volumes(svc), f"{name}: database directory not mounted"


def test_every_service_shares_the_same_database_directory(compose):
    paths = {svc["environment"]["SINGHQUANT_DB_PATH"] for svc in compose["services"].values()}
    assert paths == {"/app/data/trades.db"}


def test_one_shot_migration_runs_first_and_does_not_restart(compose):
    mig = compose["services"]["migrate"]
    assert mig["command"] == ["python", "-m", "tools.db_maintenance", "migrate"]
    assert mig["restart"] == "no"
    for name in ("trading_bots", "dashboard"):
        assert compose["services"][name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"


def test_dashboard_is_read_only_and_bots_get_the_model_volume(compose):
    dash, bots = compose["services"]["dashboard"], compose["services"]["trading_bots"]
    assert dash["environment"]["SINGHQUANT_DB_READONLY"] == "1"
    assert ["./model_artifacts", "/app/model_artifacts"] not in _volumes(dash)
    assert bots["environment"]["SINGHQUANT_MODEL_DIR"] == "/app/model_artifacts"
    assert ["./model_artifacts", "/app/model_artifacts"] in _volumes(bots)
    assert "SINGHQUANT_DB_READONLY" not in bots["environment"]


def test_signals_restarts_and_health_checks(compose):
    bots = compose["services"]["trading_bots"]
    common = compose["x-common"]
    assert common["init"] is True                                  # SIGTERM reaches Python
    assert bots["restart"] == "unless-stopped"
    assert bots["stop_signal"] == "SIGTERM"
    grace = int(str(bots["stop_grace_period"]).rstrip("s"))
    from core.config import SHUTDOWN_GRACE_SECONDS
    assert grace > SHUTDOWN_GRACE_SECONDS                          # Docker must not SIGKILL before run.py finishes
    assert bots["healthcheck"]["test"] == ["CMD", "python", "-m", "tools.healthcheck", "liveness"]
    from core.config import HEARTBEAT_STALE_SECONDS, PAUSED_HEARTBEAT_SECONDS
    assert PAUSED_HEARTBEAT_SECONDS < HEARTBEAT_STALE_SECONDS      # a paused bot never looks dead
    assert compose["services"]["dashboard"]["healthcheck"]["test"][-1] == "dashboard"


def test_logs_are_rotated(compose):
    opts = compose["x-common"]["logging"]["options"]
    assert compose["x-common"]["logging"]["driver"] == "json-file"
    assert opts["max-size"] == "10m" and opts["max-file"] == "5"


def test_runtime_state_never_enters_the_image_or_git():
    docker_ignore = open(os.path.join(ROOT, ".dockerignore")).read().split()
    git_ignore = open(os.path.join(ROOT, ".gitignore")).read().split()
    for entry in ("data/", "model_artifacts/", "*.db", "*.pkl"):
        assert entry in docker_ignore, entry
    for entry in ("/data/", "/model_artifacts/"):
        assert entry in git_ignore, entry
