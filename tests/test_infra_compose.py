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


# ---------------------------------------------------------------------------
# Phase 2 I-5: image, dependency pins and ARM64 CI
# ---------------------------------------------------------------------------

def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def test_dockerfile_invariants():
    d = _read("Dockerfile")
    assert "FROM python:3.11-slim-bookworm" in d                 # pinned distro -> stable glibc / wheel tags
    assert "PIP_NO_CACHE_DIR=1" in d
    assert "SINGHQUANT_DB_PATH=/app/data/trades.db" in d and "SINGHQUANT_MODEL_DIR=/app/model_artifacts" in d
    assert "bash setup.sh --pinned --with-rl" in d
    assert "python -m tools.smoke_imports --require-rl" in d    # the build fails if the CPU's wheels do not work
    assert "STOPSIGNAL SIGTERM" in d


def test_alpaca_runtime_dependencies_are_exactly_pinned_for_image_builds():
    lines = [l.strip() for l in _read("requirements-alpaca-runtime.txt").splitlines()
             if l.strip() and not l.startswith("#")]
    assert {l.split("==")[0].lower() for l in lines} == {"aiohttp", "deprecation", "msgpack", "pyyaml",
                                                          "websocket-client", "websockets"}
    assert all("==" in l for l in lines)
    setup = _read("setup.sh")
    assert "pip install -r requirements-alpaca-runtime.txt" in setup


def test_ci_builds_and_tests_natively_on_arm64():
    wf = yaml.safe_load(_read(".github/workflows/test.yml"))
    assert "ubuntu-24.04-arm" in wf["jobs"]["test"]["strategy"]["matrix"]["runner"]
    assert "ubuntu-24.04-arm" in wf["jobs"]["docker-image"]["strategy"]["matrix"]["runner"]
    steps = " ".join(s.get("run", "") for s in wf["jobs"]["docker-image"]["steps"])
    assert "docker build" in steps and "--network none" in steps and "pytest" in steps
    assert "docker push" not in steps                              # CI never publishes images


def test_smoke_imports_pass_without_network_or_credentials(tmp_path):
    import subprocess, sys
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}   # no keys, no DB path, nothing inherited
    r = subprocess.run([sys.executable, "-m", "tools.smoke_imports"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-1500:]
    assert "smoke test ok" in r.stdout
    assert not os.path.exists(os.path.join(ROOT, "trades.db"))
