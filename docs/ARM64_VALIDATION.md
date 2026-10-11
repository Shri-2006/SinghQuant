# ARM64 (Oracle Cloud Ampere) validation

## What was checked, and how

**Dependency availability (2026-10-11, from an x86 sandbox, no Docker daemon).**
The full dependency closure of `requirements-py311-pinned.txt` plus
`requirements-alpaca-runtime.txt` was resolved, and every exact version was
checked against PyPI's file listing for a Linux aarch64 wheel that
CPython 3.11 can use:

| Result | Packages |
|---|---|
| 35 compiled aarch64 wheels | numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy, xgboost 3.2.0, numba / llvmlite (vectorbt), pyarrow, torch 2.11.0 (PyPI), aiohttp, PyYAML, msgpack, websockets 16.1.1, cryptography, pillow, … |
| 93 pure-Python wheels | everything else |
| 2 pure-Python source packages | `ta==0.11.0`, `sgmllib3k` (pulled in by feedparser): need only setuptools, no compiler |
| Problems | none |

`python:3.11-slim-bookworm` has glibc 2.36, which accepts every manylinux tag
from `2_17` through `2_36`. No compiler is needed, and none is installed.

**Not verifiable from the sandbox:**

* `download.pytorch.org` was blocked by its network policy, so the CPU-only torch
  wheel for aarch64 (`setup.sh --pinned --with-rl` installs torch from that
  index) was not checked. The CI arm64 jobs and step 3 below check it.
* No image was built and nothing ran on ARM hardware here.

**Notes**

* `xgboost==3.2.0` on Linux pulls `nvidia-nccl-cu12` (about 200 MB) on both
  architectures. It is unused on CPU. `xgboost-cpu==3.2.0` avoids it, but
  switching was deferred (owner decision D5); it would also need a re-check
  that existing pickles load.
* The pinned set requires Python 3.11 (numpy 2.2.6 has no wheels for newer
  versions). The image pins it.

## Continuous integration (automatic, every push)

The repository is public, so `.github/workflows/test.yml` runs on GitHub's
native `ubuntu-24.04-arm` runners (no emulation, no cost):

* `test` job: Python 3.11 (pinned set, CPU torch) and 3.12 (ranges) on amd64 **and** arm64.
* `docker-image` job on amd64 **and** arm64:
  1. validates `docker-compose.yml`;
  2. builds the image (the build runs `tools/smoke_imports.py --require-rl`);
  3. prints `uname -m` and key versions;
  4. runs the full test suite inside the image with `--network none`.

  Nothing is pushed to any registry.

A green arm64 `docker-image` job is the main evidence that the image works on Ampere.

## On the Oracle host (manual, staging only, no trading)

Use a **separate directory and compose project** so production data is never touched.

```bash
uname -m                                  # aarch64
docker version && docker compose version
df -h /                                   # image ~2-3 GB; keep 5 GB free

git clone https://github.com/Shri-2006/SinghQuant ~/singhquant-staging
cd ~/singhquant-staging && git checkout claude-infrastructure

# 1. build natively; the build itself runs the smoke test
docker build -t singhquant:staging .

# 2. platform, versions, CPU torch
docker run --rm singhquant:staging sh -c 'uname -m && python -m tools.smoke_imports --require-rl'
docker run --rm singhquant:staging python -c "import torch; print(torch.__version__, 'cuda:', torch.version.cuda)"
#    expect cuda: None (CPU build)

# 3. the full test suite, offline, dummy keys
docker run --rm --network none -e POLYGON_API_KEY=test_key -e ALPACA_API_KEY=test_key \
  -e ALPACA_SECRET_KEY=test_key singhquant:staging \
  python -m pytest tests/ -q --deselect tests/test_regime.py::TestVIXSignal::test_get_vix_returns_float_or_none

# 4. installed versions match the pins
docker run --rm singhquant:staging pip freeze > staging-freeze.txt
grep -iE '^(numpy|pandas|xgboost|scikit-learn|torch|websockets|aiohttp|pyyaml|msgpack)==' staging-freeze.txt

# 5. container lifecycle with DUMMY keys (the bots cannot reach Alpaca; that is expected)
printf 'ALPACA_API_KEY=dummy\nALPACA_SECRET_KEY=dummy\nPOLYGON_API_KEY=dummy\n' > .env
docker compose -p singhquant-staging up -d
docker compose -p singhquant-staging ps                     # migrate Exited (0); others Up
ls -la data/                                                # trades.db (+ -wal / -shm) on the host
time docker compose -p singhquant-staging stop trading_bots # should take seconds, not the 60 s grace period
docker compose -p singhquant-staging run --rm --no-deps migrate python -m tools.safe_mode status   # last run: clean
docker compose -p singhquant-staging down
```

With dummy keys the bots log 401 errors and never write a strategy heartbeat,
so `trading_bots` reports **unhealthy**. That is correct liveness behaviour,
not an image problem. A full functional run needs a **separate Alpaca paper
account** (Phase 3); never use production keys for staging.

Record the output of steps 2–4 with the deployment notes.
