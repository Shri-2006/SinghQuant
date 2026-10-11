# SinghQuant runtime image (multi-arch: linux/amd64 and linux/arm64 for Oracle Cloud Ampere).
# Debian bookworm is pinned so glibc (2.36) and therefore the manylinux wheels
# chosen are the same on every rebuild; every pinned dependency ships an
# aarch64 or pure-Python wheel (see docs/ARM64_VALIDATION.md), so no compiler
# is needed.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SINGHQUANT_DB_PATH=/app/data/trades.db \
    SINGHQUANT_MODEL_DIR=/app/model_artifacts
# PYTHONUNBUFFERED: print() reaches `docker logs` immediately.
# The database and models live on volumes (see docker-compose.yml), never in the image.

WORKDIR /app

COPY requirements*.txt setup.sh ./
# Pinned versions reproduce the original deployment; torch comes from the CPU
# wheel index; alpaca's runtime imports are pinned in requirements-alpaca-runtime.txt.
RUN bash setup.sh --pinned --with-rl

COPY . .

# Fails the build if anything does not import or the native wheels do not work on this CPU.
RUN mkdir -p /app/data /app/model_artifacts \
 && python -m tools.smoke_imports --require-rl

STOPSIGNAL SIGTERM
CMD ["python", "run.py"]
