#!/bin/bash
# SinghQuant dependency install.
#
# Usage:
#   bash setup.sh            # compatible ranges (requirements.txt)
#   bash setup.sh --pinned   # original exact pins for Python 3.11 (requirements-py311-pinned.txt)
#
# alpaca-trade-api 3.0.0 pins websockets<11 and aiohttp==3.8.2, which conflict
# with polygon-api-client and do not build on Python 3.12+. It is therefore
# installed WITHOUT its declared dependencies and the transitive packages it
# actually imports are installed loosely afterwards (same approach as CI).
set -euo pipefail

REQ="requirements.txt"
if [[ "${1:-}" == "--pinned" ]]; then
  REQ="requirements-py311-pinned.txt"
fi

python -m pip install --upgrade pip
python -m pip install -r "$REQ"
python -m pip install "alpaca-trade-api==3.0.0" --no-deps
python -m pip install msgpack websocket-client PyYAML deprecation aiohttp "websockets>=10"

echo "Dependencies installed from $REQ."
echo "Next: create .env (see README), then run: python -m pytest tests/ -q"
