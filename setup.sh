#!/bin/bash
# SinghQuant dependency install.
#
# Usage:
#   bash setup.sh                       # core paper-trading runtime + pytest (CPU-only, no torch)
#   bash setup.sh --with-rl             # + risky2 PPO: torch from the CPU wheel index, SB3, gymnasium
#   bash setup.sh --with-backtest       # + vectorbt
#   bash setup.sh --with-dashboard      # + streamlit
#   bash setup.sh --with-optional       # + google-genai macro fallback
#   bash setup.sh --all                 # everything above
#   bash setup.sh --pinned [--all]      # original exact pins (Python 3.11 only)
#
# Notes
# * alpaca-trade-api 3.0.0 pins websockets<11 and aiohttp==3.8.2, which conflict
#   with polygon-api-client and do not build on Python 3.12+. It is installed
#   WITHOUT its declared dependencies and the packages it actually imports are
#   installed loosely afterwards (same recipe as CI).
# * torch is always taken from https://download.pytorch.org/whl/cpu so a Linux
#   install never downloads the CUDA/NVIDIA wheels (several GB).
set -euo pipefail

PINNED=0; RL=0; BT=0; DASH=0; OPT=0
for arg in "$@"; do
  case "$arg" in
    --pinned) PINNED=1 ;;
    --with-rl) RL=1 ;;
    --with-backtest) BT=1 ;;
    --with-dashboard) DASH=1 ;;
    --with-optional) OPT=1 ;;
    --all) RL=1; BT=1; DASH=1; OPT=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

python -m pip install --upgrade pip

if [[ "$PINNED" == "1" ]]; then
  # The pinned file lists every component; torch is pre-installed from the CPU
  # index so the pin resolves to the +cpu build instead of the CUDA one.
  if [[ "$RL" == "1" ]]; then
    python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
  fi
  python -m pip install -r requirements-py311-pinned.txt
else
  python -m pip install -r requirements-core.txt -r requirements-dev.txt
  [[ "$RL" == "1" ]]   && python -m pip install -r requirements-rl.txt
  [[ "$BT" == "1" ]]   && python -m pip install -r requirements-backtest.txt
  [[ "$DASH" == "1" ]] && python -m pip install -r requirements-dashboard.txt
  [[ "$OPT" == "1" ]]  && python -m pip install -r requirements-optional.txt
fi

python -m pip install "alpaca-trade-api==3.0.0" --no-deps
python -m pip install msgpack websocket-client PyYAML deprecation aiohttp "websockets>=10"

echo "Dependencies installed (pinned=$PINNED rl=$RL backtest=$BT dashboard=$DASH optional=$OPT)."
echo "Next: create .env (see README), then run: python -m pytest tests/ -q"
