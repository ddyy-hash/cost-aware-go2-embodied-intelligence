#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ISAAC_SIM_ENV_NAME:-isaac-sim-py312}"
MICROMAMBA="${MICROMAMBA_BIN:-/home/a/micromamba/bin/micromamba}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$MICROMAMBA" create -y -n "$ENV_NAME" python=3.12 pip
"$MICROMAMBA" run -n "$ENV_NAME" python -m pip install \
  -r "$ROOT/mapping/pc/isaac_sim_requirements.txt"
printf 'ISAAC_SIM_WSL_READY env=%s version=' "$ENV_NAME"
"$MICROMAMBA" run -n "$ENV_NAME" python -c 'import importlib.metadata; print(importlib.metadata.version("isaacsim"))'
