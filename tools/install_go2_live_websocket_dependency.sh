#!/usr/bin/env bash
# Installs the small pure-Python WebSocket client used by the live preview.
set -Eeuo pipefail

wheel=${1:-}

if python3 -c 'import websocket' >/dev/null 2>&1; then
  python3 -c 'import websocket; print("websocket-client already installed: %s" % getattr(websocket, "__version__", "unknown"))'
  exit 0
fi

python3 -m pip --version >/dev/null 2>&1 || {
  echo 'websocket-client install refused: python3 pip is unavailable' >&2
  exit 2
}

if [[ -n $wheel ]]; then
  [[ -f $wheel ]] || { echo "websocket-client wheel not found: $wheel" >&2; exit 2; }
  python3 -m pip install --user --no-index "$wheel"
else
  python3 -m pip install --user 'websocket-client==1.8.0'
fi

python3 -c 'import websocket; print("websocket-client installed: %s" % getattr(websocket, "__version__", "unknown"))'
