#!/usr/bin/env bash
set -u

readonly REGION_ID="${REGION_ID:-901}"
readonly TAILSCALE_BIN="${TAILSCALE_BIN:-/usr/bin/tailscale}"

for _ in $(seq 1 24); do
    if "$TAILSCALE_BIN" status --json >/dev/null 2>&1; then
        "$TAILSCALE_BIN" debug force-prefer-derp "$REGION_ID" || exit 0
        "$TAILSCALE_BIN" debug rebind || true
        logger -t tailscale-autodl-derp "preferred DERP region ${REGION_ID} applied"
        exit 0
    fi
    sleep 5
done

logger -t tailscale-autodl-derp "tailscale local API was not ready; preference was not changed"
exit 0
