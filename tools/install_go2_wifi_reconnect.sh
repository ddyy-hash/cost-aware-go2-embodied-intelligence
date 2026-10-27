#!/usr/bin/env bash
# Install a bounded NetworkManager reconnect watchdog for the Go2 Wi-Fi NIC.
# Run through sudo while wlan0 is connected to the intended saved profile.
set -Eeuo pipefail

interface=${GO2_WIFI_INTERFACE:-wlan0}
profile=${GO2_WIFI_CONNECTION:-}

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo 'run this installer through sudo' >&2
  exit 2
fi
command -v nmcli >/dev/null || { echo 'NetworkManager nmcli is required' >&2; exit 3; }
systemctl is-active --quiet NetworkManager || { echo 'NetworkManager is not active' >&2; exit 3; }
nmcli -t -f DEVICE,TYPE device status | grep -Fxq "$interface:wifi" || {
  echo "wireless interface not found: $interface" >&2
  exit 3
}

if [[ -z $profile ]]; then
  profile=$(nmcli -g GENERAL.CONNECTION device show "$interface" | head -n 1)
fi
if [[ -z $profile || $profile == '--' ]]; then
  echo "no active Wi-Fi profile on $interface; connect once before installing" >&2
  exit 4
fi

# A value of zero means retry indefinitely. Disable Wi-Fi power saving only
# for the selected profile so the Go2 is not left in a stale associated state.
nmcli connection modify "$profile" \
  connection.autoconnect yes \
  connection.autoconnect-retries 0 \
  connection.autoconnect-priority 100 \
  802-11-wireless.powersave 2

install -d -m 0755 /usr/local/sbin /etc/go2-wifi-reconnect
umask 077
printf 'INTERFACE=%q\nPROFILE=%q\n' "$interface" "$profile" > /etc/go2-wifi-reconnect/config
chmod 0600 /etc/go2-wifi-reconnect/config

cat > /usr/local/sbin/go2-wifi-reconnect <<'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail
source /etc/go2-wifi-reconnect/config

is_connected() {
  nmcli -g GENERAL.STATE device show "$INTERFACE" | grep -q '^100 '
}

if is_connected; then
  exit 0
fi

logger -t go2-wifi-reconnect "wlan disconnected; requesting NetworkManager recovery on $INTERFACE"
nmcli networking on || true
nmcli device set "$INTERFACE" managed yes || true
nmcli device wifi rescan ifname "$INTERFACE" || true

# Prefer the explicitly saved profile. The fallback lets NetworkManager select
# a suitable autoconnect profile if its profile UUID or name has changed.
timeout 25s nmcli connection up id "$PROFILE" ifname "$INTERFACE" || \
  timeout 25s nmcli device connect "$INTERFACE" || true
SCRIPT
chmod 0755 /usr/local/sbin/go2-wifi-reconnect

cat > /etc/systemd/system/go2-wifi-reconnect.service <<'UNIT'
[Unit]
Description=Go2 Wi-Fi reconnect guard
Wants=NetworkManager.service
After=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/go2-wifi-reconnect
UNIT

cat > /etc/systemd/system/go2-wifi-reconnect.timer <<'UNIT'
[Unit]
Description=Retry Go2 Wi-Fi only while disconnected

[Timer]
OnBootSec=20s
OnUnitActiveSec=20s
Persistent=true
Unit=go2-wifi-reconnect.service

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now go2-wifi-reconnect.timer
systemctl start go2-wifi-reconnect.service
systemctl is-active --quiet go2-wifi-reconnect.timer
printf 'GO2_WIFI_RECONNECT_INSTALLED interface=%s profile=%s retries=infinite interval_s=20\n' "$interface" "$profile"
