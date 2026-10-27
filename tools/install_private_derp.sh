#!/usr/bin/env bash
# Install an uploaded derper binary as an authenticated, TLS-terminated service.
set -Eeuo pipefail

domain=${1:?usage: install_private_derp.sh DOMAIN DERPER_BINARY}
source_binary=${2:?usage: install_private_derp.sh DOMAIN DERPER_BINARY}
install_root=/opt/go2-private-derp
state_root=/var/lib/go2-private-derp
service_file=/etc/systemd/system/go2-private-derp.service
tailscale_socket=/var/run/tailscale/tailscaled.sock

[[ $domain =~ ^[A-Za-z0-9][A-Za-z0-9.-]{1,251}[A-Za-z0-9]$ ]] || { echo 'invalid DERP domain' >&2; exit 2; }
[[ -f $source_binary && -x $source_binary ]] || { echo "derper binary is missing or not executable: $source_binary" >&2; exit 2; }
[[ -S $tailscale_socket ]] || { echo "tailscaled socket is unavailable: $tailscale_socket" >&2; exit 3; }
command -v systemctl >/dev/null 2>&1 || { echo 'systemd is required' >&2; exit 3; }

install -d -m 0755 "$install_root"
install -d -m 0700 "$state_root" "$state_root/certs"
install -m 0755 "$source_binary" "$install_root/derper.new"
mv -f "$install_root/derper.new" "$install_root/derper"

temporary_service=$(mktemp "${service_file}.XXXXXX")
cleanup() { rm -f -- "$temporary_service"; }
trap cleanup EXIT
cat >"$temporary_service" <<EOF
[Unit]
Description=Go2 private Tailscale DERP relay
After=network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
ExecStart=$install_root/derper -hostname=$domain -a=:443 -http-port=80 -certmode=letsencrypt -certdir=$state_root/certs -verify-clients -socket=$tailscale_socket -stun=true -stun-port=3478
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$state_root
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
install -m 0644 "$temporary_service" "$service_file"
systemctl daemon-reload
systemctl enable --now go2-private-derp.service

for _ in $(seq 1 40); do
  systemctl is-active --quiet go2-private-derp.service && exit 0
  sleep 0.25
done
systemctl status --no-pager go2-private-derp.service >&2 || true
exit 4
