#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi

archive=${1:-/tmp/tailscale_arm64.tgz}
owner_user=${2:-unitree}
owner_key=${3:-}
developer_tool=${4:-}
wired_address=${WIRED_ADDRESS:-192.168.123.18}
wired_subnet=${WIRED_SUBNET:-192.168.123.0/24}

for path in "$archive"; do
  if [[ ! -f $path ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if ! id "$owner_user" >/dev/null 2>&1; then
  echo "Owner user does not exist: $owner_user" >&2
  exit 1
fi

install_dir=$(mktemp -d /tmp/go2-edge-install.XXXXXX)
cleanup() {
  rm -rf "$install_dir"
}
trap cleanup EXIT

tar -xzf "$archive" -C "$install_dir" --strip-components=1
for binary in tailscale tailscaled; do
  if [[ ! -x $install_dir/$binary ]]; then
    echo "Archive does not contain executable $binary" >&2
    exit 1
  fi
done

install -m 0755 "$install_dir/tailscale" /usr/bin/tailscale
install -m 0755 "$install_dir/tailscaled" /usr/sbin/tailscaled
install -m 0644 "$install_dir/systemd/tailscaled.service" /etc/systemd/system/tailscaled.service
install -m 0644 "$install_dir/systemd/tailscale-online.target" /etc/systemd/system/tailscale-online.target
install -m 0644 "$install_dir/systemd/tailscale-wait-online.service" /etc/systemd/system/tailscale-wait-online.service
install -m 0644 "$install_dir/systemd/tailscaled.defaults" /etc/default/tailscaled

getent group go2dev >/dev/null || groupadd go2dev
usermod -aG go2dev "$owner_user"
install -d -m 2770 -o "$owner_user" -g go2dev /home/"$owner_user"/edge-data
for directory in uploads projects models; do
  install -d -m 2770 -o "$owner_user" -g go2dev /home/"$owner_user"/edge-data/"$directory"
done
setfacl -R -m g:go2dev:rwx /home/"$owner_user"/edge-data
setfacl -R -d -m u::rwx,g::rwx,o::--- /home/"$owner_user"/edge-data

if [[ -n $owner_key ]]; then
  if [[ ! -f $owner_key ]] || ! ssh-keygen -l -f "$owner_key" >/dev/null 2>&1; then
    echo "Invalid owner public key: $owner_key" >&2
    exit 1
  fi
  install -d -m 0700 -o "$owner_user" -g "$owner_user" /home/"$owner_user"/.ssh
  touch /home/"$owner_user"/.ssh/authorized_keys
  chown "$owner_user:$owner_user" /home/"$owner_user"/.ssh/authorized_keys
  chmod 0600 /home/"$owner_user"/.ssh/authorized_keys
  key=$(<"$owner_key")
  grep -qxF "$key" /home/"$owner_user"/.ssh/authorized_keys || printf '%s\n' "$key" >> /home/"$owner_user"/.ssh/authorized_keys
fi

if [[ -n $developer_tool ]]; then
  install -m 0755 "$developer_tool" /usr/local/sbin/go2-developer
fi

install -d -m 0755 /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/60-go2-edge.conf <<EOF
# SSH access is key-only over every network path, including the wired
# maintenance network. This avoids configuration-dependent password exceptions.
KbdInteractiveAuthentication no
PubkeyAuthentication yes
PasswordAuthentication no
EOF

sshd -t
systemctl reload ssh
systemctl daemon-reload
systemctl enable --now tailscaled.service

mac=$(cat /sys/class/net/eth0/address 2>/dev/null || true)
suffix=$(printf '%s' "$mac" | tr -d ':' | tail -c 6)
if [[ -z $suffix ]]; then
  suffix=$(cut -c1-6 /etc/machine-id)
fi
node_name=go2-"$suffix"

echo "Installed edge-node services for $node_name."
echo "Open the following URL using the shared tailnet administrator account:"
tailscale up \
  --hostname="$node_name" \
  --operator="$owner_user" \
  --accept-dns=true \
  --accept-routes=false \
  --timeout=15s || true
