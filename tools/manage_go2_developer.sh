#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this command with sudo." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  go2-developer add USER PUBLIC_KEY_FILE
  go2-developer add-key USER PUBLIC_KEY_FILE
  go2-developer disable USER
  go2-developer list
EOF
}

validate_user() {
  if [[ ! $1 =~ ^[a-z_][a-z0-9_-]{0,30}$ ]]; then
    echo "Invalid Linux username: $1" >&2
    exit 1
  fi
}

install_key() {
  local username=$1
  local public_key_file=$2
  if [[ ! -f $public_key_file ]] || ! ssh-keygen -l -f "$public_key_file" >/dev/null 2>&1; then
    echo "Invalid SSH public key: $public_key_file" >&2
    exit 1
  fi

  local home_dir
  home_dir=$(getent passwd "$username" | cut -d: -f6)
  install -d -m 0700 -o "$username" -g "$username" "$home_dir/.ssh"
  touch "$home_dir/.ssh/authorized_keys"
  chown "$username:$username" "$home_dir/.ssh/authorized_keys"
  chmod 0600 "$home_dir/.ssh/authorized_keys"
  local key
  key=$(<"$public_key_file")
  grep -qxF "$key" "$home_dir/.ssh/authorized_keys" || printf '%s\n' "$key" >> "$home_dir/.ssh/authorized_keys"
}

action=${1:-}
case $action in
  add)
    username=${2:-}
    key_file=${3:-}
    validate_user "$username"
    if id "$username" >/dev/null 2>&1; then
      usermod -aG go2dev "$username"
      usermod --unlock --expiredate -1 "$username"
    else
      useradd --create-home --shell /bin/bash --groups go2dev "$username"
      random_password=$(openssl rand -base64 48)
      password_hash=$(printf '%s' "$random_password" | openssl passwd -6 -stdin)
      usermod --password "$password_hash" "$username"
    fi
    install_key "$username" "$key_file"
    echo "Developer $username is enabled with key-only remote access."
    ;;
  add-key)
    username=${2:-}
    key_file=${3:-}
    validate_user "$username"
    id "$username" >/dev/null
    install_key "$username" "$key_file"
    echo "Added key for $username."
    ;;
  disable)
    username=${2:-}
    validate_user "$username"
    if [[ $username == unitree ]]; then
      echo "Refusing to disable the maintenance owner account." >&2
      exit 1
    fi
    usermod --lock --expiredate 1 "$username"
    echo "Disabled $username; the home directory was preserved."
    ;;
  list)
    getent group go2dev | awk -F: '{print "go2dev members: " $4}'
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
