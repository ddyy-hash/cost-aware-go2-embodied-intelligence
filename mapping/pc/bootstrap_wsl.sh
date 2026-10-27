#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_file="$workspace_root/mapping/pc/environment.yml"
micromamba_root="${HOME}/micromamba"
micromamba_bin="$micromamba_root/bin/micromamba"

if [[ ! -x "$micromamba_bin" ]]; then
  archive=/tmp/micromamba-linux-64.tar.bz2
  mkdir -p "$micromamba_root"
  curl -fL --progress-bar https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$archive"
  tar -xjf "$archive" -C "$micromamba_root" bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$micromamba_root"
if "$micromamba_bin" env list | awk '{print $1}' | grep -qx go2-mapping-pc; then
  "$micromamba_bin" env update --prune -n go2-mapping-pc -f "$env_file"
else
  "$micromamba_bin" create --yes -f "$env_file"
fi
"$micromamba_bin" run -n go2-mapping-pc python "$workspace_root/mapping/pc/verify_environment.py"
