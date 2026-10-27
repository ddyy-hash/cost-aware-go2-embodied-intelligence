#!/usr/bin/env bash
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$script_dir/bootstrap_wsl.sh"
status=$?
printf '\nGo2 mapping environment bootstrap exit code: %d\n' "$status"
exec bash
