#!/usr/bin/env bash
# Install the pinned Unitree SDK2 Python source and its CycloneDDS binding.
# This does not initialize DDS or create a publisher.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vendor="$release_root/mapping/robot/vendor"
sdk_archive="$vendor/unitree_sdk2_python-65691c8.tar.gz"
cyclone_archive="$vendor/cyclonedds-0.10.2.tar.gz"
typing_wheel="$vendor/typing_extensions-4.7.1-py3-none-any.whl"
cyclone_home=${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}

[[ -s "$sdk_archive" && -s "$cyclone_archive" && -s "$typing_wheel" ]] || {
  echo 'SDK2 Python vendor archives are missing' >&2
  exit 3
}
[[ $(id -un) == unitree ]] || { echo 'SDK2 Python installer must run as unitree' >&2; exit 5; }
[[ -d "$cyclone_home" ]] || { echo "CycloneDDS installation is missing: $cyclone_home" >&2; exit 6; }

export CYCLONEDDS_HOME="$cyclone_home"
python3 -m pip install --user --no-index --no-deps "$typing_wheel"
python3 -m pip install --user --no-index --no-deps --no-build-isolation "$cyclone_archive"
python3 -m pip install --user --no-index --no-deps --no-build-isolation "$sdk_archive"
# setup.py does not package the upstream CRC shared objects. Install the
# architecture-specific library from the verified source archive explicitly.
sdk_tmp=$(mktemp -d)
trap 'rm -rf -- "$sdk_tmp"' EXIT
tar -xzf "$sdk_archive" -C "$sdk_tmp" --strip-components=1
sdk_site=$(python3 -c 'import pathlib, unitree_sdk2py; print(pathlib.Path(unitree_sdk2py.__file__).parent)')
mkdir -p "$sdk_site/utils/lib"
install -m 0755 "$sdk_tmp/unitree_sdk2py/utils/lib/crc_aarch64.so" "$sdk_site/utils/lib/crc_aarch64.so"
python3 -c 'from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber; from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_'
printf 'GO2_SDK2PY_READY commit=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5 cyclonedds=0.10.2\n'
