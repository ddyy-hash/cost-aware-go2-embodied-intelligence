#!/usr/bin/env bash
# Upload a finalized artifact directly to the PC Tailnet receiver.
#
# The filename is retained for release compatibility. Despite the historical
# name, this script must never use an SSH reverse tunnel for business data.
set -Eeuo pipefail

file=${1:?usage: upload_go2_segment_via_tunnel.sh FILE [RELATIVE_PATH]}
relative=${2:-$(basename "$file")}
receiver_url=${GO2_MAPPING_HIGH_FIDELITY_URL:-http://192.0.2.10:8000}
bulk_token=${GO2_MAPPING_BULK_AUTH_TOKEN:-}

[[ -f $file ]] || { echo "not a file: $file" >&2; exit 2; }
[[ $relative != /* && $relative != *..* ]] || { echo "unsafe relative path" >&2; exit 2; }
sha=$(sha256sum "$file" | awk '{print $1}')
auth_args=()
[[ -n $bulk_token ]] && auth_args=(-H "Authorization: Bearer $bulk_token")
curl --fail --retry 5 --retry-delay 3 \
  -X PUT "${auth_args[@]}" -H "X-SHA256:$sha" -H 'Expect:' --data-binary @"$file" \
  "$receiver_url/segments/$relative"
printf '\nUPLOAD sha256=%s path=%s\n' "$sha" "$relative"
