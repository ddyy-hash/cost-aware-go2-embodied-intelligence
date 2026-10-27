#!/usr/bin/env bash
set -euo pipefail

action="${1:-status}"
root="${GO2_DERP_ROOT:-/root/autodl-tmp/go2-derp}"
bind_address="${GO2_DERP_BIND:-127.0.0.1:6009}"
gateway_address="${GO2_DERP_GATEWAY_BIND:-127.0.0.1:6006}"
bulk_address="${GO2_DERP_BULK_BIND:-127.0.0.1:6008}"
control_proxy="${GO2_DERP_CONTROL_PROXY:-http://127.0.0.1:17897}"
tailscaled="$root/bin/tailscaled"
tailscale="$root/bin/tailscale"
derper="$root/bin/derper"
python_bin="${GO2_DERP_PYTHON:-/root/miniconda3/bin/python}"
gateway="$root/tools/autodl_derp_bulk_gateway.py"
receiver="$root/tools/ssh_tunnel_receiver.py"
bulk_token="$root/state/bulk-auth-token"
bulk_root="$root/bulk-spool"
state_dir="$root/state"
log_dir="$root/logs"
tailscaled_socket="$state_dir/tailscaled.sock"
tailscaled_state="$state_dir/tailscaled.state"
tailscaled_pid="$state_dir/tailscaled.pid"
derper_pid="$state_dir/derper.pid"
gateway_pid="$state_dir/gateway.pid"
receiver_pid="$state_dir/bulk-receiver.pid"

mkdir -p "$state_dir" "$log_dir"

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "missing executable: $1" >&2
        exit 2
    fi
}

pid_is_live() {
    local pid_file="$1"
    local expected_executable="$2"
    local pid
    [[ -s "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' '\n' <"/proc/$pid/cmdline" | grep --fixed-strings --line-regexp --quiet "$expected_executable"
}

wait_for_file() {
    local path="$1"
    local attempts="${2:-40}"
    local attempt
    for ((attempt = 0; attempt < attempts; attempt++)); do
        [[ -S "$path" ]] && return 0
        sleep 0.25
    done
    return 1
}

owned_tailscaled_process_matches() {
    local pid="$1"
    local cmdline="/proc/$pid/cmdline"
    local argument
    local has_tun=0
    local has_state=0
    local has_socket=0
    local executable=""
    [[ "$pid" =~ ^[0-9]+$ && -r "$cmdline" ]] || return 1
    while IFS= read -r -d '' argument; do
        if [[ -z "$executable" ]]; then
            executable="$argument"
        elif [[ "$argument" == '--tun=userspace-networking' ]]; then
            has_tun=1
        elif [[ "$argument" == "--state=$tailscaled_state" ]]; then
            has_state=1
        elif [[ "$argument" == "--socket=$tailscaled_socket" ]]; then
            has_socket=1
        fi
    done <"$cmdline"
    [[ "$executable" == "$tailscaled" && $has_tun == 1 && $has_state == 1 && $has_socket == 1 ]]
}

terminate_owned_tailscaled_instances() {
    local proc_dir
    local pid
    local attempt
    local live
    local stale_pids=()
    for proc_dir in /proc/[0-9]*; do
        pid="${proc_dir##*/}"
        if owned_tailscaled_process_matches "$pid"; then
            stale_pids+=("$pid")
        fi
    done
    (( ${#stale_pids[@]} > 0 )) || return 0

    for pid in "${stale_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for ((attempt = 0; attempt < 50; attempt++)); do
        live=0
        for pid in "${stale_pids[@]}"; do
            if owned_tailscaled_process_matches "$pid"; then
                live=1
                break
            fi
        done
        (( live == 0 )) && return 0
        sleep 0.1
    done
    for pid in "${stale_pids[@]}"; do
        if owned_tailscaled_process_matches "$pid"; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

start_tailscaled() {
    require_executable "$tailscaled"
    require_executable "$tailscale"
    if [[ -S "$tailscaled_socket" ]] && "$tailscale" --socket="$tailscaled_socket" status >/dev/null 2>&1; then
        return 0
    fi

    # A failed authorization can leave an orphaned userspace daemon behind.
    # Replace only exact state/socket matches before creating a new singleton.
    terminate_owned_tailscaled_instances
    rm -f "$tailscaled_socket"
    HTTP_PROXY="$control_proxy" HTTPS_PROXY="$control_proxy" ALL_PROXY="$control_proxy" \
        nohup "$tailscaled" \
        --tun=userspace-networking \
        --state="$tailscaled_state" \
        --socket="$tailscaled_socket" \
        --port=0 \
        >"$log_dir/tailscaled.log" 2>&1 < /dev/null &
    echo "$!" >"$tailscaled_pid"
    if ! wait_for_file "$tailscaled_socket"; then
        echo "tailscaled socket did not appear; see $log_dir/tailscaled.log" >&2
        exit 3
    fi
    # Restoring an authorized node can take several seconds through the
    # control-plane proxy. A socket only proves that LocalAPI is listening.
    local attempt
    for ((attempt = 0; attempt < 60; attempt++)); do
        if "$tailscale" --socket="$tailscaled_socket" status >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo "tailscaled did not restore its authorized state; see $log_dir/tailscaled.log" >&2
    exit 4
}

start_derper() {
    require_executable "$derper"
    if pid_is_live "$derper_pid" "$derper" && curl --silent --fail --max-time 2 "http://$bind_address/" >/dev/null; then
        return 0
    fi

    if [[ -s "$derper_pid" ]] && pid_is_live "$derper_pid" "$derper"; then
        kill "$(<"$derper_pid")"
        sleep 1
    fi
    nohup "$derper" \
        -a "$bind_address" \
        -http-port=-1 \
        -stun=false \
        -verify-clients \
        -socket="$tailscaled_socket" \
        >"$log_dir/derper.log" 2>&1 < /dev/null &
    echo "$!" >"$derper_pid"

    local attempt
    for ((attempt = 0; attempt < 40; attempt++)); do
        if curl --silent --fail --max-time 2 "http://$bind_address/" >/dev/null; then
            return 0
        fi
        sleep 0.25
    done
    echo "DERP did not become healthy; see $log_dir/derper.log" >&2
    exit 5
}

start_bulk_receiver() {
    require_executable "$python_bin"
    [[ -r "$receiver" ]] || { echo "missing receiver: $receiver" >&2; exit 2; }
    mkdir -p "$bulk_root"
    if pid_is_live "$receiver_pid" "$python_bin" && curl --silent --fail --max-time 2 "http://$bulk_address/health" >/dev/null; then
        return 0
    fi
    if [[ -s "$receiver_pid" ]] && pid_is_live "$receiver_pid" "$python_bin"; then
        kill "$(<"$receiver_pid")"
        sleep 1
    fi
    nohup "$python_bin" "$receiver" \
        --root "$bulk_root" \
        --status-path "$bulk_root/.progressive-status.json" \
        --bind "${bulk_address%:*}" \
        --port "${bulk_address##*:}" \
        --allow-client 127.0.0.1 \
        >"$log_dir/bulk-receiver.log" 2>&1 < /dev/null &
    echo "$!" >"$receiver_pid"
    local attempt
    for ((attempt = 0; attempt < 40; attempt++)); do
        curl --silent --fail --max-time 2 "http://$bulk_address/health" >/dev/null && return 0
        sleep 0.25
    done
    echo "bulk receiver did not become healthy; see $log_dir/bulk-receiver.log" >&2
    exit 6
}

start_gateway() {
    require_executable "$python_bin"
    [[ -r "$gateway" ]] || { echo "missing gateway: $gateway" >&2; exit 2; }
    [[ -s "$bulk_token" ]] || { echo "missing bulk token: $bulk_token" >&2; exit 2; }
    chmod 0600 "$bulk_token"
    if pid_is_live "$gateway_pid" "$python_bin" && curl --silent --fail --max-time 2 \
        -H "Authorization: Bearer $(<"$bulk_token")" "http://$gateway_address/health" >/dev/null; then
        return 0
    fi
    if [[ -s "$gateway_pid" ]] && pid_is_live "$gateway_pid" "$python_bin"; then
        kill "$(<"$gateway_pid")"
        sleep 1
    fi
    nohup "$python_bin" "$gateway" \
        --bind "${gateway_address%:*}" \
        --listen-port "${gateway_address##*:}" \
        --derp-port "${bind_address##*:}" \
        --http-port "${bulk_address##*:}" \
        --token-file "$bulk_token" \
        >"$log_dir/gateway.log" 2>&1 < /dev/null &
    echo "$!" >"$gateway_pid"
    local attempt
    for ((attempt = 0; attempt < 40; attempt++)); do
        if curl --silent --fail --max-time 2 -H "Authorization: Bearer $(<"$bulk_token")" \
            "http://$gateway_address/health" >/dev/null; then
            return 0
        fi
        sleep 0.25
    done
    echo "DERP/bulk gateway did not become healthy; see $log_dir/gateway.log" >&2
    exit 7
}

print_status() {
    echo "GO2_AUTODL_DERP_STATUS"
    if [[ -S "$tailscaled_socket" ]]; then
        "$tailscale" --socket="$tailscaled_socket" status || true
    else
        echo "tailscaled=stopped"
    fi
    if pid_is_live "$derper_pid" "$derper"; then
        local root_code derp_code
        root_code="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 3 "http://$bind_address/" || true)"
        derp_code="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 3 "http://$bind_address/derp" || true)"
        echo "derper=running pid=$(<"$derper_pid") root_http=$root_code derp_without_upgrade_http=$derp_code"
        curl --silent --max-time 3 "http://$bind_address/debug/vars" \
            | grep -oE 'gauge_clients_(local|remote|total)": [0-9]+' || true
    else
        echo "derper=stopped"
    fi
    if pid_is_live "$receiver_pid" "$python_bin"; then
        echo "bulk_receiver=running pid=$(<"$receiver_pid")"
    else
        echo "bulk_receiver=stopped"
    fi
    if pid_is_live "$gateway_pid" "$python_bin"; then
        local bulk_code
        bulk_code="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 3 \
            -H "Authorization: Bearer $(<"$bulk_token")" "http://$gateway_address/health" || true)"
        echo "gateway=running pid=$(<"$gateway_pid") bulk_health_http=$bulk_code"
    else
        echo "gateway=stopped"
    fi
}

case "$action" in
    start)
        start_tailscaled
        start_derper
        start_bulk_receiver
        start_gateway
        print_status
        ;;
    status)
        print_status
        ;;
    stop)
        if pid_is_live "$gateway_pid" "$python_bin"; then kill "$(<"$gateway_pid")"; fi
        if pid_is_live "$receiver_pid" "$python_bin"; then kill "$(<"$receiver_pid")"; fi
        if pid_is_live "$derper_pid" "$derper"; then
            kill "$(<"$derper_pid")"
        fi
        rm -f "$derper_pid" "$gateway_pid" "$receiver_pid"
        print_status
        ;;
    stop-all)
        if pid_is_live "$gateway_pid" "$python_bin"; then kill "$(<"$gateway_pid")"; fi
        if pid_is_live "$receiver_pid" "$python_bin"; then kill "$(<"$receiver_pid")"; fi
        if pid_is_live "$derper_pid" "$derper"; then kill "$(<"$derper_pid")"; fi
        if pid_is_live "$tailscaled_pid" "$tailscaled"; then kill "$(<"$tailscaled_pid")"; fi
        rm -f "$derper_pid" "$gateway_pid" "$receiver_pid" "$tailscaled_pid" "$tailscaled_socket"
        print_status
        ;;
    *)
        echo "usage: $0 {start|status|stop|stop-all}" >&2
        exit 64
        ;;
esac
