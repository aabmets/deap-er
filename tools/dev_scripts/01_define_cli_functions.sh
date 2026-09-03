# shellcheck disable=SC2155,SC2164,SC2001

# =====================
# Define CLI functions
# =====================

_run_allure() {
    local allure_bin="$1"
    shift

    if command -v node >/dev/null 2>&1; then
        "$allure_bin" "$@"
        return $?
    fi

    if command -v bun >/dev/null 2>&1; then
        bun "$allure_bin" "$@"
        return $?
    fi

    "$allure_bin" "$@"
}

pycov() {
    local report_path="$(pwd)/.htmlcov/index.html"
    if [[ -f "$report_path" ]]; then
        cmd.exe /c start "" "$(wslpath -w "$report_path")" 2>/dev/null
    else
        >&2 echo "ERROR: Coverage report not found at '${report_path}'."
        return 1
    fi
}

allure() {
    local allure_bin="${PROJECT_DIR}/node_modules/.bin/allure"

    if [[ ! -f "$allure_bin" ]]; then
        >&2 echo "ERROR: Allure CLI not found at '${allure_bin}'."
        >&2 echo "Re-source 'tools/dev' to install project-local Allure via bun."
        return 1
    fi

    if [[ $# -gt 0 ]]; then
        _run_allure "$allure_bin" "$@"
        return $?
    fi

    local port=13001
    local bind_host="127.0.0.1"
    local check_url="http://127.0.0.1:${port}"
    local browser_url="http://127.0.0.1:${port}"
    local report_dir="$(pwd)/reports/allure-report"
    local wsl_ip=""
    local i=0

    if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
        wsl_ip="$(hostname -I | awk '{print $1}')"
        if [[ -n "$wsl_ip" ]]; then
            # Windows cannot reach WSL's 127.0.0.1; bind all interfaces and open the WSL IP.
            bind_host="0.0.0.0"
            browser_url="http://${wsl_ip}:${port}"
        fi
    fi

    if [[ ! -d "$report_dir" ]]; then
        >&2 echo "ERROR: Allure report not found at '${report_dir}'. Generate it first."
        return 1
    fi

    if [[ ! -f "$report_dir/index.html" ]]; then
        >&2 echo "ERROR: '${report_dir}' does not look like a generated Allure report."
        return 1
    fi

    fuser -k "${port}/tcp" >/dev/null 2>&1 || true

    (
        set +m
        if command -v node >/dev/null 2>&1; then
            nohup "$allure_bin" open "$report_dir" --host "$bind_host" --port "$port" >/dev/null 2>&1 &
        elif command -v bun >/dev/null 2>&1; then
            nohup bun "$allure_bin" open "$report_dir" --host "$bind_host" --port "$port" >/dev/null 2>&1 &
        else
            nohup "$allure_bin" open "$report_dir" --host "$bind_host" --port "$port" >/dev/null 2>&1 &
        fi
    )

    while (( i < 40 )); do
        if curl -sf "$check_url" >/dev/null 2>&1; then
            cmd.exe /c start "" "$browser_url" >/dev/null 2>&1 || true
            echo "Allure report serving at ${browser_url}"
            return 0
        fi
        sleep 0.25
        ((i++))
    done

    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    >&2 echo "ERROR: Allure report server failed to start on ${browser_url}."
    return 1
}

codegraph() {
    (cd /mnt/c && cmd.exe /c start "http://localhost:9749")
}
