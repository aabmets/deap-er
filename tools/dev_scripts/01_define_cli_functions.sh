# shellcheck disable=SC2155,SC2164,SC2001

# =====================
# Define CLI functions
# =====================

_DEAP_ER_PROMPT_PREFIX='\[\033[1;35m\][deap-er]\[\033[0m\] '

_deap_er_enable_prompt() {
    if [[ -n "${_DEAP_ER_PROMPT_ACTIVE:-}" ]]; then
        return 0
    fi
    if [[ "${PS1-}" == "${_DEAP_ER_PROMPT_PREFIX}"* ]]; then
        _DEAP_ER_PROMPT_ACTIVE=1
        return 0
    fi
    _DEAP_ER_OLD_PS1="${PS1-}"
    PS1="${_DEAP_ER_PROMPT_PREFIX}${PS1-}"
    export PS1
    _DEAP_ER_PROMPT_ACTIVE=1
}

_deap_er_disable_prompt() {
    if [[ "${PS1-}" == "${_DEAP_ER_PROMPT_PREFIX}"* ]]; then
        PS1="${PS1#"$_DEAP_ER_PROMPT_PREFIX"}"
        export PS1
    elif [[ -n "${_DEAP_ER_OLD_PS1+_}" ]]; then
        PS1="$_DEAP_ER_OLD_PS1"
        export PS1
    fi
    unset _DEAP_ER_OLD_PS1
    unset _DEAP_ER_PROMPT_ACTIVE
    return 0
}

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
    local report_path="$(pwd)/reports/coverage-html/index.html"
    if [[ -f "$report_path" ]]; then
        cmd.exe /c start "" "$(wslpath -w "$report_path")" 2>/dev/null
    else
        >&2 echo "ERROR: Coverage report not found at '${report_path}'."
        return 1
    fi
}

rtfm() {
    local docs_path="$(pwd)/site/index.html"
    if [[ -f "$docs_path" ]]; then
        cmd.exe /c start "" "$(wslpath -w "$docs_path")" 2>/dev/null
    else
        >&2 echo "ERROR: Documentation site not found at '${docs_path}'."
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
    local check_url="http://127.0.0.1:${port}"  # NOSONAR localhost Allure report
    local browser_url="http://127.0.0.1:${port}"  # NOSONAR localhost Allure report
    local report_dir="$(pwd)/reports/allure-report"
    local wsl_ip=""
    local i=0

    if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
        wsl_ip="$(hostname -I | awk '{print $1}')"
        if [[ -n "$wsl_ip" ]]; then
            # Windows cannot reach WSL's 127.0.0.1; bind all interfaces and open the WSL IP.
            bind_host="0.0.0.0"
            browser_url="http://${wsl_ip}:${port}"  # NOSONAR localhost Allure report
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
        if curl -sf "$check_url" >/dev/null 2>&1; then  # NOSONAR localhost Allure report
            cmd.exe /c start "" "$browser_url" >/dev/null 2>&1 || true  # NOSONAR localhost Allure report
            echo "Allure report serving at ${browser_url}"  # NOSONAR localhost Allure report
            return 0
        fi
        sleep 0.25
        ((i++))
    done

    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    >&2 echo "ERROR: Allure report server failed to start on ${browser_url}."  # NOSONAR localhost Allure report
    return 1
}

codegraph() {
    (cd /mnt/c && cmd.exe /c start "http://localhost:9749")  # NOSONAR local codegraph UI
    return 0
}

loadenv() {
    local env_file="${PROJECT_DIR}/.env"

    if [[ ! -f "$env_file" ]]; then
        printf '%s\n' \
            'SONARQUBE_TOKEN=' \
            'SONARQUBE_ORG=' \
            > "$env_file"
        >&2 echo "Created ${env_file} — fill in SONARQUBE_TOKEN and SONARQUBE_ORG."
    fi

    set -a
    # shellcheck disable=SC1091
    source "$env_file"
    set +a
}

exitdev() {
    local active=0
    if declare -F deactivate >/dev/null 2>&1; then
        deactivate
        active=1
    fi
    if [[ -n "${_DEAP_ER_PROMPT_ACTIVE:-}" || "${PS1-}" == "${_DEAP_ER_PROMPT_PREFIX}"* ]]; then
        active=1
    fi
    _deap_er_disable_prompt
    if [[ "$active" -eq 0 ]]; then
        >&2 echo "ERROR: Dev environment is not active."
        return 1
    fi
}
