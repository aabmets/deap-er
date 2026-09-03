# shellcheck disable=SC2155,SC2164,SC2001

# =====================
# Install Node dependencies (project-local Allure)
# =====================
ensure_node_dependencies() {
    local allure_bin="${PROJECT_DIR}/node_modules/.bin/allure"

    if [[ -f "$allure_bin" ]]; then
        return 0
    fi

    echo "Installing Node dependencies for deap-er..." >&2
    (
        cd "$PROJECT_DIR"
        if ! bun install; then
            echo "Failed to install Node dependencies for deap-er" >&2
            return 1
        fi
    ) || return $?

    if [[ ! -f "$allure_bin" ]]; then
        echo "Failed to install Allure CLI at '${allure_bin}'" >&2
        return 1
    fi
}

ensure_node_dependencies || return $?
unset ensure_node_dependencies
