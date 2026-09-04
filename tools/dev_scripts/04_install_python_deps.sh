# shellcheck disable=SC2155,SC2164,SC2001

# =====================
# Install Python dependencies
# =====================
ensure_python_dependencies() {
    local proj_dir="$PROJECT_DIR"
    local venv_dir="${proj_dir}/.venv"
    local created=0
    local needs_sync=0

    if [[ ! -d "$venv_dir" ]]; then
        local toml="${proj_dir}/pyproject.toml"
        local req_line=$(grep 'requires-python' "$toml" | head -1)
        local req_py=$(echo "$req_line" | sed -E 's/.*= *"(.*)".*/\1/')
        local clean_ver=$(echo "$req_py" | sed 's/[><=~^ ]//g')
        local py_ver=$(echo "$clean_ver" | cut -d. -f1,2)

        echo "Creating Python v${py_ver} venv for deap-er..." >&2
        if (cd "$proj_dir" && uv venv -p "$py_ver" > /dev/null 2>&1); then
            created=1
            needs_sync=1
        else
            echo "Failed to create Python venv for deap-er" >&2
            return 1
        fi
    fi

    if [[ ! -x "${venv_dir}/bin/pytest" ]]; then
        needs_sync=1
    fi

    if [[ $needs_sync -eq 1 ]]; then
        if [[ $created -eq 1 ]]; then
            echo "Installing Python dependencies for deap-er..."
        else
            echo "Repairing Python dependencies for deap-er..."
        fi
        (
            cd "$proj_dir"
            # shellcheck disable=SC1091
            source "${venv_dir}/bin/activate"
            if ! uv sync --active --group dev; then  # NOSONAR --no-build would skip this package
                echo "Failed to install Python dependencies for deap-er" >&2
                return 1
            fi
        ) || return $?
    fi
}

ensure_python_dependencies || return $?
unset ensure_python_dependencies
