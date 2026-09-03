# shellcheck disable=SC2155,SC2164,SC2001

# =====================
# Install third-party CLI tools and project binaries
# =====================
#
# Project binaries (GitHub releases → $BINARIES_DIR):
#   binary_name            project                           version  filename
#   rtk                    rtk-ai/rtk                        latest   rtk-x86_64-unknown-linux-musl.tar.gz
#   codebase-memory-mcp    DeusData/codebase-memory-mcp      latest   codebase-memory-mcp-linux-amd64-portable.tar.gz

download_github_binary() {
    local binary_name="$1"
    local project="$2"
    local version="$3"
    local filename="$4"
    local checksums_name="$5"
    local dest="${BINARIES_DIR}/${binary_name}"
    local tmp url checksums_url expected actual

    mkdir -p "$BINARIES_DIR"

    if [[ "$version" == "latest" ]]; then
        url="https://github.com/${project}/releases/latest/download/${filename}"
        checksums_url="https://github.com/${project}/releases/latest/download/${checksums_name}"
    else
        url="https://github.com/${project}/releases/download/${version}/${filename}"
        checksums_url="https://github.com/${project}/releases/download/${version}/${checksums_name}"
    fi

    tmp=$(mktemp -d) || return 1

    if ! curl -fL -o "${tmp}/${filename}" "$url"; then
        echo "Failed to download ${filename} from ${project}" >&2
        rm -rf "$tmp"
        return 1
    fi

    if ! curl -fL -o "${tmp}/${checksums_name}" "$checksums_url"; then
        echo "Failed to download ${checksums_name} from ${project}" >&2
        rm -rf "$tmp"
        return 1
    fi

    expected=$(awk -v n="$filename" '$2 == n || $2 == "*" n { print $1; exit }' "${tmp}/${checksums_name}")
    if [[ -z "$expected" ]]; then
        echo "No checksum for ${filename} in ${checksums_name}" >&2
        rm -rf "$tmp"
        return 1
    fi

    actual=$(sha256sum "${tmp}/${filename}" | awk '{print $1}')
    if [[ "$expected" != "$actual" ]]; then
        echo "Checksum mismatch for ${filename}" >&2
        rm -rf "$tmp"
        return 1
    fi

    case "$filename" in
        *.tar.gz|*.tgz)
            if ! tar --no-same-owner -xzf "${tmp}/${filename}" -C "$tmp"; then
                echo "Failed to extract ${filename}" >&2
                rm -rf "$tmp"
                return 1
            fi
            if [[ ! -f "${tmp}/${binary_name}" ]]; then
                echo "Archive ${filename} does not contain ${binary_name}" >&2
                rm -rf "$tmp"
                return 1
            fi
            mv -f "${tmp}/${binary_name}" "$dest"
            ;;
        *)
            mv -f "${tmp}/${filename}" "$dest"
            ;;
    esac

    chmod +x "$dest"
    rm -rf "$tmp"
}

ensure_cli_tools() {
    local name="$1"
    local dest="${BINARIES_DIR}/${name}"

    case "$name" in
        rtk|codebase-memory-mcp)
            if [[ -x "$dest" ]]; then
                return 0
            fi
            ;;
        *)
            if command -v "$name" >/dev/null 2>&1; then
                return 0
            fi
            ;;
    esac

    echo "Installing $name..." >&2

    case "$name" in
        "rtk")
            download_github_binary \
                "rtk" "rtk-ai/rtk" "latest" "rtk-x86_64-unknown-linux-musl.tar.gz" "checksums.txt" \
                || return $?
            if ! "$dest" --version >/dev/null 2>&1; then
                echo "Failed to verify rtk (--version)" >&2
                return 1
            fi
            if ! "$dest" gain >/dev/null 2>&1; then
                echo "Failed to verify rtk (gain). Wrong package if this fails." >&2
                return 1
            fi
            ;;
        "codebase-memory-mcp")
            download_github_binary \
                "codebase-memory-mcp" "DeusData/codebase-memory-mcp" "latest" \
                "codebase-memory-mcp-linux-amd64-portable.tar.gz" "checksums.txt" \
                || return $?
            if ! "$dest" --version >/dev/null 2>&1; then
                echo "Failed to verify codebase-memory-mcp (--version)" >&2
                return 1
            fi
            ;;
        "uv")
            curl -LsSf https://astral.sh/uv/install.sh | sh
            ;;
        "bun")
            curl -LsSf https://bun.sh/install | bash
            ;;
        "jq")
            INSTALLED_ANY_CLI=1
            sudo -v
            sudo DEBIAN_FRONTEND=noninteractive apt-get update && \
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y jq
            ;;
        "unzip")
            INSTALLED_ANY_CLI=1
            sudo -v
            sudo DEBIAN_FRONTEND=noninteractive apt-get update && \
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y unzip
            ;;
        "agent")
            curl https://cursor.com/install -fsS | bash
            ;;
        *)
            echo "Unknown CLI tool: $name" >&2
            return 1
            ;;
    esac

    case "$name" in
        rtk|codebase-memory-mcp)
            if [[ ! -x "$dest" ]]; then
                echo "Failed to install $name" >&2
                return 1
            fi
            ;;
        *)
            if ! command -v "$name" >/dev/null 2>&1; then
                echo "Failed to install $name" >&2
                return 1
            fi
            ;;
    esac
}

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required but was not found on PATH." >&2
    echo "Install curl with your package manager, then source tools/dev again." >&2
    return 1
fi

INSTALLED_ANY_CLI=0
mkdir -p "$BINARIES_DIR"

ensure_cli_tools "uv" || return $?
ensure_cli_tools "bun" || return $?
ensure_cli_tools "jq" || return $?
ensure_cli_tools "unzip" || return $?
ensure_cli_tools "agent" || return $?
ensure_cli_tools "rtk" || return $?
ensure_cli_tools "codebase-memory-mcp" || return $?

if [[ "$INSTALLED_ANY_CLI" -eq 1 ]]; then
    sudo apt-get autoremove -y > /dev/null 2>&1
    sudo apt-get autoclean -y > /dev/null 2>&1
fi

unset download_github_binary
unset ensure_cli_tools
unset INSTALLED_ANY_CLI
