# =====================
# Ensure Git config
# =====================
ensure_git_config() {
    if ! command -v git >/dev/null 2>&1; then
        echo "Error: git is not installed or not in PATH." >&2
        return 1
    fi

    local email name
    email=$(git config --local user.email)
    name=$(git config --local user.name)

    if [[ -z "$email" ]]; then
        echo "Git user.email is not set for this project."
        read -rp "Enter your email: " email
        git config --local user.email "$email"
        echo ""
    fi

    if [[ -z "$name" ]]; then
        echo "Git user.name is not set for this project."
        read -rp "Enter your name: " name
        git config --local user.name "$name"
        echo ""
    fi

    git config --local pull.rebase true
    git config --local fetch.prune true
    git config --local diff.colorMoved zebra
}

ensure_git_config || return $?
unset ensure_git_config
