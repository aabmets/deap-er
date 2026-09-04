#!/usr/bin/env bash

BASHRC_FILE="$HOME/.bashrc"
if [[ ! -f "$BASHRC_FILE" ]]; then
    touch "$BASHRC_FILE"
fi

START_MARKER="# >>> DEAP-ER AUTOSOURCE START <<<"
END_MARKER="# >>> DEAP-ER AUTOSOURCE END <<<"
AUTOSOURCE_LOGIC=$(cat << 'EOF'
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
    README="${REPO_ROOT}/README.md"
    SCRIPT="${REPO_ROOT}/tools/dev"
    TITLE="DEAP-ER"

    if [[ -f "$README" ]] && grep -q "$TITLE" "$README"; then
        if [[ -f "$SCRIPT" ]]; then
            source "$SCRIPT"
        fi
    fi
fi
EOF
)

NEW_BLOCK=$(cat << EOF
$START_MARKER
$AUTOSOURCE_LOGIC
$END_MARKER
EOF
)

replace_bashrc_block() {
    local start="$1"
    local end="$2"
    local block="$3"
    local temp_bashrc

    temp_bashrc=$(mktemp)
    awk -v start="$start" -v end="$end" -v block="$block" '
        BEGIN { found=0 }
        $0 == start { found=1; print block; next }
        $0 == end { found=0; next }
        !found { print }
    ' "$BASHRC_FILE" > "$temp_bashrc"
    mv "$temp_bashrc" "$BASHRC_FILE"
    return 0
}

if grep -qF "$START_MARKER" "$BASHRC_FILE" && grep -qF "$END_MARKER" "$BASHRC_FILE"; then
    replace_bashrc_block "$START_MARKER" "$END_MARKER" "$NEW_BLOCK"
else
    echo "" >> "$BASHRC_FILE"
    echo "$NEW_BLOCK" >> "$BASHRC_FILE"
fi
