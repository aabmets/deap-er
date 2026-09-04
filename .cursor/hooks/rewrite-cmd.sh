#!/usr/bin/env bash
# Print the RTK rewrite of the given command, or nothing if unsupported.
# Always exits 0. `rtk rewrite` exits 3 on a successful rewrite — do not
# treat a non-zero exit as failure.
#
# Usage: rewrite-cmd.sh <command>
# Env:   RTK_BIN — absolute path to the rtk binary (optional)

set +e

cmd=$1
if [[ -z "$cmd" ]]; then
    exit 0
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)

if [[ -n "$RTK_BIN" && -x "$RTK_BIN" ]]; then
    rtk_bin=$RTK_BIN
elif [[ -x "${repo_root}/.bin/rtk" ]]; then
    rtk_bin=${repo_root}/.bin/rtk
elif [[ -x "${repo_root}/.bin/rtk.exe" ]]; then
    rtk_bin=${repo_root}/.bin/rtk.exe
else
    exit 0
fi

rtk_version=$("$rtk_bin" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [[ -n "$rtk_version" ]]; then
    major=$(echo "$rtk_version" | cut -d. -f1)
    minor=$(echo "$rtk_version" | cut -d. -f2)
    if [[ "$major" -eq 0 && "$minor" -lt 23 ]]; then
        exit 0
    fi
fi

rewritten=$("$rtk_bin" rewrite "$cmd" 2>/dev/null) || true
if [[ -n "$rewritten" && "$cmd" != "$rewritten" ]]; then
    printf '%s\n' "$rewritten"
fi
exit 0
