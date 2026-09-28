#!/usr/bin/env bash
# Claude Code PreToolUse adapter: rewrite Bash commands via RTK.
# Delegates to `rtk hook claude`, which reads the hook JSON on stdin and
# prints an updatedInput rewrite (never auto-approves). Fails open when the
# binary is missing or too old; always exits 0.

set +e

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

"$rtk_bin" hook claude 2>/dev/null
exit 0
