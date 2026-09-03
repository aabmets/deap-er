#!/usr/bin/env bash
# Cursor preToolUse adapter: rewrite Shell commands via RTK.
# Reads Cursor hook JSON on stdin; always prints JSON and exits 0.

set +e

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
rewrite_cmd=${script_dir}/rewrite-cmd.sh
input=$(cat)

python3 -c '
import json, subprocess, sys

rewrite_cmd = sys.argv[1]
raw = sys.argv[2]
if not raw.strip():
    print("{}")
    sys.exit(0)

try:
    data = json.loads(raw)
except Exception:
    print("{}")
    sys.exit(0)

tool_input = data.get("tool_input")
if not isinstance(tool_input, dict):
    tool_input = {}
cmd = tool_input.get("command") or data.get("command") or ""
if not cmd:
    print("{}")
    sys.exit(0)

try:
    proc = subprocess.run(
        [rewrite_cmd, cmd],
        capture_output=True,
        text=True,
        timeout=5,
    )
except Exception:
    print("{}")
    sys.exit(0)

rewritten = (proc.stdout or "").strip()
if not rewritten or rewritten == cmd:
    print("{}")
    sys.exit(0)

updated = dict(tool_input)
updated["command"] = rewritten
print(json.dumps({"permission": "allow", "updated_input": updated}, separators=(",", ":")))
' "$rewrite_cmd" "$input"
exit 0
