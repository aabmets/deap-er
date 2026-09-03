---
name: use-rtk
description: >-
  Use the RTK CLI proxy for compact shell output. Use when running git, pytest,
  grep, ls, find, or other RTK-supported commands, when shell output is large,
  or when compacting results before they reach the model.
---

# Use RTK

[RTK](https://github.com/rtk-ai/rtk) rewrites supported commands to `rtk <command>` so the model sees compact output.

Binary: `.bin/rtk`. Hook setup: skill `install-rtk`.

## Before using

1. Prefer the project hook (`.cursor/hooks.json` → `.cursor/hooks/rtk-pretooluse.sh`). If it is missing, follow `install-rtk`.
2. If `.bin/rtk` is not on `PATH` for a direct call:

```bash
export PATH="$(git rev-parse --show-toplevel)/.bin:$PATH"
```

3. Confirm: `rtk --version` and `rtk gain`. Wrong package if `gain` fails.

If the binary is not installed yet, run the original command. Do not block the task.

## How rewrite works

The hook intercepts Shell tool calls only. Built-in Read / Grep / Glob do not pass through it. For compact output there, use shell (`rg`, `ls`) or `rtk grep` / `rtk find` / `rtk read`.

Preview:

```bash
rtk rewrite "git status"   # prints "rtk git status"; exit 3 means rewritten
```

Empty stdout or exit 1 → leave the command unchanged. Successful rewrite **exits 3**.

## Last-resort prefix (no hook)

```bash
rewritten=$(rtk rewrite "$cmd" 2>/dev/null) || true
# run "$rewritten" if non-empty and different, else "$cmd"
```

## Related

- Wire hooks: skill `install-rtk`
- Structural exploration: skill `use-codebase-memory-mcp`
