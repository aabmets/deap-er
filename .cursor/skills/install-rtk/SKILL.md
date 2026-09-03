---
name: install-rtk
description: >-
  Wire a project-local RTK rewrite hook for this repo. Use when setting up RTK,
  repairing hooks.json, or when Shell commands are not being rewritten to rtk.
---

# Install RTK hooks

[RTK](https://github.com/rtk-ai/rtk) compresses shell output. A `preToolUse` hook rewrites supported commands (`git status` → `rtk git status`).

Binary: `.bin/rtk` (gitignored via `**/.bin`). Day-to-day use: skill `use-rtk`.

## Do not use `rtk init -g`

`rtk init --agent cursor` writes **global** `~/.cursor/hooks.json` (and other vendor homes). Never run `rtk init -g` / `--global` / `--auto-patch`. Wire **this repo only**.

## Agent responsibility

Before the current Cursor session can auto-rewrite:

1. Confirm the binary: `.bin/rtk --version` and `.bin/rtk gain` (wrong package if `gain` fails — that is Rust Type Kit on crates.io).
2. If `.bin/rtk` is missing, stop after writing config. Do not download binaries unless the user asked.
3. Ensure project hooks exist (create or merge, do not wipe unrelated hooks):

`.cursor/hooks.json`:

```json
{
  "version": 1,
  "hooks": {
    "preToolUse": [
      {
        "command": ".cursor/hooks/rtk-pretooluse.sh",
        "matcher": "Shell"
      }
    ]
  }
}
```

Project hooks run from the repo root. Use that relative `command`. Do not use `${workspaceFolder}`.

Committed adapter: `.cursor/hooks/rtk-pretooluse.sh` (calls `.cursor/hooks/rewrite-cmd.sh`). Both **fail open** when the binary is missing.

4. Ask the user to restart or re-approve hooks if the harness requires it.

## Rewrite engine

`rtk rewrite` prints the rewritten command and **exits 3** on success. Hooks that treat any non-zero exit as failure drop every rewrite. Always use `|| true` and compare stdout. `rewrite-cmd.sh` already does this.

Lookup order in `rewrite-cmd.sh`: `$RTK_BIN`, then `<repo>/.bin/rtk`. No `PATH` fallback — a binary from another project must not rewrite this repo's commands.

## Verification

```bash
.bin/rtk --version
.bin/rtk gain
.bin/rtk rewrite "git status"; echo exit:$?
printf '%s' '{"tool_input":{"command":"git status"}}' | .cursor/hooks/rtk-pretooluse.sh
```

Expect `rtk git status` and exit 3 from `rtk rewrite`. The adapter should print JSON with `updated_input.command`. If the binary is absent, the adapter prints `{}` and the original command runs.

## Related

- Using RTK: skill `use-rtk`
- Graph exploration: skill `use-codebase-memory-mcp`
