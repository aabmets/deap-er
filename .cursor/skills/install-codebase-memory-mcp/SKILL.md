---
name: install-codebase-memory-mcp
description: >-
  Wire a project-local codebase-memory-mcp server for this repo. Use when
  setting up or repairing MCP, mcp.json, or when graph tools are unavailable.
---

# Install codebase-memory MCP

[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) exposes a knowledge graph of **this** repo over MCP (CLI fallback if MCP cannot be wired).

Binary: `.bin/codebase-memory-mcp` (gitignored via `**/.bin`). Day-to-day use: skill `use-codebase-memory-mcp`.

## Agent responsibility

Before graph tools can run:

1. Confirm the binary: `.bin/codebase-memory-mcp --version`. If it is missing, still write `mcp.json` with the paths below, then continue without graph tools. Do not download binaries unless the user asked.
2. Write or merge `.cursor/mcp.json`. Keep unrelated servers. **`command` and `cwd` must be absolute** — Cursor CLI does not expand `${workspaceFolder}`.

```json
{
  "mcpServers": {
    "codebase-memory": {
      "command": "/ABS/REPO/.bin/codebase-memory-mcp",
      "cwd": "/ABS/REPO"
    }
  }
}
```

Replace `/ABS/REPO` with `git rev-parse --show-toplevel` at write time. On this machine that is `/mnt/projects/private/deap-er`.

3. Ask the user to restart or re-approve MCP if the harness requires it (config changes change approval hashes).

Prefer native MCP. CLI fallback is in skill `use-codebase-memory-mcp`.

## Verification

| Step | Expected |
|---|---|
| `.bin/codebase-memory-mcp --version` | Prints a version |
| `.cursor/mcp.json` | Absolute `command` and `cwd` for **this** repo |
| Cursor: `GetDynamicTools` / graph tools callable | After the user reconnects MCP |
| `list_projects` after `index_repository` | Project slug returned |

If the binary is absent, spawn fails with `ENOENT` until it lands in `.bin/`. That is expected — do not point `command` at another repository.

## Related

- Using the graph: skill `use-codebase-memory-mcp`
