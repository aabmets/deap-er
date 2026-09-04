---
name: codebase-memory-mcp
description: >-
  Explore deap-er via the codebase-memory-mcp knowledge graph (MCP or CLI). Use
  when tracing call graphs, finding symbols, understanding package structure, or
  analyzing dependencies more efficiently than grep.
---

# codebase-memory MCP

Graph of **this** repo (`deap_er`, `tests`, `examples`, `docs`). Binary: `.bin/codebase-memory-mcp` (gitignored). Cursor config: `.cursor/mcp.json` (workspace-relative `command` and `cwd`). Bootstrap: `source tools/dev`. Tool catalog: [reference.md](reference.md).

If the binary is missing, `source tools/dev` or use Grep / Read / Glob. Do not call another project's server.

CLI fallback:

```bash
.bin/codebase-memory-mcp cli <tool> --flag value
.bin/codebase-memory-mcp cli <tool> --help
```

## Mandates

1. **Index once per session** before any graph query: `index_repository` with `repo_path` = absolute git root and `mode=full`, then `list_projects` + `index_status`. Re-index if counts look stale.
2. `repo_path` is the repo root (`/mnt/projects/private/deap-er` on this machine), never a subdirectory.
3. Pass `"project": "<slug>"` on every tool except `list_projects`. The slug is path-derived (e.g. `mnt-projects-private-deap-er`).
4. Graph first for structure. Read files you will edit; Grep for strings. Empty graph result ≠ missing — confirm with Grep.
5. Use `qualified_name` (e.g. `Toolbox.register`), never bare names. `trace_path` omits tests unless `include_tests: true`.
