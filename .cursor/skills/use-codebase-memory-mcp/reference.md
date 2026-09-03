# codebase-memory — tool catalog

Read this only when the short `SKILL.md` mandates are not enough.
Setup: skill `install-codebase-memory-mcp`.

## Tools

| Tool | Notes |
|:-----|:------|
| `index_repository` | Requires `repo_path`. Once per session. Modes: `full` (default), `moderate`, `fast`, `cross-repo-intelligence`. |
| `index_status` | Staleness / node+edge counts. |
| `search_graph` | Nodes by name regex / `label` (`Class`, `Function`, `Method`, `File`, `Module`, `Variable`) / file pattern. |
| `search_code` | Text search; returns enclosing symbols and line numbers. |
| `get_code_snippet` | Requires full dotted `qualified_name` from `search_graph`. |
| `trace_path` | `function_name`, `direction` (`inbound`/`outbound`), `depth`. Tests excluded unless `include_tests: true`. |
| `get_architecture` | Overview by label. |
| `query_graph` | Read-only Cypher. Run `get_graph_schema` first (`file_path`, not `file`). |
| `detect_changes` | Uncommitted change impact. |
| `get_graph_schema` | Labels, relationships, property names. |
| `list_projects` | Indexed projects; no `project` param. |
| `manage_adr` | Architecture decision records. |
| `ingest_traces` | Runtime traces. |

## Typical sequences

- Find a symbol: `search_graph` → `get_code_snippet` on `qualified_name`.
- Who calls X: `trace_path` + `include_tests: true` if tests matter.
- Package or helper name: `search_code` → `get_code_snippet` on the owner.

## Graph UI

While the MCP server is up it may serve a UI at <http://localhost:9749>.
