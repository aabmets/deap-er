# SonarCloud MCP — tool catalog

Read this when the short `SKILL.md` loop is not enough.
Live schemas win: `GetDynamicTools` namespace `sonarqube`. Names below match
https://docs.sonarsource.com/sonarqube-mcp-server/reference/tools

## Setup

- Endpoint: `https://api.sonarcloud.io/mcp` (US orgs: `https://api.sonarqube.us/mcp`).
- Auth headers: `Authorization: Bearer ${env:SONARQUBE_TOKEN}`, `SONARQUBE_ORG`.
- Hosted defaults to **read-only**. Heavy local analysis, Vortex, and IDE bridge tools are not registered.
- Reload Cursor MCP after env or `mcp.json` changes.

## Issue search

`search_sonar_issues_in_projects`

- `projectKeys`: array of keys.
- `issueStatuses`: `OPEN`, `CONFIRMED`, `FALSE_POSITIVE`, `ACCEPTED`, `FIXED`, `IN_SANDBOX`.
- `severities`: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `BLOCKER`.
- `impactSoftwareQualities`: `MAINTAINABILITY`, `RELIABILITY`, `SECURITY`.
- `issueKey`: one issue.
- `inNewCodePeriod`: needs exactly one of `projectKeys` / `files`.
- `pageIndex` (1-based), `pageSize` (≤ 500).
- `branch` or `pullRequest`, not both.

`show_rule` — `key` is the rule key from the issue.

## Hotspots, gates, coverage

| Tool | Notes |
|:-----|:------|
| `search_security_hotspots` | `status`: `TO_REVIEW` / `REVIEWED`. |
| `show_security_hotspot` | `hotspotKey`. |
| `get_project_quality_gate_status` | `projectKey`. |
| `search_files_by_coverage` | Worst coverage first. |
| `get_file_coverage_details` | Line-level; `key` is `projectKey:path`. |
| `get_component_measures` | e.g. `ncloc`, `violations`, `coverage`. |

## Project discovery

| Tool | Notes |
|:-----|:------|
| `search_my_sonarqube_projects` | `q` matches name (partial) or key (exact). |
| `list_branches` | `branchTypes`: `ALL` / `LONG` / `SHORT`. |
| `list_pull_requests` | Use returned key as `pullRequest`. |

## Writes (user must ask)

Hosted MCP is read-only unless `SONARQUBE_READ_ONLY=false` is sent as a header.

- `change_sonar_issue_status`: `accept` / `falsepositive` / `reopen`
- `change_security_hotspot_status`: `TO_REVIEW` / `REVIEWED` (+ `resolution` when reviewed)
