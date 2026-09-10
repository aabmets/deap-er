---
name: sonarqube-mcp
description: >-
  Fetch SonarCloud analysis results and iteratively fix reported issues. Use
  when the user mentions SonarQube, SonarCloud, sonar issues, quality gates,
  security hotspots, coverage from Sonar, or asks to remediate findings from
  sonarcloud.io.
---

# SonarCloud MCP

Hosted server in `.cursor/mcp.json`: `https://api.sonarcloud.io/mcp` (no Docker, no local process). Cloud Agents also need that URL on `.cursor/environment.json` `mcpServerAllowlist`, plus `SONARQUBE_TOKEN` / `SONARQUBE_ORG` as environment secrets. There is no MCP dropdown: on [cursor.com/agents](https://cursor.com/agents), click **+** (left of the prompt, next to the model picker) → **MCP Servers** → **Add MCP**, then add the HTTP URL. Team admins can instead add it under Dashboard → Integrations & MCP.

If repo-root `.env` is missing or lacks `SONARQUBE_TOKEN` / `SONARQUBE_ORG`, create or update `.env` with those keys (empty values) and **stop**. Ask the user to fill them in (user token from https://sonarcloud.io/account/security, org key from SonarCloud), then re-run `source tools/dev` or `loadenv` before `agent`. Do not invent tokens. If `.env` already has other keys, only append the missing `SONARQUBE_*` lines.

`source tools/dev` calls `loadenv`, which creates `.env` if needed and sources it into the shell. Cursor CLI (`agent`) then inherits `${env:SONARQUBE_TOKEN}` and `${env:SONARQUBE_ORG}` for the hosted MCP.

If the `sonarqube` namespace is missing, `needsAuth`, or `error`, follow the `.env` step above (create keys if needed), then tell the user to fill the values, run `loadenv` or re-source `tools/dev`, and start `agent` again. Do not invent a REST client as a substitute.

Hosted Cloud exposes a **fixed, smaller toolset**. Local analysis, Vortex, IDE bridge, and Server `system` tools are unavailable.

Tool catalog: [reference.md](reference.md). Discover live schemas with `GetDynamicTools` on namespace `sonarqube` before calling.

## Mandates

1. **MCP first.** List issues, gates, hotspots, and measures through SonarCloud MCP tools. Do not scrape the Sonar UI.
2. **Resolve the project.** `search_my_sonarqube_projects` (`q` = `deap-er` or the repo name). Pass `projectKeys` / `projectKey` on later calls.
3. **Branch vs PR.** Omit both for the default branch. Use `list_branches` / `list_pull_requests` when the user names a branch or PR. Never pass a git branch name as `pullRequest`. Never pass both `branch` and `pullRequest`.
4. **Iterate locally.** Cloud issue lists do not change until the next Sonar scan. After each fix, re-check with `python-validation` (and tests only if the user asked). Do not call `analyze_code_snippet` — it is not on the hosted server.
5. **Do not close issues on the server** (`change_sonar_issue_status`, `change_security_hotspot_status`) unless the user explicitly asks. Hosted MCP is read-only by default.
6. **Scope.** Fix only the findings the user asked about. Follow `python-architecture` and `python-validation` for Python under `deap_er/`, `tests/`, or `examples/`.

## Fix loop

1. Discover tools: `GetDynamicTools` namespace `sonarqube`.
2. Snapshot: `get_project_quality_gate_status`, then `search_sonar_issues_in_projects` with `issueStatuses: ["OPEN", "CONFIRMED"]`. Paginate (`pageIndex` / `pageSize`, max 500). Also `search_security_hotspots` with `status: TO_REVIEW` when security is in scope.
3. Prioritize `BLOCKER` / `HIGH`, then `SECURITY`, then the rest. Call `show_rule` (and `show_security_hotspot` for hotspots) before editing.
4. Read the local file, apply the minimum correct fix, then run the validation skill.
5. Repeat on the next finding. Stop when the requested set is fixed locally, the remaining items are out of scope, or the user stops you.
6. Report: what was fixed, what still needs a new SonarCloud scan to clear on the server, and any findings you skipped (with rule key + reason).
