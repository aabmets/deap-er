# Agent Directives and Best Practices

This document contains **always-on** mandates for AI agents. Procedural playbooks live in [`.cursor/skills/`](.cursor/skills/) — read the relevant skill **before** starting work that needs it.

**deap-er** is a single-package scientific library (`deap_er/`) for evolutionary algorithms. There is no `backend/`, frontend, database, or devstack. `tools/dev` is developer bootstrap only — not a second installable package.

## 1. Core Principles

*   **KISS (Keep It Simple, Stupid):** Prioritize simple logic. Avoid premature optimization or convoluted architectures. If two solutions exist, choose the simpler one.
*   **DRY (Don't Repeat Yourself):** Abstract shared logic into reusable utilities. Do not prematurely abstract superficially similar code.
*   **The Forest and the Trees:** Maintain awareness of the public API and package layout when working on isolated features. Consider callers in `tests/` and `examples/`, and whether a change belongs on the documented surface (`base`, `creator`, `tools`, `gp`, `env`, `typedefs`).
*   **Library Developer Mindset:** Design interfaces to be intuitive, flexible, and robust against misuse. This *is* a library — prefer stable, toolbox-friendly functions over framework ceremony.

### Scope Discipline (Strict — No Scope Creep)

**Do exactly what the user asked for. Nothing more, unless they explicitly approve it first.**

Scope creep wastes time and review bandwidth. Agents must treat the user's request as a hard boundary, not a starting suggestion.

**Forbidden without explicit user permission:**

- Refactoring, renaming, or "cleaning up" code outside the requested change
- Touching unrelated files, modules, tests, docs, or configs
- Adding tests, helpers, or abstractions the user did not ask for
- Updating documentation because a skill or workspace rule *could* apply—ask first
- "While I'm here" fixes, DRY extractions, or consistency passes across the codebase
- Expanding a narrow task (e.g. "add a helper") into a multi-file feature delivery

**Required behavior:**

1. **Implement the minimum correct diff** that satisfies the literal request.
2. **If additional work seems advisable** (tests, docs sync, related callers, follow-up refactors), **stop and ask**—present options briefly; do not implement until the user says yes.
3. **Mention optional follow-ups in the reply** instead of silently doing them.
4. **When rules conflict**, the user's explicit instruction for *this task* wins over skill mandates (e.g. docs sync, tests, validation) until they approve the extra scope.

If unsure whether something is in scope, assume it is **out of scope** and ask.

## 2. Hard Boundaries (Always Apply)

These rules apply on every task regardless of skills.

### Single-package library

This repo is `deap_er/` + `tests/` + `examples/` + MkDocs `docs/`. `tools/dev` is sourced bootstrap (CLI helpers, host tools, `.bin/` binaries, uv/bun installs).

- Do not invent `backend/`, `frontend/`, `services/`, a second installable package, or a `tools/` app tree.
- Do not impose service/DI/Pydantic patterns. Operators and algorithms are **module-level functions**. Package `__init__.py` star-exports are the public surface.
- `creator.create` mutates types at runtime; `env.Checkpoint` serializes with **dill**. Do not "fix" either.
- Runtime dependencies are `numpy`, `scipy`, and `dill` unless the user asks to add one.

Full layout: read the [`python-architecture` skill](.cursor/skills/python-architecture/SKILL.md) before any Python edit under `deap_er/`, `tests/`, or `examples/` (mandatory — see [§3](#library-python-mandatory)).

### Type Safety

Follow the typing already used in the file you are editing (`deap_er.base.typedefs` and local aliases). Do not introduce Pydantic models or a parallel typed settings layer.

### Dependency Management

- Review installed dependencies before adding new ones.
- Prefer well-maintained, popular libraries with good DX.
- Do not use deprecated or abandoned packages.
- **Package manager:** `uv`. Avoid `pip` / `poetry` / `npm` when `uv` can do the job.

### Tool Usage

- **Hooks:** project `preToolUse` rewrites supported Shell commands via RTK (`.cursor/hooks.json` → `.cursor/hooks/rtk-pretooluse.sh`). Fails open until `.bin/rtk` exists. Never run `rtk init -g`. If the binary is missing, `source tools/dev`.
- **MCP:** `.cursor/mcp.json` registers **this** repo’s `.bin/codebase-memory-mcp` and the SonarCloud-hosted MCP (`https://api.sonarcloud.io/mcp`). Do not use another project’s codebase-memory server. Graph tools need `source tools/dev` if the binary is missing. SonarCloud tools need `SONARQUBE_TOKEN` and `SONARQUBE_ORG` in the shell that runs `agent` (`source tools/dev` calls `loadenv`; skill `sonarqube-mcp` creates `.env` keys for the user to fill in).
- **Structured tools first:** Use Read, Grep, Write, and StrReplace when they fit; Shell is for commands that need a real terminal environment.

## 3. Task Skills (Read Before Doing)

Before starting work, identify which skills apply. **Open and follow each relevant `SKILL.md` first** — do not rely on memory or grep alone.

### Library Python (Mandatory)

**Any task that creates, modifies, or refactors Python under `deap_er/`, `tests/`, or `examples/` must read and follow the [`python-architecture` skill](.cursor/skills/python-architecture/SKILL.md) before writing or editing code.**

This is required for every Python touch — including narrow one-function fixes, new modules, import changes, and test helpers. Do not skip it because the diff is small or the task looks trivial.

| Skill | Path | Read when |
|:------|:-----|:----------|
| **Codebase Memory MCP** | [`.cursor/skills/codebase-memory-mcp/SKILL.md`](.cursor/skills/codebase-memory-mcp/SKILL.md) | Exploring code structure, tracing calls, finding symbols, impact analysis |
| **SonarQube MCP** | [`.cursor/skills/sonarqube-mcp/SKILL.md`](.cursor/skills/sonarqube-mcp/SKILL.md) | Fetching SonarCloud issues or iteratively fixing findings |
| **Python architecture** | [`.cursor/skills/python-architecture/SKILL.md`](.cursor/skills/python-architecture/SKILL.md) | **Required** — any Python work under `deap_er/`, `tests/`, or `examples/` |
| **Pytest** | [`.cursor/skills/python-pytest/SKILL.md`](.cursor/skills/python-pytest/SKILL.md) | Writing or fixing tests — **only when tests are in scope** (see Scope Discipline) |
| **MkDocs docs** | [`.cursor/skills/mkdocs-docs/SKILL.md`](.cursor/skills/mkdocs-docs/SKILL.md) | Updating `docs/`, `mkdocs.yml`, or README — **only when docs are in scope** |
| **Validation** | [`.cursor/skills/python-validation/SKILL.md`](.cursor/skills/python-validation/SKILL.md) | After Python edits, before reporting a task complete (when validation is in scope) |
| **RTK** | [`.cursor/skills/rtk/SKILL.md`](.cursor/skills/rtk/SKILL.md) | Compact shell output; missing binary: `source tools/dev` |
| **Allure** | [`.cursor/skills/install-allure/SKILL.md`](.cursor/skills/install-allure/SKILL.md) | Project-local `node_modules/.bin/allure` via bun / `source tools/dev` |

### Typical Task → Skill Sequence

| User task | Skills to read (in order) |
|:----------|:--------------------------|
| Find how an operator or algorithm works | codebase-memory-mcp |
| Fix SonarQube / SonarCloud issues | sonarqube-mcp → **python-architecture** (if Python) → python-validation (if Python) |
| Implement a library feature (full request) | codebase-memory-mcp → **python-architecture** → python-pytest (if tests requested) → mkdocs-docs (if docs requested) → python-validation |
| Fix a bug with tests requested | codebase-memory-mcp → **python-architecture** → python-pytest → python-validation |
| Any Python edit (narrow or broad) | **python-architecture** (always) → other skills only when in scope for that task |
| Update tutorials / reference / Pages | **mkdocs-docs** (always when docs are in scope) |

**Indexed codebase graph project:** `mnt-projects-private-deap-er` (repo root `/mnt/projects/private/deap-er`). Session startup: see [`codebase-memory-mcp`](.cursor/skills/codebase-memory-mcp/SKILL.md). Missing `.bin/codebase-memory-mcp`: `source tools/dev`.

## 4. Finishing a Task

1. Confirm the diff matches the user's request (Scope Discipline).
2. Confirm Python edits comply with [`python-architecture`](.cursor/skills/python-architecture/SKILL.md) (read before coding; re-check before done).
3. Run skills that apply and were in scope (validation, tests, docs).
4. Do not mark work complete while lint, typecheck, or requested tests are failing.

When in doubt about scope, ask — do not expand the diff to satisfy a skill the user did not invoke.

Validation and test commands (repo root):

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
uv run pytest
uv run mkdocs build
```

Config lives in `ruff.toml`, `ty.toml`, and `pyproject.toml`. Docs publish via `.github/workflows/docs.yml` to https://aabmets.github.io/deap-er/ — Pages source is **GitHub Actions**, not branch `/docs`.
