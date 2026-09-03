---
name: install-allure
description: >-
  Install the project-local Allure CLI and open pytest HTML reports. Use when
  the user asks for Allure reports, allure generate/open/serve, or when
  node_modules/.bin/allure is missing.
---

# Allure CLI

`allure-pytest` (uv `dev` group) writes raw results. The HTML report needs the Allure CLI (`allure-commandline` in root `package.json`), installed into this repo’s `node_modules` via bun. Do not use a global `allure`.

## What lives where

| Piece | Location | Role |
|---|---|---|
| Adapter | `allure-pytest` via `uv sync --group dev` | Pytest writes `reports/allure-results/` (`--alluredir` in `pyproject.toml`) |
| Generate hook | `tests/harness/allure_report_plugin.py` (loaded from `tests/conftest.py`) | After the session, runs `allure generate` into `reports/allure-report/` |
| CLI | `allure-commandline` via `bun install` | `node_modules/.bin/allure` |
| HTML | `reports/allure-report/` | Generated output (gitignored under `reports/`) |

`node_modules/` is gitignored. Commit `package.json` and `bun.lock`. Java 8+ is required (`java -version`).

## Agent responsibility

When Allure HTML is needed:

1. Confirm Java: `java -version` (need 8+).
2. If `node_modules/.bin/allure` is missing: `source tools/dev` (or `bun install` if the rest of the bootstrap is already healthy). Do not `bun install -g` or `npm install`.
3. Invoke the project binary, not `PATH`. If `node` is missing: `bun ./node_modules/.bin/allure --version`. After `source tools/dev`, the `allure` shell function uses that binary.

## Generate and open

`uv run pytest` already generates HTML via the harness plugin (controller process only; xdist workers skip it). `--clean-alluredir` wipes `reports/allure-results/` at session start.

```bash
uv run pytest
allure
```

Bare `allure` (from `tools/dev`) serves `reports/allure-report/` on port 13001. On WSL it binds `0.0.0.0` and opens the WSL IP in the Windows browser. With arguments it forwards to the project CLI.

Manual generate (only if the hook did not run):

```bash
./node_modules/.bin/allure generate reports/allure-results --clean -o reports/allure-report
```

## Verification

| Step | Expected |
|---|---|
| `java -version` | 8+ |
| `test -f node_modules/.bin/allure` | exists |
| `./node_modules/.bin/allure --version` | 2.x (e.g. `2.43.0`) |
| `reports/allure-report/index.html` after pytest | exists |

## Related

- Tests: skill `python-pytest`
- Bootstrap: `source tools/dev`
