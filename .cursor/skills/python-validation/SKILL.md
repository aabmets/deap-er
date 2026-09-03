---
name: python-validation
description: >-
  Lint, format, and type-check commands after Python edits in deap-er. Use after
  modifying files under deap_er/, tests/, or examples/, before reporting the
  task complete.
---

# deap-er validation

Run from the **repo root** after Python edits the user asked you to finish.

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
```

Config is `ruff.toml` and `ty.toml` (`src` / `include` = `deap_er` and `tests`). Do not add a second Ruff or ty config. Do not `cd` into a subproject.

Fix errors the tools report. `ty.toml` sets `error-on-warning = false` — do not treat ty warnings as a hard gate unless the user asked.

If tests are in scope, run `uv run pytest` after this checklist (skill `python-pytest`).

If docs are in scope, `uv run mkdocs build` (skill `mkdocs-docs`).
