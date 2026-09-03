---
name: python-pytest
description: >-
  Pytest layout, coverage threshold, mock policy, and commands for deap-er. Use
  when writing or fixing tests, when the user requests coverage, or when test
  work is in scope.
---

# deap-er pytest

Do not add or expand tests unless the user asked for tests, a bug fix with a reproduction, or explicitly approved test work.

## Commands

From the repo root:

```bash
uv run pytest
uv run pytest tests/test_base/test_toolbox.py -n0
uv run pytest tests/test_base/test_toolbox.py::TestToolbox::test_registration -n0
```

Default `addopts` in `pyproject.toml`: coverage on `deap_er`, xdist `worksteal` (max 8), Allure under `reports/`, HTML coverage under `.htmlcov/`. `session_timeout = 60`. Use `-n0` when isolating a single test.

## Layout

| Path | Role |
|---|---|
| `tests/test_base/` | Toolbox, Fitness |
| `tests/test_creator/` | creator factory and overrides |
| `tests/test_controllers/` | Checkpoint |
| `tests/test_gp/` | genetic programming |
| `tests/test_records/` | statistics / logbook |
| `tests/test_utilities/` | initializers, hypervolume |
| `tests/test_algorithms.py` | algorithm smoke tests |

`examples/` are docs fixtures (snippet-included), not the test suite.

## Conventions

- Apache header on new test files (same block as `deap_er`).
- Mix of classes (`TestToolbox`) and module-level functions is fine — match the file you are editing.
- `creator.create` types must be torn down (`del creator.__dict__[NAME]`) so tests do not leak types.
- Sync tests only. No `pytest-asyncio`, no `integ` marker, no devstack, no `@required_services`.

## Coverage and mocks

- Do not let project coverage drop below `fail_under = 80` in `pyproject.toml`.
- For new logic (when tests are in scope), cover the new code.
- Mock only external I/O (filesystem, process). Do not mock internals to force a pass.
- `pyproject.toml` ignores `DeprecationWarning`. Do not add broad warning filters.

## Related

- Lint/typecheck: skill `python-validation`
- Package conventions: skill `python-architecture`
