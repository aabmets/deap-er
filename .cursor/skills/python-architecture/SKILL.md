---
name: python-architecture
description: >-
  Package layout, public exports, Apache headers, and coding conventions for the
  deap-er library. Use before writing or refactoring Python under deap_er/,
  tests/, or examples/, creating packages, or adding files.
---

# deap-er Python architecture

Single-package scientific library. There is no `backend/`, `tools/`, frontend, or service layer.

## Layout

```
deap_er/           # installable package
  base/            # Toolbox, Fitness
  creator/         # runtime type factory
  algorithms/      # ea_simple, mu+lambda, generate-update, variation
  operators/       # crossover, mutation, migration, selection/
  strategies/      # CMA variants
  records/         # logbook, statistics, hall of fame, history
  utilities/       # initializers, hypervolume, constraints, metrics
  benchmarks/      # test problems
  gp/              # genetic programming
  persistence/     # Checkpoint (imported as deap_er.env)
  dtypes.py        # shared aliases
  tools.py         # re-exports algorithms, operators, strategies, records, utilities, benchmarks
tests/             # pytest suite
examples/          # runnable scripts; also included into docs via snippets
docs/              # MkDocs source — see skill mkdocs-docs
```

Public import surface:

```python
from deap_er import base, creator, tools, gp, env, dtypes
```

`env` is `deap_er.persistence`. Do not invent a second persistence namespace.

## When adding files

Match the existing package. Do not add `utils/`, `config/`, `models/`, `services/`, or `app/`. Split a module only for a real domain boundary or a circular import — not because a file grew.

## Public API

- Package `__init__.py` files star-export (`from .crossover import *`). That is the public surface. `ruff.toml` ignores F401/F403 there.
- Function modules declare `__all__`.
- Operators and algorithms are **module-level functions** (`cx_one_point`, `sel_nsga_2`, `ea_simple`), not classes or services.
- `_`-prefixed helpers in the same module are normal (`_slicer` in `operators/crossover.py`).
- `from deap_er.base.dtypes import *` is the usual typing import inside the package.

Do not "clean up" star imports, flatten operators into classes, or forbid module-level `_` names.

## Creator and checkpoints

- `creator.create(...)` mutates the `creator` module at runtime. Tests create types and delete them in teardown. Do not replace this with Pydantic or static dataclasses.
- `persistence/checkpoint.py` serializes with **dill** (lambdas and creator types). Keep dill; do not switch to pickle or cloudpickle unless asked.

## New Python files

Copy this header verbatim:

```python
#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
```

Public classes and functions need Google-style docstrings (`Args:`, `Returns:`, `Raises:`). mkdocstrings reads them with `docstring_style: google`. Document the caller-facing contract, not internals. Do not repeat types that are already on the signature. Omit `Returns` when the function returns `None`. Ruff enforces this via `lint.pydocstyle.convention = "google"` (`D` rules). Module and package docstrings are not required (`D100` / `D104` ignored).

## Do not impose

- Pydantic models, typed settings objects, or application DI
- Proprietary license headers
- A ban on module-level `_` names or on functions in class-containing modules
- `backend/` vs `tools/` import boundaries
- `from __future__ import annotations` unless the file already uses it or forward refs need it

## Related

- After Python edits: skill `python-validation`
- Tests (when in scope): skill `python-pytest`
- Docs (when in scope): skill `mkdocs-docs`
