---
name: python-architecture
description: >-
  Package layout, public exports, Toolbox/operator contracts, RNG, file-length
  limits, Apache headers, and coding conventions for the deap-er library. Use
  before writing or refactoring Python under deap_er/, tests/, or examples/,
  creating packages, or adding files.
---

# deap-er Python architecture

Single-package scientific library (`deap_er/`). There is no `backend/`, frontend, or service layer. `tools/dev` is sourced bootstrap only. `deap_er/tools.py` is the public operator barrel — not an app `tools/` tree.

Runtime deps: `numpy`, `scipy`, `dill`, `moocore`. Optional extra: `numba`.

## Layout

```
deap_er/
  base/              # Toolbox, Fitness
  creator/           # runtime type factory
  algorithms/        # ea_*, var_*; shared _loop.py
  operators/         # crossover, mutation, migration; selection/; _bounds.py
  strategies/        # CMA variants; _common.py
  records/           # logbook, statistics, hall of fame, history
  utilities/         # initializers, constraints, metrics; hypervolume/; sorting/
  benchmarks/        # test problems (bm_*)
  gp/                # genetic programming (own cx_/mut_/gen_*)
  persistence/       # Checkpoint (imported as deap_er.env)
  rng.py             # process-wide RNG; re-exported via tools and utilities
  typedefs.py        # re-exports base/, gp/, records/ typedefs
  tools.py           # explicit barrel: algorithms, operators, strategies, records, utilities, benchmarks
tests/               # pytest suite (Checkpoint lives in tests/test_controllers/)
examples/            # runnable scripts; snippet-included into docs
docs/                # MkDocs source — see skill mkdocs-docs
```

Public import surface:

```python
from deap_er import base, creator, tools, gp, env, typedefs
```

Root `__init__.py` binds `base`, `gp`, `env` (`persistence`), and `creator`. `tools` and `typedefs` are sibling modules. `env` is `deap_er.persistence` — do not invent a second persistence namespace.

## Placement

Match the existing package. Do not add `utils/`, `config/`, `models/`, `services/`, or `app/`. Split a module only for a real domain boundary, a circular import, or [file length](#file-length) — not because a file grew while still under the soft limit.

| Kind | Put it here |
|:-----|:------------|
| Sequence GA crossover / mutation / migration | `operators/` |
| Selection | `operators/selection/` |
| GP trees, primitives, tapes | `gp/` (`cx_one_point` exists in both `operators` and `gp` — do not merge) |
| Algorithm loop | `algorithms/` (reuse `_loop`) |
| CMA / ES | `strategies/` |
| Test problem | `benchmarks/` (`bm_*`) |
| Stats / HoF / history | `records/` |
| Init / metrics / constraints / HV | `utilities/` |
| Shared types | nearest `typedefs.py`; re-export from `deap_er/typedefs.py` if public |

## File length

Applies to every `.py` file under `deap_er/` and `tests/`. `examples/` are exempt. Count is the full file (header, blanks, comments, `__all__`).

| Limit | Lines | Meaning |
|:------|------:|:--------|
| Soft | 250 | Preferred maximum. Stay at or under it when a coherent split exists. |
| Hard | 270 | Absolute maximum. Anything over this must be split. |
| Split floor | 30 | Do not create a leftover module whose line count is ≤ 30. |

**Always in scope.** Plans must check these limits. Every change to library source or pytests must check and fix them — including when the user did not ask for a length refactor.

**≤ 250** — done. Do not split for length.

**251–270** — analyze whether the file can be split into two modules along a real domain boundary such that the **smallest** resulting file is **> 30** lines (and each child still follows Placement and Public API). If the only splits leave a piece ≤ 30 lines, the soft ceiling applies: leave the file as one module, permitted up to 270. If a coherent split exists with both sides > 30, split and re-export.

**> 270** — hard ceiling. Refactor now. Keep splitting recursively until every resulting module is ≤ 270 (prefer ≤ 250) and no child is ≤ 30 lines. If a two-way split would violate the floor, choose a different boundary or more than two modules — do not leave a file over 270.

After a split, wire public names through the package `__init__.py` (and `tools.py` when the symbol belongs on the barrel). Internal leftovers stay `_`-prefixed with `__all__: list[str] = []`.

## Public API

Package `__init__.py` files and `tools.py` use **explicit named imports** plus `__all__`. `ruff.toml` ignores F401 on `__init__.py`. Function modules declare `__all__`. Internal `_` modules (`_loop`, `_bounds`, `_common`) set `__all__: list[str] = []`.

New public symbol:

1. Define it and list it in the module `__all__`.
2. Re-export it by name in the package `__init__.py` and that `__all__`.
3. If it belongs on the toolbox barrel, also add it to `deap_er/tools.py` and its `__all__`.
4. Leave root `__init__.py` alone unless adding a new top-level public package.

Operators and algorithms are **module-level functions**, not classes or services. `_`-prefixed helpers in the same module are normal (`_slicer` in `operators/crossover.py`). Classes are for state: `Toolbox`, `Fitness`, `Checkpoint`, `Strategy*`, `HallOfFame`, `PrimitiveSet`, `SelNSGA3WithMemory`.

Do not flatten operators into classes, invent star-exports, or forbid module-level `_` names.

## Naming

This is a DEAP rewrite. Use snake_case prefixes: `cx_`, `mut_`, `sel_`, `ea_`, `bm_`, `init_`, `mig_`, `var_`, `gen_`. Parameters: `cx_prob`, `mut_prob`, `sel_count`, `contestants` — not DEAP's `cxpb`, `mutpb`, `k`, `tournsize`.

## Contracts

- **Crossover** — in-place; return `Mates` (`tuple[Individual, Individual]`).
- **Mutation** — in-place; return `Mutant` (a **1-tuple**). Callers unpack `(ind,) = toolbox.mutate(ind)`.
- **Selection** — return `list[Individual]`.
- **Benchmarks** — return a fitness tuple (symbolic regression may return `float`).
- **Algorithms** — `toolbox` first; return `EvoAlgoResult` `(population, logbook)`.
- **Toolbox aliases** algorithms expect: `mate`, `mutate`, `select`, `evaluate`, `clone`, `map`. Optional: `evaluate_batch`. Default `clone` is `deepcopy`; `clone_individual` is the fast path.
- **Fitness** — `weights > 0` maximize, `< 0` minimize. Evaluate writes `ind.fitness.values`. Clear with `del ind.fitness.values`.
- **RNG** — `from deap_er.rng import rng` then `rng.random()`, `rng.choice()`, … Never `import random` in library code. Seed with `seed()` / `tools.seed`. Checkpoint persists this generator.

Typing imports are named: `from deap_er.base.typedefs import Individual, Mates`. Cross-subpackage imports are absolute (`from deap_er.rng import rng`); same-package and `__init__.py` use relative named imports. `import numpy` (not `as np`) unless the file already aliases it.

Requires Python `>=3.12` (`type` aliases, PEP 695 generics). Follow the typing already used in the file.

## Imports

**Every import goes at the top of the module.** Ruff does not enforce this (`PLC0415` is not selected), so it is on you.

A function-level import is allowed only when a top-level one would not work or would cost every user something they did not ask for:

- **Circular imports** — a top-level import that Python cannot resolve. Prefer splitting the module; defer only when splitting is worse.
- **Optional dependencies** — anything outside the runtime set (`numpy`, `scipy`, `dill`, `moocore`) behind a `[project.optional-dependencies]` extra. Example: `import numba` in `gp/numba_ops.py`.
- **Import-time cost the package does not otherwise pay** — a heavy module used by one function, where hoisting it would slow `import deap_er` for everyone. Example: `from scipy.signal import ...` in `gp/window_ops.py`.

"It felt tidier", "it is only used once", and "it keeps the header short" are not reasons. Neither is a circular import you introduced by putting a helper in the wrong module.

Every deferred import carries a short comment saying which of the three reasons applies. Without that comment, hoist it.

## Creator and checkpoints

- `creator.create(...)` mutates the `creator` module at runtime. Tests create types and delete them in teardown. Do not replace this with Pydantic or static dataclasses.
- `persistence/checkpoint.py` serializes with **dill** (lambdas and creator types). Keep dill; do not switch to pickle or cloudpickle unless asked.

## New Python files

Library and test files copy this header verbatim. **Examples omit it** (`examples/` are docs fixtures).

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

Public classes and functions need Google-style docstrings (`Args:`, `Returns:`, `Raises:`). mkdocstrings uses `docstring_style: google`. Document the caller-facing contract, not internals. Do not repeat types already on the signature. Omit `Returns` when the function returns `None`. Ruff enforces this via `lint.pydocstyle.convention = "google"` (`D` rules). Module and package docstrings are not required (`D100` / `D104` ignored). Examples and tests skip `D` (`examples/**`, `tests/**`).

Examples import `from deap_er import base, creator, tools`, call `tools.seed(...)` at the top, and use `setup()` / `main()`. They are not a second package.

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
