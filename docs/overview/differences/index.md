# Overview

This library started as a rewrite of
[DEAP](https://github.com/DEAP/deap) and kept that toolbox model —
register operators, run an algorithm. The published API is not a
drop-in rename: function names, parameter order, and a few contracts
changed, and the surface now includes families DEAP does not ship.
Counted from the pages in this section:

- **18** still-open [DEAP](https://github.com/DEAP/deap) issues
  [implemented](../../bugfixes/deap_fixes.md) (some older than a decade)
- **73** correctness bugs fixed — [operators](../../bugfixes/operators.md),
  [GP](../../bugfixes/gp.md),
  [CMA](../../bugfixes/strategies.md),
  [records](../../bugfixes/records.md),
  [checkpoints](../../bugfixes/persistence.md),
  published [benchmarks](../../bugfixes/benchmarks.md),
  [creator](../../bugfixes/creator.md),
  [utilities](../../bugfixes/utilities.md),
  and [algorithms](../../bugfixes/algorithms.md)
- **57** capabilities DEAP does not have, including boxed CMA,
  mixed-gene mutation, logbook JSON,
  [columnar GP](../../tutorials/columnar_gp.md), and
  [Push GP](../../tutorials/push_gp.md)

The rest of this section is that inventory: the rewrite itself (this
page), then the operators, bookkeeping, genetic programming,
quality-diversity archives, and correctness work that accumulated on
top of that base. Timed hot paths versus DEAP are on the
[performance page](../performance.md).

## Rewrite and public API

1. The codebase is a complete rewrite for Python 3.12 and newer. The
   license is Apache-2.0.
2. Algorithms, strategies, and benchmarks live under the **tools**
   namespace.
3. All **camelCase** functions and methods are **snake_case**.
4. Some functions and methods were renamed entirely.
5. The public surface has type hints.
6. Many parameters were reordered or renamed.
7. Some class properties are now method calls.
8. Hypervolume, hypervolume contributions, and Pareto ranking delegate
   to [moocore](https://pypi.org/project/moocore/) (prebuilt C wheels).
   deap-er stays a pure-Python package. moocore is licensed
   LGPL-2.1-or-later; do not vendor its sources into this tree.
9. **3.0 breaking changes:** `sort_log_non_dominated` and the
   `HyperVolume` class are removed. `sort_non_dominated` no longer
   takes `ffo`. `sel_nsga_2` / `sel_nsga_3` no longer take
   `sorting`. `least_contrib` no longer takes `map_func`.
   `StrategyMultiObjective` no longer takes `mp_pool`.
10. State persistence is implemented with the `Checkpoint` class.
11. Deprecated and obsolete DEAP APIs are removed.
12. The documentation matches the current API.
