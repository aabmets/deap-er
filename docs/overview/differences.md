# Important Differences

As this library is an *evolution* of the original [DEAP](https://github.com/DEAP/deap)
library (pun intended), there are a few important differences that the user should be aware of:

1. The whole codebase has been completely refactored for better usability and maintainability.
2. Algorithms, strategies and benchmarks have been moved into the **tools** namespace.
3. All **camelCase** functions and methods have been renamed to **snake_case**.
4. Some functions and/or methods have been completely renamed.
5. All functions and methods have received proper type hints.
6. The parameters of many functions have been reordered and/or renamed.
7. Some properties of classes have been changed into method calls.
8. Hypervolume, hypervolume contributions, and Pareto ranking delegate
   to [moocore](https://pypi.org/project/moocore/) (prebuilt C wheels).
   deap-er stays a pure-Python package. moocore is licensed
   LGPL-2.1-or-later; do not vendor its sources into this tree.
9. **3.0 breaking changes:** ``sort_log_non_dominated`` and the
   ``HyperVolume`` class are removed. ``sort_non_dominated`` no longer
   takes ``ffo``. ``sel_nsga_2`` / ``sel_nsga_3`` no longer take
   ``sorting``. ``least_contrib`` no longer takes ``map_func``.
   ``StrategyMultiObjective`` no longer takes ``mp_pool``.
10. State persistence has been implemented with the Checkpoint class.
11. All deprecated and obsolete code is removed.
12. The whole documentation has been reworked for better comprehensibility.
