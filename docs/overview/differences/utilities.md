# Constraints and utilities

1. `DeltaPenalty` and `ClosestValidPenalty` treat an `ndarray`
   `delta` or `distance` as a per-objective sequence. A 0-d array
   is broadcast; a 1-d vector is not passed to `itertools.repeat`.
2. `SortingNetwork.evaluate` copies each case, sorts the copy, and
   compares it to `sorted(original)`, so integer cases are not
   scored as bit-count patterns.
3. `case_errors` reduces aligned 1D `predicted` and `target` series
   into one mean-squared error per case segment. Accepts explicit
   half-open `(start, stop)` ranges or a boolean mask (one case per
   contiguous `True` run). Non-finite samples are skipped; optional
   `valid=` covers the `vwhere` warmup trap from columnar GP.
4. `duplicate_count` hashes a NumPy-array key by shape, dtype, and
   raw bytes. Equal ndarray individuals no longer raise `ValueError`
   from `sorted()` or list membership.
5. `nsga_convergence` and `nsga_diversity` read ``fitness.values``
   for reference individuals. A list genome is no longer treated as
   the objective vector when fitness is set.
6. `inv_gen_dist` uses the same fitness-first point extraction.
   Two fronts of individuals are no longer scored as gene lists.
7. `nsga_diversity` sorts the front by the first objective before
   Deb's $\Delta$ (a permutation of the same points no longer
   changes the value), and returns $1$ when the denominator is $0$
   (a single point, or several copies of one point with extremes
   at that point). It no longer raises `ZeroDivisionError`.
8. `SortingNetwork.draw` sizes the ASCII grid so empty and
   one-level networks no longer IndexError when writing wire
   labels or last-level spacers.
9. `spawn_rng(seed, worker_id)` and `map_spawned` give each
   mapped item an independent, seedable stream that does not
   collide with process-wide `tools.rng` and does not depend
   on pool scheduling ([user-provided streams][deap-75]). The
   parent generator stays checkpointable.

[deap-75]: https://github.com/DEAP/deap/issues/75
