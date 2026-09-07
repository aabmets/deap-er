# Important Differences

This library is an *evolution* of the original
[DEAP](https://github.com/DEAP/deap) library (pun intended). It is not
a drop-in rename. Function names, parameter order, and a few contracts
changed. Same toolbox model. Counted from the sections below:

- **18** still-open [DEAP](https://github.com/DEAP/deap) issues
  [implemented](../bugfixes/deap_fixes.md) (some older than a decade)
- **36** correctness bugs fixed — [operators](../bugfixes/operators.md),
  [GP](../bugfixes/gp.md),
  [CMA](../bugfixes/strategies.md),
  [records](../bugfixes/records.md),
  [checkpoints](../bugfixes/persistence.md),
  published [benchmarks](../bugfixes/benchmarks.md),
  [creator](../bugfixes/creator.md),
  [utilities](../bugfixes/utilities.md),
  and [algorithms](../bugfixes/algorithms.md)
- **33** capabilities DEAP does not have, including boxed CMA,
  mixed-gene mutation, logbook JSON, and
  [columnar GP](../tutorials/columnar_gp.md)

The rest of this page is that inventory: the rewrite itself, then the
operators, bookkeeping, genetic programming, and correctness work that
accumulated on top of that base. Timed hot paths versus DEAP are on
the [performance page](performance.md).

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

## Operators and selection

These items were open on the DEAP tracker, or were defects inherited
from the original sources.

1. [`mut_polynomial_bounded`][deap-655] clamps each gene into
   `[low, up]` before the Deb powers and requires `eta > 0`, so an
   out-of-box gene no longer writes NaN or a complex value.
2. [`cx_simulated_binary_bounded`][deap-740] uses the same clamp and
   `eta` guard. Out-of-box parents no longer raise `TypeError` on
   complex arithmetic. The second child uses the opposite sign of
   $\beta_q$, matching Deb / DEAP (`c2` is the upper child, not a
   second lower child).
3. [`cx_blend_bounded`][deap-527] is the boxed form of blend
   crossover: the same $\gamma$ draw as `cx_blend`, then a clamp on
   each child gene.
4. [`cx_partially_matched`][deap-472], uniform PMX, and
   `cx_ordered` map alleles by value. Permutations of named cities or
   other non-`{0..n-1}` encodings no longer IndexError or silently
   write the wrong genes.
5. [`mut_heterogeneous`][deap-755] applies one mutator per gene, so a
   mixed encoding (bit + int range + choice) does not need a
   one-off mutator.
6. [`assign_crowding_dist`][deap-321] can crowd on `wvalues` via
   `use_weights=True`. The default, and `sel_nsga_2`, still use raw
   `values`.
7. [`sel_tournament_dcd`][deap-641] accepts any
   `1 ≤ k ≤ len(individuals)`. When `k` is a multiple of 4 the
   original paired-shuffle path is used; other counts run pairwise
   contests until `k` winners are collected. `k ≤ 0` returns an empty
   list.
8. `sel_roulette` and `sel_stochastic_universal_sampling` spin the
   wheel on `wvalues[0]`. A negative floor is shifted so a
   minimization weight still has a positive slice. When every slice
   is zero the draw is uniform instead of an empty list. An empty
   pool or `sel_count ≤ 0` returns `[]`.
9. `mig_ring` evicts by object identity. Two individuals with equal
   genes are no longer treated as the same slot. Emigrants are cloned
   when a replacement operator is set, so the source deme is not
   aliased into the destination. Duplicate draws take the next unused
   index instead of writing one vacancy twice.
10. `cx_messy_one_point` cuts each parent independently, so lengths
    can change. Equal-length parents no longer degenerate into a
    shared-interval two-point swap.
11. Sequence crossovers copy slices before assignment. NumPy views
    are not aliased, so a one-point or uniform swap does not destroy
    a parent. `cx_es_two_point_copy` applies the same copy to the
    strategy vector.
12. `sel_spea_2` uses the full distance row for density, with the
    self-distance set to infinity, so the $k$-th neighbour is not the
    zero pad of the upper triangle. An empty pool returns `[]`.
13. `sel_nsga_3` intercepts on the success path are $1/x + \mathrm{best}$.
    Association treats a near-zero $\mathrm{intercepts} - \mathrm{best}$
    gap as $1$ so the niche distance is not NaN. Niching stops when
    the last front is exhausted, so a $k$ larger than the pool does
    not loop forever.
14. `cx_one_point`, `cx_two_point`, `cx_ordered`, and
    `mut_shuffle_indexes` no-op when a parent is shorter than two
    genes, so a length-1 individual no longer hits an empty `randint`
    interval or a two-cut `sample` on a one-gene permutation.
15. `sel_lexicase` and `sel_epsilon_lexicase` accept `cases=` to
    filter on a per-generation subset of fitness indices. Defaults
    still use every case. `sample_informed_cases` builds that subset
    by farthest-first traversal of Hamming distances between case
    solve vectors, so synonymous cases are not over-sampled.
    `fitness_case_matrix` packs `fitness.values` into a dense matrix;
    optional `matrix=` and `trust_matrix=` let lexicase and informed
    down-sampling reuse one pack per generation.
16. `sel_sms_emoa` reduces a pool by non-dominated sorting, then
    removes the least hypervolume contributor on the critical front
    until the quota is met. Optional `ref_point` follows the same
    minimization-space convention as `hypervolume` and `least_contrib`.
    Works on `parents + offspring` or steady-state `parents + [child]`.
17. `sel_moead` and `SelMOEADWithMemory` pick one winner per
    decomposition weight from `uniform_reference_points`, using
    Tchebycheff or PBI scalarization with Pareto-rank-aware tie
    breaks and NSGA-II-style crowding on the fill pass.
18. `sel_age_moea_2` and `SelAGE2WithMemory` advance front by front:
    geodesic diversity on partial $F_1$, inverse Minkowski on later
    partial fronts, with Newton–Raphson curvature on the first front.

## Evolution strategies

1. Every CMA strategy accepts `low` / `up` (scalar or length-`dim`)
   and a `bound_mode` of `"clip"` or `"resample"`, covering the
   [box-constrained CMA][deap-500] request. Bounds are applied to the
   sampled vector before `ind_init`. Resample rejects the whole
   vector; after `resample_limit` failed draws the offspring is
   clipped so `generate` always returns `lamb` individuals. Both
   modes are constraint-handling approximations: the CMA update then
   treats the repaired point as the sample.
2. MO-CMA's rank-one covariance update gates on $\lVert w \rVert$,
   not on `w.max()`. A negative evolution path no longer skips the
   update. When $w \approx 0$ the factors still scale by
   $\sqrt{\alpha}$.
3. The stall-case $\alpha$ on MO-CMA includes the $c_{\mathrm{cov}}$
   factor, so repeated stall generations do not inflate the
   covariance.
4. MO-CMA writes one step-size trial per parent when
   $\lambda \neq \mu$, so several children of the same parent do not
   stack $\sigma$ updates. `generate` samples every parent if any
   parent fitness is invalid; `update` ranks only valid fitnesses.
5. `RestartStrategy` wraps standard, $(1+\lambda)$, or MO-CMA with
   IPOP or BIPOP restart scheduling. `ea_generate_update_restarts`
   runs the generate/update loop and logs restart regime, population
   size, and evaluation budget each generation.

## Genetic programming

Prefix-tree GP (loosely typed, strongly typed, ADFs) is still there.
The following is extra.

1. `generate()` closes a type that has terminals but no primitives.
   A leaf-only type — a rolling window length is the usual case —
   can appear in a strongly typed tree.
2. `add_primitive(..., weight=)` implements
   [weighted primitive sampling][deap-383]. Equal weights keep the
   previous RNG stream. The same weights apply to node replacement
   and insert mutation. Terminals stay uniform.
3. A zero-arity callable terminal can [format as `name()`][deap-644]
   when `add_terminal(..., call_zero=True)`, so the default `eval`
   compile path calls it instead of looking up the function object.
   Action terminals (Santa Fe ant) stay uncalled names by default.
4. `tree_to_infix` is an [infix pretty-printer][deap-24] for logs and
   papers. It is display-only.
5. `make_column_pset(names)` builds a strongly typed set with one
   `Array` input per column. The type tags are `Array` (1D
   `float64`), `Mask` (1D `bool`), and `Window` (`int`). Argument
   order is column order.
6. `add_numpy_primitives` registers a vectorized kit: arithmetic,
   protected `vdiv` / `vlog` / `vsqrt`, comparisons that produce a
   `Mask`, mask logic, and `vwhere`. Protected ops only replace a
   non-finite result that the operation itself fabricated; an input
   `nan` comes back out as `nan`.
7. `add_window_primitives` registers causal `delay`, `diff`,
   `rolling_{sum,mean,std,min,max}`, and `ema`. The window is
   `[t-n+1, t]`; samples without enough history are `nan`. Look-ahead
   is forbidden. `add_window_ephemeral` samples inclusive integer
   lengths.
8. `compile_tree` caches the default `eval` backend by expression
   text and context identity. `backend="opcode"` lowers the tree to
   a postfix tape and runs a NumPy stack machine. `backend="numba"`
   (the `deap-er[numba]` extra) runs the same tape in one
   process-wide compiled interpreter. Custom kernels bind at or above
   `USER_BASE` and pass one dispatcher; they are not a second
   interpreter.
9. Algorithms call `toolbox.evaluate_batch(invalids)` when that
   operator is registered, otherwise `toolbox.map(toolbox.evaluate,
   invalids)`. A generation can be scored against one shared matrix
   without changing `map`'s contract.
10. `tools.clone_individual` shallow-copies a list/array individual
    and deepcopies only the fitness. GP toolboxes should register it;
    the default Toolbox clone remains `deepcopy`.
11. `cx_semantic` builds each child from a snapshot of the original
    parents. The second child is no longer derived from the already
    mutated first child.
12. `cx_one_point` always groups nodes by return type. An `object`
    root on the first parent no longer disables strongly typed
    matching.
13. `static_limit` replaces an oversized offspring with a
    `clone_individual` copy of a parent, so the two offspring slots
    never share one parent object.
14. HARM places the size cutoff on evaluated individuals only, scales
    the half-life by the cutoff (not by each individual's size), and
    does not crash on an empty candidate slice.
15. `add_primitive` and `add_terminal` reject a name that matches a
    primitive-set argument, so a compiled lambda parameter cannot
    shadow the symbol. `rename_arguments` rejects the inverse: a new
    name that is already an argument, primitive, or terminal.
16. `mut_insert` leaves the tree unchanged when a sibling type has
    no terminals, instead of raising `IndexError`.
17. `add_pair_window_primitives` registers causal `rolling_corr`,
    `rolling_cov`, and `rolling_beta` over two `Array` arguments and
    a `Window`. Moments use the population divisor; beta is the OLS
    slope of the first series on the second. Python, opcode, and
    Numba paths agree.
18. `add_ts_primitives` registers causal `ts_rank`, `ts_argmax`, and
    `ts_argmin`. Rank is the average rank of the current sample
    scaled to $[0, 1]$; a window of 1 is `nan`. Arg-extremum is how
    many samples ago the extreme occurred (`0` is now); a tie keeps
    the most recent. Python, opcode, and Numba paths agree.
19. `interpret_tapes` scores many tapes against one packed
    `(rows, columns)` matrix and returns `(n_individuals, n_rows)`.
    The opcode path unpacks columns once. The Numba path is a
    compiled loop; `parallel=True` gives each thread its own
    workspace. Unique programs are compiled once by `str(tree)`
    and lowered from the tree object.
20. `SlimTree` stores a GP head plus semantic delta blocks. `mut_slim`,
    `mut_slim_inflate`, and `mut_slim_deflate` append or remove deltas
    without re-wrapping the whole tree; `cx_slim_donor` swaps a donor
    block size-preservingly. `compile_slim_tree` evaluates
    $\mathrm{head} + \sum \delta_i$.
21. `PrimitiveTree` slice assignment treats a missing start as `0`.
    `tree[:]` and `tree[:n]` no longer raise `TypeError` when the
    replacement is a complete tree.

The columnar contract is in the
[columnar GP tutorial](../tutorials/columnar_gp.md). Shared-array
evaluation is in the
[multiprocessing tutorial](../tutorials/multiprocessing.md).

## Records, statistics, and algorithms

1. `MultiStatistics.register(..., chapters=)` can target a subset of
   chapters, so fitness min/max and a size statistic need not share
   one function ([DEAP#720][deap-720]).
2. `MultiStatistics.compile` materializes the input once. A
   generator, `map`, or `zip` is no longer exhausted by the first
   chapter.
3. An empty `Logbook` with a `header` [prints that header][deap-694]
   from `stream()` instead of the word “empty”.
4. `Logbook.to_json` / `from_json` [round-trip][deap-121] entries,
   nested chapters, and the header. NumPy scalars become Python
   numbers.
5. `duplicate_count` is a [variety statistic][deap-350]: population
   length minus distinct keys. Hashable keys scan in linear time;
   unhashable but sortable keys use an adjacent-run count after sort.
6. The four `ea_*` algorithms accept `log_time` ([per-generation
   wall time][deap-426]), `logger` ([instead of only
   `print`][deap-750]), and `fronts` (append a new `ParetoFront` of
   that generation's survivors, [not the cumulative
   hall of fame][deap-735]).
7. `HallOfFame.update` inserts into an empty archive, no-ops at
   `maxsize=0`, and replaces a similar member when the new individual
   is strictly better.
8. `HallOfFame.remove` raises `IndexError` on an out-of-range index
   instead of desynchronizing `keys` and `items`.
9. `Logbook.pop` normalizes a negative index before comparing it to
   the stream cursor.
10. `Logbook.pop` removes the chapter row that shares that
    generation, so `pop(i)` and `del logbook[i]` stay aligned.
11. `Logbook.__delitem__` removes the chapter row that shares the
    same generation — including a later occurrence of a repeated
    `gen` and every index in a slice — not the same list index.
12. `Logbook.stream` and `str` pair chapter cells by `gen`. A
    generation recorded without a chapter no longer shifts later
    values onto the wrong row or IndexErrors once the stream cursor
    is past the shorter chapter.
13. `History.update` records every member of a batch. A single
    individual without `history_index` no longer orphans the rest.
14. `GridArchive` tessellates behavior descriptors into a MAP-Elites
    grid: `add` keeps the best individual per cell, `random_elites`
    samples parent copies, and `stats` reports coverage and
    `qd_score`. `ea_map_elites` drives evaluate → archive → `var_or`
    and logs archive metrics each generation. Fitness stays on
    `ind.fitness`; behavior measurement stays on the caller.
15. `var_or` mates two clones of the only parent when the pool has a
    single individual, so $(1,\lambda)$ / $(1+\lambda)$ with
    `cx_prob > 0` no longer raises `ValueError` on `sample(..., 2)`.
    `ea_mu_comma_lambda` with `survivors=1` can run past generation
    one.
16. `GridArchive.add` rejects a non-finite first weighted objective.
    NaN or infinity no longer replaces a finite elite or occupies an
    empty cell.

## Persistence

1. `Checkpoint.save` writes a sibling `.tmp` file and replaces the
   destination. A dump that fails part-way through does not truncate
   the last good checkpoint.
2. `Checkpoint.load` restores the constructor's `file_path`,
   `raise_errors`, and `make_dir` after unpickling, so moving a
   checkpoint file does not send the next save back to the old path.
3. Setting `save_freq = -1` while iterating `Checkpoint.range`
   disables further saves. It no longer turns saving on for every
   remaining generation.

## Creator

1. `creator.create` keeps the `typecode` of an `array.array`
   instance base. It no longer forces `"b"`.

## Constraints and utilities

1. `ClosestValidPenalty` treats a scalar NumPy distance like
   `DeltaPenalty` does: a 0-d array is broadcast, so an `ndarray`
   is not passed to `itertools.repeat`.
2. `SortingNetwork.evaluate` copies each case, sorts the copy, and
   compares it to `sorted(original)`, so integer cases are not
   scored as bit-count patterns.
3. `case_errors` reduces aligned 1D `predicted` and `target` series
   into one mean-squared error per case segment. Accepts explicit
   half-open `(start, stop)` ranges or a boolean mask (one case per
   contiguous `True` run). Non-finite samples are skipped; optional
   `valid=` covers the `vwhere` warmup trap from columnar GP.

## Benchmarks

1. DTLZ5 / DTLZ6 apply the angular $\theta$ map only to the first
   $M-1$ decision variables. Distance variables feed $g$ and are not
   multiplied into $f_1$ as extra cosines.
2. Chuang F3 scores the selector-1 branch with the published trap
   (not the inverse trap) and covers bits $0..39$ without dropping
   bit 38.
3. Moving Peaks `ALT1` uses `move_severity` $1.5$, matching its
   documented table.
4. Royal Road R2 sums R1 at every doubling of `order` that still
   fits the bit string. The top-level schema is included, so a
   64-bit all-ones individual with order $8$ scores $256$.
5. Kotanchek uses the published denominator $1.2$, not $3.2$.

[deap-24]: https://github.com/DEAP/deap/issues/24
[deap-121]: https://github.com/DEAP/deap/issues/121
[deap-321]: https://github.com/DEAP/deap/issues/321
[deap-350]: https://github.com/DEAP/deap/issues/350
[deap-383]: https://github.com/DEAP/deap/issues/383
[deap-426]: https://github.com/DEAP/deap/issues/426
[deap-472]: https://github.com/DEAP/deap/issues/472
[deap-500]: https://github.com/DEAP/deap/issues/500
[deap-527]: https://github.com/DEAP/deap/issues/527
[deap-641]: https://github.com/DEAP/deap/issues/641
[deap-644]: https://github.com/DEAP/deap/issues/644
[deap-655]: https://github.com/DEAP/deap/issues/655
[deap-694]: https://github.com/DEAP/deap/issues/694
[deap-720]: https://github.com/DEAP/deap/issues/720
[deap-735]: https://github.com/DEAP/deap/issues/735
[deap-740]: https://github.com/DEAP/deap/issues/740
[deap-750]: https://github.com/DEAP/deap/issues/750
[deap-755]: https://github.com/DEAP/deap/issues/755
