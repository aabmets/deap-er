# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.7] - 2026-10-01

Genie-usage bug hunt: 16 fixes in the Numba backend, opcode CSE, GP
checks and ephemerals, checkpoints, operators, and CMA strategies
([#141](https://github.com/aabmets/deap-er/pull/141)–[#146](https://github.com/aabmets/deap-er/pull/146)).
The public names on `deap_er`, `tools`, and `gp` are unchanged.

### Changed

- **Numba `rolling_sum` and `rolling_mean` results change.** On
  `backend="numba"` they now use compensated summation that resets
  when the window holds no finite sample, so every series changes in
  its low digits, and a series that held a large value or followed
  another packed symbol changes by a lot (see Fixed). `rolling_std`
  changes slightly as well, because its running total is compensated
  too ([#146](https://github.com/aabmets/deap-er/pull/146))
- `mut_node_replacement` and `mut_shrink` can now pick the root. Node
  replacement swaps the root for a primitive of the same signature,
  or a lone terminal for another terminal of its type; shrink can drop
  the outermost primitive. `mut_shrink` still leaves trees of fewer
  than three nodes or height one alone. **Seeded GP runs that use
  either mutation change** ([#144](https://github.com/aabmets/deap-er/pull/144))
- **Lexicase picks change when a case holds NaN or `±inf`.**
  `sel_lexicase`, `sel_epsilon_lexicase`, and
  `sel_batch_epsilon_lexicase` rank a non-finite case value as the
  worst value for that case, so it never passes a case that a finite
  value passes, and the ε-lexicase MAD is taken over the finite values
  only ([#144](https://github.com/aabmets/deap-er/pull/144))
- The three lexicase selectors raise `ValueError` when any individual
  is unevaluated, the first one included. A pool where every
  individual was unevaluated used to return uniform random picks
  ([#144](https://github.com/aabmets/deap-er/pull/144))
- `Strategy.update` and `StrategySeparable.update` raise
  `FloatingPointError` when sigma collapses to zero or the state
  turns non-finite, and restore the state they had before the call.
  `RestartStrategy` treats that as the end of a run and restarts
  ([#145](https://github.com/aabmets/deap-er/pull/145))
- CMA hyperparameters given to a constructor, `compute_params`, or
  `reset_state` stay pinned until a later call overrides them, as
  bounds already did, and are listed on `strategy.hyperparams`.
  `reset_state`, every IPOP/BIPOP restart, and every leftover-budget
  resize used to reset the ones a call did not repeat, such as
  `weights=` and the rank and cumulation rates, to their defaults.
  **Restarted runs with custom hyperparameters change**
  ([#145](https://github.com/aabmets/deap-er/pull/145))
- `Checkpoint.save` pickles with the standard C pickler and falls back
  to dill only for state the C pickler refuses (lambdas, local
  functions) or that references `__main__`. Saving is about 15-19x
  faster on large populations; `load` reads both
  ([#141](https://github.com/aabmets/deap-er/pull/141))
- `interpret_tapes(backend="opcode")` frees each common-subexpression
  column after its last consumer has run instead of holding every
  unique node until the batch ends. A 32-tape batch over 200k rows
  peaks at 97 MB instead of 447 MB; results are bit-identical
  ([#142](https://github.com/aabmets/deap-er/pull/142))
- Static and automatic ε-lexicase build each case's pass mask once per
  call instead of once per selection; the picks under a seed are
  identical ([#144](https://github.com/aabmets/deap-er/pull/144))

### Fixed

- Numba `rolling_sum` and `rolling_mean` kept one running total over
  the whole packed matrix and never rebuilt it, so one large value
  corrupted every later window, and the error carried through the
  NaN padding into the next packed symbols. They now match the
  python and opcode backends window by window, and a symbol packed
  after padding at least one window long gives the same values as
  that symbol alone. `rolling_std` no longer stays NaN for the rest of
  a series after one window overflows
  ([#146](https://github.com/aabmets/deap-er/pull/146))
- The Numba interpreter missed its disk cache in every new process,
  recompiling for about 2.5 s and adding about 260 KB to the cache
  each time. The interpreter and the builtin batch kernels now load
  from the cache, and a consumer `dispatch` kernel binds its own
  uncached interpreter once per process
  ([#146](https://github.com/aabmets/deap-er/pull/146))
- Serial `interpret_tapes(backend="numba")` shared one process-wide
  workspace, so calls from several threads at once returned wrong
  numbers without an error. Each call now allocates its own
  workspace ([#146](https://github.com/aabmets/deap-er/pull/146))
- `PrimitiveTree.from_string` raised `OverflowError` instead of
  `gp.ProgramError` for a window literal too large to convert to a
  float, such as a 400-digit integer
  ([#143](https://github.com/aabmets/deap-er/pull/143))
- A pickled tree holding an ephemeral from `add_ephemeral_constant`
  could only be loaded after a primitive set had registered that
  ephemeral, which failed in a fresh worker or on resume. The
  ephemeral now pickles its name, return type, and sampler and
  rebuilds its class on load. Pickles written by 3.1.6 still load
  once the ephemeral is registered, as before
  ([#143](https://github.com/aabmets/deap-er/pull/143))
- `Checkpoint.save` staged every writer under the same `.tmp` name
  and never fsynced, so two writers could publish a torn file and a
  power loss could leave an empty one. Each save now writes a
  uniquely named staging file, fsyncs it, replaces the target, and
  fsyncs the directory ([#141](https://github.com/aabmets/deap-er/pull/141))
- `Checkpoint.save` let a `ValueError` from `__reduce__` or a
  `RecursionError` escape `raise_errors=False` and left the `.tmp`
  file behind. Any exception while saving is now a save error, and
  the staging file is always removed
  ([#141](https://github.com/aabmets/deap-er/pull/141))
- `Strategy.update` divided the rank-μ term by `sigma**2`, which
  underflows to zero long before sigma does, and then raised a raw
  `LinAlgError` from `eigh` with the strategy half-updated. The term
  is now computed from `(samples - centroid) / sigma`, and a real
  collapse raises `FloatingPointError` as described under Changed.
  `StrategySeparable` silently went NaN in the same case
  ([#145](https://github.com/aabmets/deap-er/pull/145))
- `mig_ring` with `replacement=None` moved an emigrant by reference
  but refilled only one of its home slots, so a deme that held the
  same object twice ended up sharing it with the next deme, and
  `step_islands(eval_keys=...)` scored it with the wrong deme's
  fitness. A mover still present at home is now cloned
  ([#144](https://github.com/aabmets/deap-er/pull/144))

## [3.1.6] - 2026-10-01

### Changed

- `PrimitiveTree.from_string` checks parentheses and commas against
  each primitive's arity and raises `gp.ProgramError` on malformed
  text it used to accept, such as `vadd(a, b))`, `vadd(a b)`, and
  `(vadd(a, b))`. A zero-argument terminal can still be written
  `name()` ([#140](https://github.com/aabmets/deap-er/pull/140))

### Fixed

- `compile_tree` on the default `python` backend raised
  `SyntaxError: too many nested parentheses` for a tree nested more
  than about 200 levels deep (about 100 under window primitives),
  which `from_string` and `lower_tree` accept. Trees deeper than 64
  levels now compile to a flat function; shallower trees compile as
  before ([#140](https://github.com/aabmets/deap-er/pull/140))
- `str()` of a malformed node list silently dropped nodes:
  `[vadd, a]` printed `a`. It now raises `gp.ProgramError` for a
  list that is missing arguments or holds nodes after its root
  ([#140](https://github.com/aabmets/deap-er/pull/140))

## [3.1.5] - 2026-10-01

Audit follow-up: GP nodes and parsing, the Numba backend, records,
and CMA strategies
([#133](https://github.com/aabmets/deap-er/pull/133)–[#137](https://github.com/aabmets/deap-er/pull/137)).
The public names on `deap_er`, `tools`, and `gp` are unchanged;
`gp.ProgramError` is new.

### Added

- `HallOfFame` and `GridArchive` take an optional `key=` scalariser
  (larger is better), so a multi-objective individual can be ranked
  by one scalar. A non-finite key value is not stored, and
  `GridArchive` sums `key` into `qd_score`. `HallOfFame.from_json`
  takes `key` as well; it is not serialized
  ([#135](https://github.com/aabmets/deap-er/pull/135))
- `structural_meta_case_columns` accepts a 2-D `valid` mask of shape
  `(n_individuals, n_rows)`, one mask per individual. 1-D masks work
  as before ([#135](https://github.com/aabmets/deap-er/pull/135))
- `columnar_pset` takes `fill=` and forwards it to
  `add_numpy_primitives` ([#133](https://github.com/aabmets/deap-er/pull/133))
- `gp.ProgramError`, a subclass of both `ValueError` and `TypeError`,
  for programs that are not valid for their primitive set
  ([#137](https://github.com/aabmets/deap-er/pull/137))

### Changed

- GP nodes are read-only: rebinding or deleting an attribute of a
  `Primitive`, `Terminal`, or `Ephemeral` raises `AttributeError`.
  Nodes are shared by the primitive set and every tree that holds
  them, so editing one tree's node used to change the set and every
  other tree. Copy, deepcopy, pickle, dill, and `Checkpoint` still
  work ([#133](https://github.com/aabmets/deap-er/pull/133))
- GP raises `ProgramError` on programs it used to accept:
  `PrimitiveTree.from_string` for every parse failure, including an
  empty string, and for a root that does not return a subtype of
  `prim_set.ret`; `compile_tree` for an empty expression; and
  `compile_tree` and `lower_tree` for a leaf that does not fit its
  argument slot. A window length must be an integer in
  `[1, 2**31 - 1]` at parse, compile, and lowering time; `lower_tree`
  raised `OverflowError` above that range. `ProgramError` subclasses
  `ValueError` and `TypeError`, so existing `except` clauses keep
  working ([#137](https://github.com/aabmets/deap-er/pull/137))
- `from_string` restores a window literal as the set's window
  ephemeral, and a number whose type does not fit its slot as that
  slot's ephemeral, so `mut_ephemeral` and `tune_ephemerals` can resample it.
  A literal at the root takes `prim_set.ret` as its type
  ([#137](https://github.com/aabmets/deap-er/pull/137))
- ε-lexicase computes each case's MAD slack once per call in the
  non-dynamic epsilon modes instead of once per selection; the picks
  are identical ([#135](https://github.com/aabmets/deap-er/pull/135))

### Fixed

- `backend="numba"` ran serial batches of builtin-only tapes on the
  NumPy CSE plan of the opcode backend and compiled nothing. Every
  Numba batch, serial or parallel, now runs the compiled kernel.
  **Numeric output of serial Numba evaluation changes** by about
  `1e-9` relative on short series and about `1e-8` over `1e5` rows;
  NaN patterns are identical. The tolerance between the Numba serial,
  parallel, and opcode paths is documented on `interpret_tapes` and
  `run_tapes` ([#134](https://github.com/aabmets/deap-er/pull/134))
- A `NUMBA_CACHE_DIR` set after numba was imported was ignored; it is
  now picked up ([#134](https://github.com/aabmets/deap-er/pull/134))
- `Strategy` could take the square root of a tiny negative
  eigenvalue of a singular or nearly singular covariance, giving NaN
  in `diag_d` and a negative `cond`. It now symmetrizes the matrix
  and floors the eigenvalues at `1e-14` times the largest one
  ([#136](https://github.com/aabmets/deap-er/pull/136))
- With `bound_mode="clip"`, `Strategy` and `StrategySeparable`
  updated the mean and covariance from the clipped point, which
  zeroed the variance across box faces and drove the state
  non-finite on corner optima. They still evaluate the clipped point,
  but learn from the unclipped draw and clip the new centroid into
  the box. **Seeded trajectories with `bound_mode="clip"` change**
  ([#136](https://github.com/aabmets/deap-er/pull/136))
- `static_limit` deep-copied every positional argument as a
  candidate parent, so a primitive set or generator bound
  positionally could be returned as a child. It now clones only the
  positional arguments of the first argument's type
  ([#133](https://github.com/aabmets/deap-er/pull/133))
- A tree holding an ephemeral or a window ephemeral from
  `add_ephemeral_constant` could not be pickled with the standard
  library; the generated class now names the module that stores it
  ([#133](https://github.com/aabmets/deap-er/pull/133))

## [3.1.4] - 2026-09-28

Bug-hunt release: records, variation, algorithms, strategies and
benchmarks, GP core, selection, GP backends, and core utilities
([#122](https://github.com/aabmets/deap-er/pull/122)–[#129](https://github.com/aabmets/deap-er/pull/129)).
The public names on `deap_er`, `tools`, and `gp` are unchanged.

### Changed

- `Logbook.sort`, `Logbook.reverse`, slice assignment, and `*=` raise
  `TypeError`. They bypassed the chapter and stream-cursor bookkeeping
  and silently desynchronized the log. Integer item assignment still
  works ([#122](https://github.com/aabmets/deap-er/pull/122))
- Records raise `ValueError` on input they used to accept: a negative
  `HallOfFame` `maxsize`; a multi-objective individual in
  `GridArchive`, `CvtArchive`, or `UnstructuredArchive`, now also when
  it is unevaluated or its descriptor is NaN; a `GridArchive` range
  whose span `high - low` overflows; and a non-finite descriptor in
  `CvtArchive.nearest_centroid`, which used to return cell 0
  ([#122](https://github.com/aabmets/deap-er/pull/122))
- `ea_mu_plus_lambda`, `ea_mu_comma_lambda`, and `var_or` raise
  `ValueError` up front on an empty population when offspring are due,
  instead of failing mid-run. `evaluate_invalid`, and so every
  driver, raises `ValueError` when `evaluate_batch` returns fewer
  fitnesses than individuals; the result used to be truncated and the
  rest left unevaluated
  ([#124](https://github.com/aabmets/deap-er/pull/124))
- **`n_evals=0` now evaluates and records generation 0**, then stops.
  It used to record an unevaluated generation 0, and statistics
  crashed on it ([#124](https://github.com/aabmets/deap-er/pull/124))
- `ea_generate_update_restarts`, `ea_map_elites`, and `ea_policy`
  record through the same path as `ea_simple`, so restart fields now
  also reach `MultiStatistics` chapters
  ([#124](https://github.com/aabmets/deap-er/pull/124))
- `RestartStrategy` raises `ValueError` for a `lambda_factor` that
  shrinks λ to 0, and `Strategy` / `StrategySeparable` for `survivors`
  outside `[1, offsprings]`; both used to fail later with
  `ZeroDivisionError` or a NumPy shape error.
  `StrategyOnePlusLambda.reset_state` raises `TypeError` for a parent
  without `fitness`. `RestartStrategy.mode`, `sigma_large`,
  `lambda_factor`, and `max_large_restarts` are read-only
  ([#125](https://github.com/aabmets/deap-er/pull/125))
- Selection rejects input it used to accept: `sel_epsilon_lexicase`
  and `sel_batch_epsilon_lexicase` raise `ValueError` for a negative,
  NaN, or infinite `epsilon`; `next_downsample_cases` raises
  `ValueError` for an unknown `mode` (it silently used `random`) and
  `IndexError` for a fractional cohort index (it was truncated).
  A trusted case matrix whose width differs from `fitness.values`
  raises an error that names the case index or the width instead of a
  bare `IndexError` or `ValueError`
  ([#127](https://github.com/aabmets/deap-er/pull/127))
- GP raises `ValueError` on input it used to accept or crash on:
  `evaluate_columnar` and `bounds_from_matrix` on a matrix with no
  rows; `lower_tree` on a window leaf that is not a positive integer
  (it was truncated, and a window below 1 read outside the Numba
  buffers); window primitives on a non-integral window; and the Numba
  entry points on a malformed hand-built `gp.Tape`
  ([#128](https://github.com/aabmets/deap-er/pull/128))
- `race_stop` rejects an `alpha` outside `(0, 1)` before racing
  instead of mid-race. `nsga_convergence`, `nsga_diversity`, and
  `inv_gen_dist` raise `ValueError` for a reference individual without
  a valid fitness instead of reading its genes as objectives. `mut_de`
  raises `ValueError` when only one of `low` / `up` is given, also on
  an empty individual
  ([#123](https://github.com/aabmets/deap-er/pull/123),
  [#129](https://github.com/aabmets/deap-er/pull/129))
- Refilling the library RNG buffer is faster; the drawn stream is
  bit-identical ([#129](https://github.com/aabmets/deap-er/pull/129))

### Fixed

- `Logbook` bookkeeping: `pop` with an out-of-range index moved the
  stream cursor before raising; `remove` skipped the chapter rows and
  the cursor; `insert` before the cursor streamed rows again; a
  chapter without a first-generation row rendered no columns; a
  streamed chapter lost its column widths; and turning `log_header`
  on mid-stream left the chapter columns out of the header
  ([#122](https://github.com/aabmets/deap-er/pull/122))
- `History.get_genealogy` dropped an ancestor shared by two branches
  under `max_depth`. It now walks breadth-first and keeps each
  ancestor at its shallowest depth
  ([#122](https://github.com/aabmets/deap-er/pull/122))
- Registering a `Statistics` name again appended a duplicate field and
  log column; it now replaces the function. `SemanticSurrogate`
  nearest-neighbour prediction skips rows with non-finite values, and
  `coerce_case_exam` accepts a 1-D integer index array
  ([#122](https://github.com/aabmets/deap-er/pull/122))
- Variation operators: `mut_polynomial_bounded` could mutate at
  `mut_prob=0` (it compared with `<=`); `mut_es_log_normal` divided by
  zero on an empty individual; `mig_ring([])` raised `IndexError`;
  `cx_two_point_copy` and the other copy crossovers crashed on
  `array.array` individuals; `iso_line_int` disagreed with
  `mut_iso_line` on inverted bounds; and `mut_uniform_int` bounds are
  typed as scalars or per-gene sequences
  ([#123](https://github.com/aabmets/deap-er/pull/123))
- Policy helpers: `island_eval_keys` raised on series case exams and
  split equivalent catalog exams; `guard_policy_fitness_exam` compared
  exams by segment index and both over- and under-rejected; and
  `PolicyActionGuard.begin_generation()` without an index never
  advanced, so a promote cooldown blocked promotes forever
  ([#123](https://github.com/aabmets/deap-er/pull/123))
- `ea_policy` logged a `step_islands` action as fewer evaluations than
  ran (1 and 4 where 9 had run). It now charges the pre-dispatch
  estimate to `nevals` and the `n_evals` budget
  ([#124](https://github.com/aabmets/deap-er/pull/124))
- `step_islands` could step earlier demes before rejecting a later
  one, and an immigrant whose `id()` was reused from a freed
  individual could keep another deme's fitness. Every deme is now
  validated first, and arrivals are invalidated correctly
  ([#124](https://github.com/aabmets/deap-er/pull/124))
- Default ε-lexicase could filter out every candidate and then draw
  from the whole pool, returning strictly dominated individuals. A
  case that every remaining candidate fails now removes no one.
  **Selection output of `sel_epsilon_lexicase` and
  `sel_batch_epsilon_lexicase` changes**
  ([#127](https://github.com/aabmets/deap-er/pull/127))
- `sel_spea_2` density used the (k+1)-th nearest neighbour, which
  for N < 4 was the point itself. It now uses the k-th, with
  k = ⌊√N⌋ (Zitzler et al., 2001). **Seeded output of `sel_spea_2`
  changes** when the archive is filled by density
  ([#127](https://github.com/aabmets/deap-er/pull/127))
- `sel_nsga_3` normalized by the worst point of every sorted front
  (or of the whole population on a singular hyperplane) instead of the
  first front, and a flat objective was divided by a near-zero gap.
  `sel_nsga_3` and `sel_age_moea_2` now use the first-front worst and
  pymoo's degenerate-nadir correction. **Selection changes when the
  fallback fires or an objective is flat**
  ([#127](https://github.com/aabmets/deap-er/pull/127))
- `sel_team` failed on a trusted matrix wider than `fitness.values`,
  `sample_informed_cases` did not cap at the trusted matrix width, and
  `next_lexicase_cases` mutated the exams before rejecting a bad floor
  ([#127](https://github.com/aabmets/deap-er/pull/127))
- BIPOP drew the small-regime λ with exponent U where Hansen (2009)
  uses U². **Seeded output of BIPOP restarts changes**
  ([#125](https://github.com/aabmets/deap-er/pull/125))
- MO-CMA restarts ratcheted the parent count down and never let it
  grow back, and a leftover-budget batch collapsed the parent set.
  `RestartStrategy.best_fitness` and the `target_f` stop test read the
  `stagnation_key` value instead of the fitness
  ([#125](https://github.com/aabmets/deap-er/pull/125))
- Benchmarks: `MovingPeaks` raised on a `pfunc` pool shorter than
  `npeaks` and used a negative `uniform_height` / `uniform_width` as is
  instead of drawing random values; `bm_chuang_f3` added the wrap
  block elementwise on NumPy individuals; and `bm_rastrigin_scaled`
  divided by zero on one dimension
  ([#125](https://github.com/aabmets/deap-er/pull/125))
- `compile_tree` could return a stale compiled function when a
  discarded callable's memory address was reused
  ([#126](https://github.com/aabmets/deap-er/pull/126))
- `PrimitiveSet.rename_arguments` left the terminal's `name` stale
  ([#126](https://github.com/aabmets/deap-er/pull/126))
- `tune_ephemerals` left clip bounds on a caller's reused strategy and
  wrote floats into bool and string ephemerals. A deep copy of a
  creator-made `SlimTree` lost its class, and `compile_slim_tree`
  crashed on a zero-argument set. Push policy programs truncated float
  scores in `ADD` / `SUB` and leaked `IndexError` on stack underflow
  (now `ValueError`)
  ([#126](https://github.com/aabmets/deap-er/pull/126))
- A constant under a window op came out as an all-`nan` series on the
  Python and opcode backends but as a constant series on Numba, and
  pair windows raised on a constant operand. All backends now read it
  as a constant column. **Output changes for
  programs that window a constant**
  ([#126](https://github.com/aabmets/deap-er/pull/126),
  [#128](https://github.com/aabmets/deap-er/pull/128))
- `tape_interval` / `tape_flags` certificates were not conservative
  for `rolling_sum`, `diff`, `rolling_std`, `rolling_cov`,
  `rolling_beta`, `vdiv`, and for `vlog` / `vsqrt` on partial domains,
  and missed an all-`nan` `ema` one row short of its warmup. The CSE
  planner shared constant pools across child tapes, and `lower_tree`
  could not lower a named window terminal
  ([#128](https://github.com/aabmets/deap-er/pull/128))
- `write_affine_scale` drew from the library RNG when it built the
  ephemerals it writes. **Seeded runs that call it change**
  ([#128](https://github.com/aabmets/deap-er/pull/128))
- `warmup_numba` compiled no kernels, and the default
  `NUMBA_CACHE_DIR` was ignored when Numba was already imported
  ([#128](https://github.com/aabmets/deap-er/pull/128))
- `creator.create_type` did not instantiate a class attribute with a
  custom metaclass per individual, and re-creating a type with an
  equal ndarray attribute warned and replaced the class. `Fitness`
  kept non-float entries when `values` mixed floats and ints
  ([#129](https://github.com/aabmets/deap-er/pull/129))
- `EvalCache` keyed NumPy genomes by a rounded `str()`, so distinct
  genomes collided; it now keys them by shape, dtype, and bytes.
  `resample` with a cache and no `key`, and `race_stop` without
  `key_fn`, shared or went stale on draws across individuals; draws
  now follow the genome. `race_stop` crashed restoring survivors on
  ndarray individuals and dropped equal-but-distinct ones, and used
  1.96 for any `alpha` other than 0.10, 0.05, and 0.01. It now uses
  the exact normal quantile. **`race_stop` eliminations can change**
  ([#129](https://github.com/aabmets/deap-er/pull/129))
- `Checkpoint.load` accepted a malformed RNG buffer and failed later
  with `IndexError`. It now raises `CheckpointError` and leaves the
  instance and the RNG untouched
  ([#129](https://github.com/aabmets/deap-er/pull/129))
- `semantic_solve_bits` raised a 0/0 `RuntimeWarning` on an empty
  case, and a NaN mean corrupted the `evaluate_case_halving` ranking
  and kept the worst individuals; NaN now ranks last
  ([#129](https://github.com/aabmets/deap-er/pull/129))
- The `noisy_fitness` example used a constant resample key, so every
  mutant shared three cache entries and evolution stalled
  ([#129](https://github.com/aabmets/deap-er/pull/129))

## [3.1.3] - 2026-09-28

### Added

- `add_window_primitives(..., ema=False)` and
  `columnar_pset(..., ema=False)` leave `ema` out of the window kit
- `structural_meta_case_columns(..., valid=mask)` computes
  `non_finite_fraction` over the masked rows only (for example the
  scored rows, leaving out warmup and inter-series padding)
- `gp.UnboundedLookbackError` (a `ValueError`), raised by
  `tape_lookback` for tapes that hold `ema`
- `deap_er.CheckpointError` (a `ValueError`), raised by
  `Checkpoint.load` for a file that exists but is not a loadable
  checkpoint

### Changed

- `creator.create_type` returns the created class instead of `None`.
  Calling it again with the same name and an equal definition returns
  the existing class with no warning and no replacement (classes are
  compared by identity, other values with `==`); a different definition
  still warns and replaces
- `columnar_pset` defaults `window_name` to `None` and derives the
  ephemeral name from the bounds (`window_{low}_{high}`). A second
  `columnar_pset` with different window bounds in the same process no
  longer raises `ValueError`. Stored programs that name the old default
  `window` ephemeral need `window_name="window"`

### Fixed

- Typed GP crossover (`cx_one_point`, `cx_one_point_leaf_biased`,
  `cx_homologous`, `cx_one_point_semantic`) drew the swapped return
  type from a `set` of classes, whose order follows `id()` or the hash
  seed. One seed bred different children in different processes, and
  a checkpoint resumed in a fresh process diverged. The shared types
  are now drawn in first-occurrence order in the first parent.
  **Seeded output of strongly typed GP changes.** Untyped GP is
  unaffected
- `Checkpoint(autoload=True, raise_errors=True)` raised
  `FileNotFoundError` at construction when the file did not exist yet.
  Autoload now loads only an existing file; an explicit `load()` on a
  missing file still raises or returns `False`
- `Checkpoint.load` let `AttributeError`, `ImportError`, `ValueError`,
  `UnicodeDecodeError` and similar unpickling failures escape even with
  `raise_errors=False`. Every failure to deserialise, a payload that
  is not a `dict`, a malformed RNG state, and a failed `hof` rebuild
  now raise the new public `deap_er.CheckpointError` (a `ValueError`)
  or return `False`. An unreadable file still raises `OSError`
- `Checkpoint.load` replaced the instance state and moved the library
  RNG before it knew the file was good. It now validates everything
  first; a failed load changes neither
- `Checkpoint` stored `hof` as JSON even without `hof_ind_cls`, and then
  dropped it on load, leaving a stray `_hof_json_`. Without
  `hof_ind_cls`, `hof` is now pickled like any other attribute. A file
  that holds a JSON `hof` and no individual type raises
  `CheckpointError`. JSON-`hof` files written by 3.1.x still load when
  `hof_ind_cls` is given or was saved in the file
- `ema` turned every sample after the first interior `nan` into `nan`
  on both the opcode and numba backends, so in a frame packed series
  after series only the first series had an average. A non-finite
  sample now ends the segment; the next finite sample seeds a new one
  with its own `window - 1` warmup. **Output changes for inputs with
  interior non-finite samples**
- `tape_lookback` certified `ema` at `window - 1` rows, but an IIR
  average depends on every earlier sample of its segment. It now raises
  `UnboundedLookbackError`; `suffix_rescore` keeps scoring `ema` tapes
  over the full matrix
- `structural_meta_case_weights` maximised `non_finite_fraction`, which
  pushed lexicase toward programs that output nothing. Every structural
  meta-case is now minimised (`-1.0`). **Selection behaviour changes**
- `creator.create_type` wrote new types into the module's own globals,
  so a type named `array`, `warnings`, `cast`, `create_type` and so on
  overwrote the module's import and broke later calls. Such names now
  raise `ValueError`

## [3.1.2] - 2026-09-10

### Changed

- Ruff and ty target Python 3.12; pytest CI runs 3.12 and
  `pytest -m examples`
  ([#121](https://github.com/aabmets/deap-er/pull/121))
- Cloud Agent environment allowlists the hosted SonarCloud MCP
  ([#121](https://github.com/aabmets/deap-er/pull/121))
- Release workflow requires non-empty Unreleased notes, refuses a
  version already in the changelog, and promotes those notes under the
  new version heading (with Keep a Changelog footer links) when
  bumping `pyproject.toml`
  ([#121](https://github.com/aabmets/deap-er/pull/121))

### Fixed

- Pytest collection on Python 3.12 (`typing.Generator` still needs
  three type arguments)
  ([#121](https://github.com/aabmets/deap-er/pull/121))
- Interval analysis treated Mask terminals `True` / `False` as a
  stack underflow and crashed `evaluate_columnar`
  ([#121](https://github.com/aabmets/deap-er/pull/121))

## [3.1.1] - 2026-09-09

### Changed

- Updated performance section in README.md 
- Updated performance page in library docs

### Fixed

- Replaced relative logo URL in README.md with absolute
- Minor cosmetic fixes in README.md

## [3.1.0] - 2026-09-09

Follow-up to 3.0.0. The evolutionary algorithms DEAP already shipped
are still the 3.0 loops; this cut wires tree-GP and columnar search,
exposes `evaluate_invalid` and `ea_policy`, and ships the remaining
case-structured helpers (interval tapes, homologous / semantic
crossover, memetic leash, held-out lexicase / successive-halving,
team-from-archive, structural meta-cases, and noisy fitness resample).
Roadmap items 37–43 and the GP wiring landed as PRs
[#107](https://github.com/aabmets/deap-er/pull/107)–[#116](https://github.com/aabmets/deap-er/pull/116).

### Added

- `gp.register_gp`, `gp.columnar_pset`, and `gp.evaluate_columnar`
  for the standard tree-GP / columnar toolbox wiring
  ([#108](https://github.com/aabmets/deap-er/pull/108))
- `bounds_from_matrix`, `tape_interval`, and `tape_flags` propagate
  column bounds through the opcode kit and certificate useless tapes
  before `interpret_tapes`; `evaluate_columnar(..., static_filter=True)`
  skips flagged programs
  ([#111](https://github.com/aabmets/deap-er/pull/111))
- `cx_homologous` and `cx_one_point_semantic` — homologous subtree swap
  and semantic-nearest crossover on `interpret_tape` rows
  ([#112](https://github.com/aabmets/deap-er/pull/112))
- `MEMETIC_DEFAULT_N_GEN`, `MEMETIC_MAX_N_GEN`, `tune_ephemerals_budget`,
  and `affine_case_errors` for memetic polish and Darwinian affine
  scaling under evaluation budget
  ([#110](https://github.com/aabmets/deap-er/pull/110))
- `sel_team_archive` — greedy team assembly on occupied MAP-Elites cells
  without rewriting member fitness
  ([#113](https://github.com/aabmets/deap-er/pull/113))
- `structural_meta_case_columns` and `structural_meta_case_weights`
  for cheap lexicase bloat and always-on regularization via appended
  meta-cases ([#116](https://github.com/aabmets/deap-er/pull/116))
- `held_out_tail`, `train_head`, `case_generalization_pool`,
  `case_generalization_recipe`, `make_lexicase_train_select`,
  `case_halving_stages`, `evaluate_case_halving`, and `case_eval_charge`
  for case-structured generalization and successive-halving budgets
  ([#114](https://github.com/aabmets/deap-er/pull/114))
- `resample`, `noisy_draw_key`, and `race_stop` for averaging noisy
  fitness draws through `EvalCache` and F-Race-shaped elimination
  ([#115](https://github.com/aabmets/deap-er/pull/115))
- `evaluate_invalid` on the public algorithms / `tools` surface — score
  individuals with invalid fitness via `evaluate_batch` when registered
  ([#107](https://github.com/aabmets/deap-er/pull/107))
- `tools.ea_policy` — `ea_simple` plus one policy observe / decide /
  apply step per generation. Policy-action evaluations count toward
  `n_evals` and the generation `nevals`; a policy step that meets
  the budget is recorded without variation
  ([#108](https://github.com/aabmets/deap-er/pull/108))

## [3.0.0] - 2026-09-08

Major release since GitHub tag `2.0`. This is not a drop-in from DEAP
or from 2.0.0: names, parameter order, and several contracts changed.
Roadmap features landed as PRs [#3](https://github.com/aabmets/deap-er/pull/3)–[#103](https://github.com/aabmets/deap-er/pull/103).
Correctness work landed as commits and later bughunt PRs
([#14](https://github.com/aabmets/deap-er/pull/14)–[#79](https://github.com/aabmets/deap-er/pull/79)).
The evolutionary algorithms DEAP already shipped are still DEAP's;
the items below are the 3.0 rework, new surfaces, and fixes.

### Removed

- `sort_log_non_dominated` and the `HyperVolume` class
- `ffo` on `sort_non_dominated`, `sorting` on `sel_nsga_2` /
  `sel_nsga_3`, `map_func` on `least_contrib`, and `mp_pool` on
  `StrategyMultiObjective`
- Deprecated and obsolete DEAP APIs
- Poetry, Sphinx / Read the Docs, Black, Flake8, isort, and Mypy
  (replaced below)

### Changed

- Requires Python 3.12 or newer; license is Apache-2.0
- Packaging and lockfile moved to uv; lint and types are ruff and ty
- Docs moved to MkDocs Material on GitHub Pages
- Public imports flattened (`base`, `creator`, `tools`, `gp`);
  implementations live under `deap_er.private`
- Algorithms, strategies, and benchmarks are imported from `tools`
- Hypervolume, hypervolume contributions, and Pareto ranking delegate
  to [moocore](https://pypi.org/project/moocore/)
- Process-wide RNG is one seedable NumPy facade (`rng.take_floats`
  for batched uniforms)
- Runtime dependencies are NumPy 2, SciPy, dill, and moocore;
  Numba is an optional extra
- Hot paths were sped up where DEAP still walks Python loops
  (NSGA metrics, NSGA-II, cached `compile_tree`, SPEA-II, clone,
  tournament)

### Added

Genetic programming (columnar and case-structured search):

- Columnar primitive set with NumPy, causal window, pair-window, and
  time-series kits; `interpret_tapes` batch scoring
  ([#3](https://github.com/aabmets/deap-er/pull/3),
  [#100](https://github.com/aabmets/deap-er/pull/100))
- Opcode / Numba tape backends, compile LRU, and `static_limit` clone
  ([#10](https://github.com/aabmets/deap-er/pull/10))
- SLIM non-bloating semantic variation
  ([#11](https://github.com/aabmets/deap-er/pull/11))
- Semantic descriptors, neighbors, and `SemanticSurrogate`
  ([#42](https://github.com/aabmets/deap-er/pull/42))
- Growing typed language via `promote_subtree`
  ([#43](https://github.com/aabmets/deap-er/pull/43))
- `tune_ephemerals` (boxed CMA on numeric leaves)
  ([#41](https://github.com/aabmets/deap-er/pull/41))
- Keijzer affine scaling and Lamarckian writeback
  ([#86](https://github.com/aabmets/deap-er/pull/86))
- Causal lookback and suffix rescore
  ([#85](https://github.com/aabmets/deap-er/pull/85))
- Private Push GP policy loop (observation schema, action applicator,
  held-out fitness, action guards)
  ([#89](https://github.com/aabmets/deap-er/pull/89)–[#95](https://github.com/aabmets/deap-er/pull/95))

Selection, teams, and quality-diversity:

- `case_errors`, vectorized lexicase, down-sampled / informed /
  epsilon-lexicase, and down-sampled tournament
  ([#4](https://github.com/aabmets/deap-er/pull/4),
  [#9](https://github.com/aabmets/deap-er/pull/9),
  [#99](https://github.com/aabmets/deap-er/pull/99),
  [#101](https://github.com/aabmets/deap-er/pull/101))
- `sel_team` and co-evolving case exams
  ([#39](https://github.com/aabmets/deap-er/pull/39),
  [#44](https://github.com/aabmets/deap-er/pull/44))
- `sel_sms_emoa`, MOEA/D, and AGE-MOEA-II
  ([#6](https://github.com/aabmets/deap-er/pull/6),
  [#7](https://github.com/aabmets/deap-er/pull/7))
- Deb constraint-dominance on `sel_nsga_2`
  ([#35](https://github.com/aabmets/deap-er/pull/35))
- `GridArchive`, `CvtArchive`, `UnstructuredArchive`, and
  `ea_map_elites`
  ([#8](https://github.com/aabmets/deap-er/pull/8),
  [#36](https://github.com/aabmets/deap-er/pull/36))
- `sel_novelty` and `mut_iso_line`
  ([#96](https://github.com/aabmets/deap-er/pull/96))

CMA, islands, and toolbox operators:

- Optional box bounds on every CMA strategy; `StrategySeparable`;
  IPOP / BIPOP restarts (`RestartStrategy`,
  `ea_generate_update_restarts`)
  ([#12](https://github.com/aabmets/deap-er/pull/12),
  [#32](https://github.com/aabmets/deap-er/pull/32))
- `cx_heterogeneous`, `mut_gaussian_bounded`, `mut_de`, and
  per-gene heterogeneous mutation
  ([#31](https://github.com/aabmets/deap-er/pull/31),
  [#33](https://github.com/aabmets/deap-er/pull/33),
  [#34](https://github.com/aabmets/deap-er/pull/34))
- `step_islands`, `mig_fully_connected`, `mig_random`, and
  `island_eval_keys`
  ([#40](https://github.com/aabmets/deap-er/pull/40),
  [#98](https://github.com/aabmets/deap-er/pull/98))
- `evaluate_batch`, evaluation budget / `EvalCache`, spawned RNG
  streams, persistent hall of fame (`hof_ind_cls=`), logbook JSON,
  linear-time `duplicate_count`, and `clone_individual`
  ([#5](https://github.com/aabmets/deap-er/pull/5),
  [#84](https://github.com/aabmets/deap-er/pull/84),
  [#87](https://github.com/aabmets/deap-er/pull/87),
  [#97](https://github.com/aabmets/deap-er/pull/97))

### Fixed

- 18 still-open [DEAP issues](https://github.com/DEAP/deap/issues)
  implemented here (not as upstream patches): bounded SBX / blend /
  polynomial mutation, allele-based PMX, CMA box bounds, weighted GP
  primitives, logbook JSON, DCD tournament `k`, and others listed in
  [deap_fixes](https://aabmets.github.io/deap-er/bugfixes/deap_fixes/)
- 84 correctness bugs on operators, GP, CMA, records, checkpoints,
  benchmarks, creator, utilities, and algorithms. Inventory:
  [differences](https://aabmets.github.io/deap-er/overview/differences/).
  Later clusters include `from_string` leftover tokens and Window
  ints ([#58](https://github.com/aabmets/deap-er/pull/58),
  [#71](https://github.com/aabmets/deap-er/pull/71)), `mig_ring`
  aliasing and unequal demes
  ([#55](https://github.com/aabmets/deap-er/pull/55),
  [#70](https://github.com/aabmets/deap-er/pull/70)), CMA λ=1 /
  restarts / unevaluated parents
  ([#49](https://github.com/aabmets/deap-er/pull/49),
  [#54](https://github.com/aabmets/deap-er/pull/54),
  [#69](https://github.com/aabmets/deap-er/pull/69)), Logbook
  chapter alignment
  ([#17](https://github.com/aabmets/deap-er/pull/17),
  [#23](https://github.com/aabmets/deap-er/pull/23),
  [#67](https://github.com/aabmets/deap-er/pull/67)), and empty or
  invalid fitness in HoF / Pareto / archives
  ([#21](https://github.com/aabmets/deap-er/pull/21),
  [#25](https://github.com/aabmets/deap-er/pull/25),
  [#46](https://github.com/aabmets/deap-er/pull/46))

## [2.0.0] - 2022-08-20

First GitHub release of deap-er (tag `2.0`). The library is a rewrite
of [DEAP](https://github.com/DEAP/deap): the evolutionary algorithms
are DEAP's. This cut is the reworked package, not a new algorithm suite.
The GitHub release had no notes; the items below are from the rewrite
commits that landed on that tag.

### Changed

- Relocated the package from `deap` to `deap_er` and split operators,
  algorithms, GP, strategies, and utilities into separate modules
- Added type hints, Google-style docstrings, and renamed public
  functions and datatypes
- Refactored checkpoints into a controller and changed range-function
  behavior
- Tightened the built-in `ea_simple`, `ea_mu_plus_lambda`,
  `ea_mu_comma_lambda`, and `ea_generate_update` loops
- Dropped the Ray runtime dependency
- Limited supported Python to 3.9 and 3.10

### Added

- NumPy-compatible crossover operators
- Examples and Sphinx / Read the Docs tutorials ported onto the new API

### Fixed

- Fitness comparison
- NSGA-III selection
- Hypervolume and least-contributed indicator
- CMA strategy keyword arguments

[Unreleased]: https://github.com/aabmets/deap-er/compare/3.1.7...HEAD
[3.1.7]: https://github.com/aabmets/deap-er/compare/3.1.6...3.1.7
[3.1.6]: https://github.com/aabmets/deap-er/compare/3.1.5...3.1.6
[3.1.5]: https://github.com/aabmets/deap-er/compare/3.1.4...3.1.5
[3.1.4]: https://github.com/aabmets/deap-er/compare/3.1.3...3.1.4
[3.1.3]: https://github.com/aabmets/deap-er/compare/3.1.2...3.1.3
[3.1.2]: https://github.com/aabmets/deap-er/compare/3.1.1...3.1.2
[3.1.1]: https://github.com/aabmets/deap-er/compare/3.0.0...3.1.1
[3.1.0]: https://github.com/aabmets/deap-er/compare/3.0.0...3.1.0
[3.0.0]: https://github.com/aabmets/deap-er/compare/2.0...3.0.0
[2.0.0]: https://github.com/aabmets/deap-er/releases/tag/2.0
