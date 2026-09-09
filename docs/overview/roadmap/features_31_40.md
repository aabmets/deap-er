# Features 31-40

Items in this decade from the [roadmap overview](index.md).
Status and surfaces live on that table. Empty slots stay empty
until an item is numbered into this range.

---

## 31. Affine scaling and Lamarckian writeback

**What.** Keijzer linear scaling next to `tune_ephemerals`.
Given aligned `predicted` and `target` (and the same `valid=`
mask `case_errors` already uses), fit $a + b\,f(x)$ in the
least-squares sense.

- **Darwinian:** use the scaled series only to write fitness /
  case errors. The tree is unchanged.
- **Lamarckian:** write $a$ and $b$ back as ephemerals or as a
  wrapping Slim delta, then invalidate fitness and the compile
  cache for that expression.

**Today.** `affine_scale(predicted, target, *, valid=)` fits
Keijzer $a + b\,f(x)$ in the least-squares sense on the same
`valid=` mask `case_errors` uses. Darwinian callers apply
$a + b\,f(x)$ only when writing fitness or case errors; the
tree is unchanged. `write_affine_scale(ind, a, b, prim_set)`
writes $a$ and $b$ back as ephemeral leaves wrapping a
`PrimitiveTree`, or as wrapping Slim deltas
($a + b\cdot\mathrm{head}$, and $b$ on each existing delta),
then invalidates fitness and the compile-cache entries for the
old expression. No Autograd. No domain fitness.
`tune_ephemerals` is unchanged.

**Benefit.** Structure plus $a + b\,f(x)$ is the usual difference
between a shape that still fights intercept and slope and a law
the numeric engine can finish. Both halves already exist; they
do not meet.

**Scope.** One scaling helper and an optional writeback onto
leaves / Slim deltas. No Autograd. No domain fitness.

Related: [item 6](features_1_10.md#6-case-structured-evaluation-helper),
[item 24](features_21_30.md#24-memetic-constants),
[Push GP P17](push_gp.md#p17-affine-scaling-and-lamarckian-writeback).

---

## 32. Population tape CSE

**What.** Hash-cons postfix suffixes across a generation, evaluate
each unique sub-tape once against the packed matrix, and stitch
the results. Same Python oracle, same `nan` warmup, same
`interpret_tapes` return shape $(n_{\mathrm{ind}}, n_{\mathrm{rows}})$.

**Today.** `interpret_tapes` hash-conses postfix subexpressions
across a batch, evaluates each unique sub-tape once against the
packed matrix, and stitches the results. The Python oracle,
warmup ``nan`` contract, and ``(n_ind, n_rows)`` return shape are
unchanged. Incremental ``ts_rank`` stays deferred.

**Benefit.** The next columnar speedup after items 3, 4, and 9
that does not need a C rewrite. A generation of related trees
shares most of its suffixes. Same programs, less work.

**Scope.** Common-subexpression elimination inside the batch
interpreter. Incremental `ts_rank` stays a kernel tweak in
[Under consideration](under_consideration.md). No new language.

Related: [item 4](features_1_10.md#4-batch-tape-evaluation),
[item 9](features_1_10.md#9-compile-and-clone-path).

---

## 33. Evaluation budget and eval cache

**What.** Two plumbing pieces on the shared algorithm loop:

1. **`n_evals=`** (or an equivalent stop) on `ea_simple`,
   `ea_mu_plus_lambda`, `ea_mu_comma_lambda`, and
   `ea_map_elites`. `ea_generate_update_restarts` already stops
   on evaluations. Generations remain the default.
2. **`EvalCache`** — wrap `evaluate` / `evaluate_batch` with a
   key of expression text (or a caller key) plus matrix identity
   / row count. `promote_subtree` and `tune_ephemerals` already
   invalidate compile-cache entries; the fitness cache must
   drop those keys too.

**Today.** `n_evals=` is an optional stop on `ea_simple`,
`ea_mu_plus_lambda`, `ea_mu_comma_lambda`, and `ea_map_elites`.
The generation that meets or exceeds the budget is finished,
then the loop returns. Generations remain the default.
`ea_generate_update_restarts` already stops on evaluations.
`EvalCache` wraps `evaluate` / `evaluate_batch` with a key of
expression text (or a caller key) plus matrix identity and
row count. A hit does not call the wrapped callable;
`n_evals` / `nevals` still count the fitness assignment.
`clear_compile_cache` clears every live `EvalCache`;
`invalidate_compiled` drops matching expression keys. That is
the same path `promote_subtree` and `tune_ephemerals` already
use for the compile LRU.

**Benefit.** GP papers report evaluation budgets. A cache is the
other half of the compile LRU: the same tree on the same matrix
should not pay `evaluate` twice.

**Scope.** A stop condition and a cache with explicit
invalidation. Not adaptive operator rates, not racing, not a
new algorithm.

Related: [item 9](features_1_10.md#9-compile-and-clone-path),
[item 13](features_11_20.md#13-ipop-bipop-cma-restarts),
[item 24](features_21_30.md#24-memetic-constants),
[Push GP P15](push_gp.md#p15-evaluation-budget-and-eval-cache).

---

## 34. Parallel RNG streams

**What.** Independent, seedable streams for spawned workers that
still reproduce a run. Process-wide `rng` stays the default and
stays checkpointable. A worker map (or a documented
`spawn_rng(seed, worker_id)` helper) draws from a stream that
does not collide with the parent and does not depend on
scheduling order.

**Today.** Process-wide `tools.rng` is still the default NumPy
`Generator`, and `Checkpoint` still persists it.
`spawn_rng(seed, worker_id)` derives an independent child
stream from the run seed and a stable task id without
advancing the parent. `map_spawned` binds that stream for
each item and returns results in input order, so a pool's
completion order cannot change the run.

**Benefit.** The last reproducibility hole that is still open on
DEAP ([user-provided streams][deap-75] for parallel runs). Golden
tests and papers that use `evaluate_batch` in a pool need a
stream contract, not “hope the OS schedules the same way.”

**Scope.** Stream derivation and a worker entry point. Not a
second RNG library. Not switching the parent stream off NumPy.

Related: [item 4](features_1_10.md#4-batch-tape-evaluation),
[Multiprocessing](../../tutorials/multiprocessing.md),
[Push GP P18](push_gp.md#p18-parallel-rng-streams).

---

## 35. Island topologies

**What.** `mig_fully_connected` / `mig_random` next to
`mig_ring`, plus a helper that computes `eval_keys` from the
current exam.

**Today.** `mig_fully_connected` visits every directed deme pair;
`mig_random` picks one destination per source. Both reuse
`mig_ring` placement rules. `island_eval_keys` hashes each
deme's `CaseExam` and optional matrix identity for
`step_islands(..., eval_keys=)`. Custom graphs stay
`mig_ring(..., mig_indices=)`. Archives still do not
auto-merge.

**Benefit.** Heterogeneous islands shipped in item 25; the
missing half is more than a ring. A named topology plus an
exam-derived key stops every island recipe from reinventing
`mig_indices` and a hash of the case set.

**Scope.** Two migrate callables and an `eval_keys` helper.
Archives still do not auto-merge. Custom graphs stay
`mig_ring(..., mig_indices=)`.

Related: [item 25](features_21_30.md#25-streaming-and-island-ecology),
[Operators](../../reference/operators.md),
[Multiprocessing](../../tutorials/multiprocessing.md).

---

## 36. Persistent hall of fame

**What.** A first-class `hof` slot on `Checkpoint`, or
`HallOfFame.to_json` matching `Logbook`
([persistency of the hall of fame][deap-25]).

**Today.** `HallOfFame.to_json` / `from_json` round-trip
`maxsize` and members as ``genes`` plus ``fitness`` values.
`Checkpoint(..., hof_ind_cls=)` stores ``hof`` as JSON instead
of dill and rebuilds it on load. `Logbook` already round-trips
JSON the same way.

**Benefit.** A resumed run should restore the archive without
relying on dill for the one object papers want to inspect.
Matches the logbook JSON path and the still-open DEAP
request.

**Scope.** A convenience API, not a new record type. Not a
per-generation snapshot mode beyond what `fronts=` already
logs.

Related: [Using checkpoints](../../tutorials/using_checkpoints.md),
[Logging statistics](../../tutorials/logging_statistics.md).

---

## 37. Interval analysis on tapes

**What.** Given column bounds (or empirical min/max), propagate
intervals through the opcode kit. Flag programs that are
identically `nan`, constant, or that use `vwhere` to hide
warmup — before a full `interpret_tapes` / `evaluate` pass.

**Today.** `bounds_from_matrix` derives per-column bounds from a
packed matrix. `tape_interval` and `tape_flags` walk the builtin
opcode kit and certificate identically-``nan``, constant, or
warmup-hiding ``vwhere`` programs before ``interpret_tapes``.
``evaluate_columnar(..., static_filter=True)`` skips those tapes
and writes the existing ``empty`` sentinel instead. The runtime
warmup ``nan`` contract is unchanged; ``tape_lookback`` and
``suffix_rescore`` stay the oracle path for legal suffixes.

**Benefit.** Columnar populations waste evaluations on programs
that cannot be a law. A cheap certificate lets
`evaluate_invalid` and `evaluate_columnar` skip them, or write
a sentinel the caller already uses for empty overlap.

**Scope.** Interval propagation over the builtin opcode kit and
a flag / helper next to `tape_lookback`. Not a domain fitness.
Not a substitute for the runtime `nan` contract.

Related: [item 30](features_21_30.md#30-causal-lookback-and-suffix-rescore),
[item 32](#32-population-tape-cse),
[Columnar programs](../../tutorials/columnar_gp.md).

---

## 38. Structural meta-case regularization

**What.** Extra cheap cases — size, depth, unique opcodes,
promote-library hits, time-in-output / non-finite fraction —
appended to the case matrix so lexicase regularizes bloat and
“always on” programs without a second fitness weight.

**Today.** `structural_meta_case_columns` returns size, depth,
unique opcodes, promote-library hits, and non-finite fraction for
a packed population. `structural_meta_case_weights` supplies
default lexicase signs. Append with `numpy.hstack` to
`fitness_case_matrix`, pass `trust_matrix=True`, and extend
`fit_weights` when the matrix is wider than `fitness.values`.
The caller still owns `evaluate` and may omit columns via
`columns=`.

**Benefit.** Machine-checkable pressure on the same path as
case exams. A constant or giant tree fails an extra case
instead of requiring a magic penalty in `evaluate`.

**Scope.** A helper that returns extra columns for a packed
population. The caller still owns `evaluate` and may omit
any column. Not human-in-the-loop. Not a second objective
on `Fitness.weights` unless the caller concatenates them
there.

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 27](features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament),
[item 28](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule),
[item 41](features_41_50.md#41-case-structured-generalization-path).

---

## 39. Homologous and semantic crossover

**What.** Align similar subtrees (homologous) or prefer nodes
whose `interpret_tape` vectors are close (`semantic_nearest`
on subtrees). Type-matched one-point stays the default.

**Today.** `gp.cx_homologous` swaps subtrees at the same
root-to-node path when return types match, otherwise falling
back to type-matched one-point. `gp.cx_one_point_semantic`
picks the type-matched partner whose ``interpret_tape`` row is
nearest to the anchor subtree via ``semantic_nearest`` on
batched ``interpret_tapes`` rows. ``gp.cx_one_point`` remains
the default mate. SlimGP still moves in output space through
``cx_semantic`` / ``cx_slim_donor``.

**Benefit.** Long, expensive tapes survive variation more
often. Random typed swaps on unrelated subtrees are the usual
way a good law dies in one generation.

**Scope.** Optional crossover(s) next to `cx_one_point`.
Default stays type-matched one-point. Not a second genome.

Related: [item 7](features_1_10.md#7-non-bloating-semantic-variation),
[item 22](features_21_30.md#22-semantic-search-space),
[item 31](#31-affine-scaling-and-lamarckian-writeback),
[item 32](#32-population-tape-cse).

---

## 40. Noisy fitness resample

**What.** `resample(ind, evaluate, n)` and a racing stop
(F-Race-shaped) for noisy cases. Repeats go through
`EvalCache` when the key is unchanged; a noisy `evaluate`
must use a key that includes the draw.

**Today.** `resample(ind, evaluate, n)` averages ``n`` independent
draws and optionally writes ``fitness.values``. Repeats route
through ``EvalCache`` when ``cache=`` is set; ``noisy_draw_key``
pairs a caller key with the draw index so identical draws hit
the cache. ``race_stop`` adds one resample per survivor per
round and drops challengers whose first objective is
significantly worse than the leader (F-Race-shaped elimination).
``race_eval_charge`` counts evaluate units for ``n_evals=``
budgeting. ``evaluate`` stays on the caller.

**Benefit.** Time-series and case-structured search treat a
single history pass as an estimate. Racing spends the budget
on individuals whose rank is still unstable.

**Scope.** A resample helper and an optional race stop.
`evaluate` stays on the caller. Not a domain metric. Not a
new algorithm loop.

Related: [item 33](#33-evaluation-budget-and-eval-cache),
[item 6](features_1_10.md#6-case-structured-evaluation-helper),
[item 41](features_41_50.md#41-case-structured-generalization-path).

[deap-25]: https://github.com/DEAP/deap/issues/25
[deap-75]: https://github.com/DEAP/deap/issues/75
