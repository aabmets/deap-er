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

**Today.** `interpret_tapes` batches individuals against one
matrix. Shared *subexpressions* across individuals are evaluated
again.

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

[deap-75]: https://github.com/DEAP/deap/issues/75
