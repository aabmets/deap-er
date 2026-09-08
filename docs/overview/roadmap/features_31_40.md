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

## 32. Constraint-dominance on remaining selectors

**What.** The same Deb rule already on `sel_nsga_2` — feasible
beats infeasible; two feasibles use ordinary Pareto / crowding;
two infeasibles prefer the smaller violation — on
`sel_nsga_3`, `sel_sms_emoa`, `sel_spea_2`, and
`sel_age_moea_2`. Optional `feasible=` / `violation=` kwargs.
Omitted kwargs keep the unconstrained path. Fitness values are
not rewritten.

**Today.** `constraint_dominates` and `sel_nsga_2(..., feasible=,
violation=)` exist. The other environmental selectors ignore
feasibility.

**Benefit.** Constrained many-objective users stop inventing a
penalty scale for every selector. Penalties remain the other
path. Closes the rest of the [constraint-handling][deap-30]
request that item 18 started.

**Scope.** One comparison rule on the selectors that already
rank fronts. Not a constraint framework. Stochastic ranking and
ε-level comparison stay in
[Under consideration](under_consideration.md).

Related: [item 18](features_11_20.md#18-constraint-dominance-selection).

---

## 33. Archive-improving CMA

**What.** A `generate` / `update` wrapper (for example
`ArchiveStrategy`) around `Strategy` or `StrategySeparable`.
Each sample's *search* fitness is archive improvement: new cell,
or strictly better elite in that cell — not `ind.fitness` alone.
`add` still ranks the cell by the individual's real fitness.
`RestartStrategy` can wrap it. Box constraints from the inner
strategy are preserved.

**Today.** CMA strategies optimize a fitness vector. Archives
`add` whatever the caller evaluated. The two meet only in the
caller's loop.

**Benefit.** CMA-ME is “CMA's objective is archive improvement.”
Both halves shipped; this is the meeting point. Not a learned
quality-diversity model — the emitter is still Hansen CMA.

**Scope.** One strategy wrapper and an improvement score.
CMA-MAE's decaying threshold, Pareto-per-cell archives, and
hypervolume-per-cell updates stay in
[Under consideration](under_consideration.md) until this object exists.

Related: [item 8](features_1_10.md#8-quality-diversity-archive),
[item 13](features_11_20.md#13-ipop-bipop-cma-restarts),
[item 19](features_11_20.md#19-sep-cma),
[item 29](features_21_30.md#29-novelty-selection-and-isoline).

---

## 34. RVEA and R-NSGA-II

**What.** Two selectors that reuse `uniform_reference_points`:

- **RVEA** — angle-penalized distance (APD) scalarization plus
  reference-vector adaptation. The many-objective path when
  NSGA-III's simplex assumption is the wrong geometry.
- **R-NSGA-II** — caller-supplied aspiration / reference
  point(s); crowding becomes distance-to-preference. Not
  interactive evolution — no clicks.

**Today.** NSGA-III, MOEA/D (Tchebycheff / PBI), and AGE-MOEA-II
cover the usual many-objective set. There is no APD selector and
no preference-point crowding.

**Benefit.** The two requests those three still miss: an irregular
front that wants reference-vector adaptation, and an engineer
who cares about one corner of the front.

**Scope.** Two selectors. IBEA, HypE, GDE3, and AGE-MOEA-II+
stay in [Under consideration](under_consideration.md) — they catalog
more of pymoo rather than close a hole.

Related: [item 12](features_11_20.md#12-moead-and-age-moea-ii),
[item 18](features_11_20.md#18-constraint-dominance-selection).

---

## 35. Adaptive DE strategy

**What.** A `generate` / `update` object (for example
`StrategyDE`) with a SHADE-style success memory for $F$ and
`CR`. `generate` writes trials with `mut_de` (and, if cheap,
current-to-pbest/1). `update` keeps the better of parent and
trial and records successful parameters. Optional `low` / `up`
match the existing clamp on `mut_de`.

**Today.** `mut_de` is DE/rand/1/bin. Selection and adaptive
$F$ / `CR` stay on the caller. There is no `ea_de` — item 17
scoped that out on purpose.

**Benefit.** Adaptive DE is the algorithm people actually run.
A strategy keeps that promise without a new algorithm family.

**Scope.** One generate/update object. Extra trial recipes are
parameters of that strategy, not a catalog. No first-class
`ea_de` or PSO algorithm.

Related: [item 17](features_11_20.md#17-differential-evolution-operators).

---

## 36. Population tape CSE

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

## 37. Evaluation budget and eval cache

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
row count. A hit does not call the wrapped callable.
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

## 38. Parallel RNG streams

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

[deap-30]: https://github.com/DEAP/deap/issues/30
[deap-75]: https://github.com/DEAP/deap/issues/75
