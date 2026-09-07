# Roadmap

Planned library work on the existing public surface
(`gp`, `operators`, `records`, `strategies`, `algorithms`). This is
a backlog, not a schedule and not a promise of release dates.

The toolbox contract does not change. Evaluation, data loading, and
any domain metric stay on the caller. New work is operators, primitive
kits, archives, and evaluation plumbing — not a second genome family,
not a new top-level package, and not an extra runtime dependency
unless an item says otherwise.

Items are ordered by how much they help **columnar and case-structured
program search** first, then multi-objective and continuous search,
then housekeeping. Each item names the current gap and the intended
surface.

| # | Item | Surface | Status |
|:--|:-----|:--------|:-------|
| 1 | [Two-input causal windows](#1-two-input-causal-windows) | `gp` | shipped |
| 2 | [Causal time-series unaries](#2-causal-time-series-unaries) | `gp` | shipped |
| 3 | [Incremental window kernels](#3-incremental-window-kernels) | `gp` (same API) | shipped |
| 4 | [Batch tape evaluation](#4-batch-tape-evaluation) | `gp` | shipped |
| 5 | [Down-sampled and informed lexicase](#5-down-sampled-and-informed-lexicase) | `operators` | shipped |
| 6 | [Case-structured evaluation helper](#6-case-structured-evaluation-helper) | utilities + docs | shipped |
| 7 | [Non-bloating semantic variation](#7-non-bloating-semantic-variation) | `gp` | shipped |
| 8 | [Quality-diversity archive](#8-quality-diversity-archive) | `records` | shipped |
| 9 | [Compile and clone path](#9-compile-and-clone-path) | `gp`, `tools` | shipped |
| 10 | [Vectorized lexicase and plexicase](#10-vectorized-lexicase-and-plexicase) | `operators` | shipped |
| 11 | [SMS-EMOA](#11-sms-emoa) | `operators` | shipped |
| 12 | [MOEA/D and AGE-MOEA-II](#12-moead-and-age-moea-ii) | `operators` | shipped |
| 13 | [IPOP / BIPOP CMA restarts](#13-ipop-bipop-cma-restarts) | `algorithms`, `strategies` | planned |
| 14 | [Linear-time duplicate count](#14-linear-time-duplicate-count) | `tools` | planned |

Shipping an item updates this page and the matching tutorial or
reference stub.

!!! note
    deap-er stays a pure-Python package. Native work remains an
    optional extra (`numba` today) or an upstream wheel
    ([moocore](https://pypi.org/project/moocore/)). An in-tree C,
    Cython, or Rust rewrite of operators, CMA, or selection is not
    on this list.

---

## 1. Two-input causal windows

**What.** Optional helpers next to
`gp.add_window_primitives` that register binary rolling operators
over two `Array` arguments and a `Window`: `rolling_corr`,
`rolling_cov`, and `rolling_beta`. Each stays **causal** — output
sample $t$ depends on the two series on $[t - n + 1, t]$ only.
Samples without enough history are `nan`, matching the unary kit.
Python, opcode, and Numba paths ship together with parity tests.

**Today.** `add_pair_window_primitives` registers causal
`rolling_corr`, `rolling_cov`, and `rolling_beta`. Moments use the
population divisor; beta is the OLS slope of the first series on
the second. The unary kit is unchanged. Python, opcode, and Numba
paths agree.

**Benefit.** The type tags already allow a second `Array`. Callers
stop forking the primitive set for the most common two-series
aggregates. Time-series GP and general columnar programs gain the
same language.

**Scope.** Kit functions plus opcodes, not a named-indicator library.
Every new opcode needs a Python reference implementation.

Related: [Columnar programs](../tutorials/columnar_gp.md),
[Genetic programming](../reference/gp.md).

---

## 2. Causal time-series unaries

**What.** A second optional kit (for example `add_ts_primitives`)
for causal operators that the unary rolling stats do not cover:

- time-series rank of the current sample inside a trailing window
  (`ts_rank`)
- index of the window maximum / minimum (`ts_argmax`, `ts_argmin`)
- exponential decay or a weighted trailing sum
- a protected one-step change (`delta` / `delay`) that keeps the
  existing `nan` warmup contract

Cross-sectional rank across many series is out of scope unless the
caller stacks those series as columns and treats the operation as
row-wise.

**Today.** `add_ts_primitives` registers causal `ts_rank`,
`ts_argmax`, and `ts_argmin` over an `Array` and a `Window`. Rank is
the 1-based average rank of the current sample, scaled by
$(r - 1) / (n - 1)$ so a unique window low is $0$ and a unique high
is $1$. A window of 1 is `nan`. Arg-extremum is the age of the
extreme (`0` is the current sample); a tie keeps the most recent.
`delay`, `diff`, and `ema` stay on the unary window kit. Python,
opcode, and Numba paths agree.

**Benefit.** Programs can say “where this sample sits in the last
$n$ values” and “when the trailing extremum occurred” without each
user binding a custom opcode. Same three-backend parity rule as
item 1.

**Scope.** Optional kit, not a domain catalog of named indicators.
Composable windows plus `bind_numba_opcode` remain the way to add
a specialized kernel.

---

## 3. Incremental window kernels

**What.** Keep the public window API. Change the Numba (and, if
needed, NumPy) kernels from a nested scan over `(t, j)` to running
aggregates: $O(\text{rows})$ sum / mean / std, and a monotonic
deque for min / max. `nan` warmup and “`nan` in the window ⇒ `nan`
out” for min / max stay identical. The current Python functions
remain the definition; new kernels are accepted only with parity
tests.

**Today.** Numba walks each rolling opcode in $O(\text{rows})$:
running sums for `rolling_{sum,mean,std}` and the pair moments,
and a monotonic index ring for `rolling_{min,max}` and
`ts_argmax` / `ts_argmin`. `ts_rank` is still a per-window scan.
Python and the `opcode` backend stay the definition —
`sliding_window_view` plus a ufunc reduce, and for `rolling_std` and
the pair moments each window is centered on its own mean before
squaring, in row blocks. The kernel keeps running sums and recenters
a window only once they have lost too many digits, so its result
follows the oracle without depending on how either side rounds.
Parity tests compare Numba to that oracle, including `nan` recovery,
$\pm\inf$, columns far from zero, and a variance that collapses.

**Benefit.** A tree with several rolling nodes on a long column and
windows of tens to hundreds of samples spends almost all of its
evaluation time here. Same programs, same results, less work per
sample. Every current window user benefits; no new primitive names.

---

## 4. Batch tape evaluation

**What.** A compiled entry point of the form
`interpret_tapes(tapes, matrix) -> (n_individuals, n_rows)`: one
call, many tapes, one packed `(rows, columns)` matrix. Document a
recipe that compiles each unique tree string once and then scores
the generation in one shot. `evaluate_batch` on the toolbox already
replaces `map` for a generation; this is the matching interpreter.

**Today.** `interpret_tapes(tapes, matrix)` returns
`(n_individuals, n_rows)` from one packed `(rows, columns)` matrix.
The opcode path unpacks columns once. The Numba path is a compiled
loop over jagged tapes; `parallel=True` uses one workspace per
thread. Algorithms still call `toolbox.evaluate_batch(invalids)`
when that operator is registered — the matching interpreter is
this function. `compile_tree` is unchanged: the opcode runner
still stacks columns per individual.

**Benefit.** Columnar evaluation becomes population-wide instead of
only per tree. One kernel launch, one matrix, no per-individual
repack or result copy. That is the gap `evaluate_batch` is waiting
on.

Related: [Multiprocessing](../tutorials/multiprocessing.md).

---

## 5. Down-sampled and informed lexicase

**What.** Optional `cases=` on `sel_lexicase` and
`sel_epsilon_lexicase` so one selector call filters on an explicit
subset of case indices. Informed sampling is a **case-subset
builder** used *by* lexicase, not a third ad-hoc selector.
`sel_lexicase` defaults do not change.

**Today.** `sel_lexicase` and `sel_epsilon_lexicase` accept
`cases=` (keyword-only). When omitted, every fitness index is used.
`sample_informed_cases` builds a subset by farthest-first traversal
of Hamming distances between case solve vectors (Boldi et al.), so
synonymous cases are not over-sampled. A case is solved when its
value is $0$. Evaluation stays on the caller.

**Benefit.** When fitness is a vector of cases — folds, series,
regimes, or per-point residuals — full lexicase is
$O(\textit{sel\_count} \times n \times \textit{cases})$ and always
uses every case. A subset buys more individuals per evaluation
budget. Informed sampling avoids wasting that budget on cases that
the current population already treats as interchangeable.

---

## 6. Case-structured evaluation helper

**What.** A thin utility, for example
`case_errors(predicted, target, ranges)`, that given a predicted
series, a target series, and a boolean mask or a list of index
ranges, returns one loss per range. Those values are the cases
lexicase (item 5) and a multi-objective fitness vector consume.
A short addition to the
[columnar tutorial](../tutorials/columnar_gp.md) states the
honest pattern: ignore non-finite warmup, do not score a `vwhere`
that hid a `nan`, and keep prediction–target alignment on the
caller.

**Today.** `tools.case_errors` turns aligned `predicted` and `target`
series into one MSE per explicit `[start, stop)` range or per
contiguous `True` run in a boolean mask. Non-finite samples are
skipped by default; an optional `valid` mask covers the `vwhere`
warmup trap documented in the columnar tutorial. There is still no
built-in chronological split or metric catalog.

**Benefit.** The usual “several segments, hide some” recipe is a
few lines if the helper exists, and a class of off-by-one and
lookahead bugs if everyone writes it. This is not a split
algorithm, a metric catalog, or an evaluation framework — only
the reduction from a series to per-case errors.

**Scope.** No built-in chronological split algorithm. No public
fitness functions that assume application-specific state the
library does not have.

---

## 7. Non-bloating semantic variation

**What.** SLIM-style operators next to the existing wrapping pair:

- inflate mutation (classic geometric semantic mutation) plus a
  **deflate** mutation that shortens the genotype while remaining
  a ball mutation of radius `ms` in semantic space
- a size-preserving semantic crossover (XOBDn / SLIMMER, or a
  minimal subset) whose offspring are no larger than the parents

Reuse the current requirement that the primitive set expose `lf`,
`add`, `mul`, and `sub`. Keep `cx_semantic` and `mut_semantic`
for compatibility. A small example that combines HARM with SLIM
shows a non-bloating semantic path.

**Today.** `mut_slim_inflate`, `mut_slim_deflate`, `mut_slim`, and
`cx_slim_donor` implement SLIM+SIG2 inflate/deflate mutation and
best-donor crossover (XOBDn). The `SlimTree` genotype backs those
operators only; standard GP continues to use `PrimitiveTree`.
`cx_semantic` and `mut_semantic` remain the classical wrapping
operators. An example combines HARM with SLIM semantic variation.

**Benefit.** Geometric semantic variation makes crossover and
mutation geometric in the space of input–output vectors, which
turns a convex loss into a unimodal landscape. Deflate and
size-preserving crossover keep that landscape without producing
trees that cannot be read or that trip `static_limit` every
generation. HARM remains the bloat control for standard (syntactic)
variation.

---

## 8. Quality-diversity archive

**What.** A `GridArchive` (or equivalent) in `records`: the caller
supplies a behavior descriptor, then `add` and `random_elites` for
variation. A thin `ea_map_elites` can wrap
`evaluate_invalid` + archive + `var_or`. Typical descriptor
coordinates are program length, output variance, or how often a
mask is true — whatever the caller measures.

**Today.** `GridArchive` bins caller-supplied behavior descriptors into a
uniform grid and keeps the best individual per cell. `add` and
`random_elites` are the variation surface; `ea_map_elites` wraps
`evaluate_invalid`, archive updates, and `var_or`. Fitness stays on
`ind.fitness`; descriptors stay on the caller.

**Benefit.** Quality-diversity keeps the best individual per cell
instead of a single champion or a single Pareto front. That is
the usual antidote to one overfit program when many competent
shapes exist. The archive is a data structure, not a framework:
no mandatory algorithm class, no learned QD / meta-BBO.

Related: [Records](../reference/records.md).

---

## 9. Compile and clone path

**What.** Four small changes that keep the same APIs and cut waste
on long columnar runs:

1. **LRU compile cache** — replace the current
   “store 1024 entries, then clear the whole dict” policy so a
   diverse working set is not evicted together.
2. **Packed matrix on the opcode backend** — accept a pre-packed
   `(rows, columns)` array the way the Numba backend already does,
   instead of `numpy.stack` per individual.
3. **Numba interpreter cache** — compile the VM with `cache=True`
   (or an explicit cache directory) so spawned workers do not
   re-JIT the interpreter in every process.
4. **`static_limit` and `clone_individual`** — when the individual
   is a sequence of immutable nodes, reject oversized offspring
   with `clone_individual` instead of `deepcopy`. If that is too
   sharp a change, document that a decorated GP toolbox must
   register the fast clone (today the decorator always
   `deepcopy`s).

**Today.** `CompileCache` evicts one LRU entry at a time instead of
clearing the whole compile cache at 1024 entries. The opcode backend
accepts a pre-packed `(rows, columns)` matrix like Numba.
`gp.warmup_numba()` sets a default `NUMBA_CACHE_DIR` when unset so
spawned workers reload the compiled interpreter from disk.
`static_limit` rejects oversized offspring with
`clone_individual` instead of `deepcopy`.

**Benefit.** Multi-hour runs with many unique trees keep their
compile working set, stop restacking columns, start faster in
worker processes, and do not recopy whole trees on every rejected
variation. No new language, no new selector.

Related: [Performance](performance.md),
[Important differences](differences.md).

---

## 10. Vectorized lexicase and plexicase

**What.** Filter lexicase cases against a dense
$(n, \textit{cases})$ error matrix instead of a Python loop of
list comprehensions. **Plexicase** (Ding, Chen, Spector) is a
tractable approximation of lexicase’s selection *distribution*:
faster, and it yields a probability that can be annealed or mixed.
Plexicase is in scope only if item 5 is not enough and profiles
show selection dominating evaluation.

**Today.** `fitness_case_matrix` packs ``fitness.values`` once per
generation. `sel_lexicase`, `sel_epsilon_lexicase`, and
`sample_informed_cases` accept optional ``matrix=`` so informed
down-sampling and lexicase filtering share one NumPy array.
Filtering is vectorized; golden RNG streams are unchanged.
Plexicase remains deferred until a profile shows selection still
dominates after downsampling (item 5) and this path.

**Benefit.** When cases are numerous (many folds, many series, or
per-point residuals), selection becomes the next bottleneck after
window kernels. A matrix filter is also the input format item 5
needs. Plexicase is an optimization of the *selector*, not a
replacement for down-sampling.

---

## 11. SMS-EMOA

**What.** Environmental selection that discards the individual
with the smallest hypervolume contribution. Steady-state or
generational. `least_contrib` and `hypervolume` already delegate
to moocore.

**Today.** `sel_sms_emoa` performs SMS-EMOA environmental selection:
non-dominated sorting, then repeated removal of the least hypervolume
contributor on the critical front via ``least_contrib`` (moocore).
Use it generational on ``parents + offspring`` or steady-state on
``parents + [child]``. Optional ``ref_point`` follows the
``hypervolume`` / ``least_contrib`` minimization-space convention.

**Benefit.** The expensive part of SMS-EMOA is already done. The
algorithm is a loop over contributions. Completes the story that
the library computes hypervolume and should be able to *search*
with it, without a new dependency.

Related: [Operators](../reference/operators.md).

---

## 12. MOEA/D and AGE-MOEA-II

**What.** Two many-objective selectors:

- **MOEA/D** — a weight-vector loop with Tchebycheff or PBI
  scalarization. NSGA-III already exposes
  `uniform_reference_points`; the missing piece is decomposition
  selection.
- **AGE-MOEA-II** — estimate front curvature (Newton–Raphson) and
  assign survival by geodesic distance. After MOEA/D: more code,
  more payoff when the front is not a simplex.

**Today.** `sel_moead` and `SelMOEADWithMemory` decompose with
Tchebycheff or PBI scalarization over `uniform_reference_points`
weight vectors. `sel_age_moea_2` and `SelAGE2WithMemory` estimate
front curvature with Newton–Raphson and break ties on the last
partial front with geodesic diversity.

**Benefit.** Many-objective users currently stop at NSGA-III.
MOEA/D is the standard decomposition method. AGE-MOEA-II is the
usual next step when NSGA-III’s simplex assumption fails on an
irregular front. Parity with the algorithms papers name, without
becoming a general MOO framework.

---

## 13. IPOP / BIPOP CMA restarts

**What.** A small algorithm next to `ea_generate_update` that
grows $\lambda$ and resets the CMA state on stagnation
(IPOP / BIPOP). It wraps the existing `Strategy` objects
(`Strategy`, `StrategyOnePlusLambda`, `StrategyMultiObjective`);
it does not add a new covariance-update variant.

**Today.** CMA, $(1+\lambda)$-CMA, and MO-CMA ship with box
constraints (`clip` / `resample`). Restarts are caller-written
wrappers. No Sep-CMA, LM-CMA, or learned step-size controller is
planned until a user hits a documented wall; high-dimension
Sep-CMA is the only plausible next variant.

**Benefit.** This is how CMA is used on hard landscapes: enlarge
the population, reset the model, continue. ES users stop writing
the same wrapper. Secondary for prefix-tree GP; first-class for
continuous search.

Related: [Strategies](../reference/strategies.md),
[Algorithms](../reference/algorithms.md).

---

## 14. Linear-time duplicate count

**What.** Rewrite `duplicate_count` to $O(n)$ for hashable keys
(a `set`, or a sort when values are ordered but unhashable). The
function’s contract stays the same: how many individuals are
duplicates of an earlier one under an optional `key`.

**Today.** The implementation is `value not in unique` on a list —
$O(n^2)$. Fine for typical hall-of-fame sizes; not for a
population-wide statistic on thousands of individuals.

**Benefit.** Housekeeping that does not change the search language.
Callers who log uniqueness on a large population get a linear
scan instead of a quadratic one. No new operator.

Related: [Utilities](../reference/utilities.md).

---

## Not planned

These ideas stay off the library surface. They are listed so the
backlog above is not read as “everything in the 2025 GP literature.”

| Idea | Why not |
|:-----|:--------|
| PushGP, Cartesian GP, or linear GP as a second public genome | Prefix trees plus the tape already linearize a program. A second representation needs a caller who cannot use trees or columnar arrays. |
| Transformer or LLM mutation | Heavy optional dependencies, unstable operators, and they do not compose with the tape. A recipe in a notebook is enough. |
| A catalog of named domain indicators as primitives | Composable windows and user opcodes. Named catalogs age badly. |
| Built-in domain fitness functions | They need application state the library does not own. |
| Learned quality-diversity / meta-BBO | A research paper, not a toolbox function. |
| Switching persistence off dill, or replacing `creator` with dataclasses | Forbidden by the project contract. |
| An in-tree C / Cython rewrite of operators, CMA, or selection | Not the bottleneck; those modules must stay readable. Revisit an *optional* tape backend only after items 3, 4, and 9 are in and a profile still points at `interpret`. |
