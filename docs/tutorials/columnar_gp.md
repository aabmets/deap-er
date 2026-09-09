# Columnar Programs

The [genetic programming](../reference/gp.md) package can evolve programs
over a whole table of numeric columns instead of one scalar at a time. A
compiled tree takes one array per column and returns one array, so a
population of a few hundred programs can be scored against hundreds of
thousands of samples without a Python loop over rows.

This tutorial covers the columnar primitive set, the vectorized and
window primitive kits, the three evaluation backends, and the rules you
have to respect to keep the results honest.

## The data contract

Convert your data **once**, before the evolution starts, into one
`float64` array per column:

- one-dimensional, C-contiguous, and all of the same length
- missing values as `numpy.nan`, not as zero or a sentinel
- integer counts promoted to `float64`
- identifiers, labels, and timestamps left out of the primitive set

Anything else — loading, joining, resampling, and the fitness function
itself — stays on your side. The library only owns the step from a tree
to an array.

## Building the primitive set

`make_column_pset` creates a strongly typed set with one input per
column. Argument order *is* column order, so `pset.arguments[i]` names
the array you pass in position `i`.

```python
import numpy
from deap_er import gp

names = ["level", "flow", "noise"]

pset = gp.make_column_pset(names)
gp.add_numpy_primitives(pset)
gp.add_window_primitives(pset)
gp.add_window_ephemeral(pset, "window", 2, 64)
```

Column names have to be plain Python identifiers, because a compiled
tree becomes a lambda over them. Keywords, duplicates, and names that
look like the default `ARG0` prefix are rejected.

!!! attention
    A column may not share a name with a primitive. `level` is fine,
    `vadd` is not: the lambda parameter would shadow the operator and
    the tree would quietly compute the wrong thing. Both
    `add_numpy_primitives`, `add_window_primitives`,
    `add_pair_window_primitives`, and `add_ts_primitives` raise instead
    of letting that happen.

### Type tags

Three tag classes drive the strongly typed search. They are markers for
the type system only — the values that flow at runtime are ordinary
NumPy arrays and Python integers.

| Tag | Runtime value | Role |
|:----|:--------------|:-----|
| `Array` | 1D `float64` | a column, or anything derived from one |
| `Mask` | 1D `bool` | a condition |
| `Window` | `int` | a rolling length or a delay |

Keeping `Mask` apart from `Array` is what makes `vwhere` well defined
and stops a condition from being added to a series. No primitive returns
a `Window`, so a window is always a leaf.

## The vectorized kit

`add_numpy_primitives` registers arithmetic (`vadd`, `vsub`, `vmul`,
`vneg`, `vabs`, `vsin`, `vcos`), protected operations (`vdiv`, `vlog`,
`vsqrt`), comparisons (`vgt`, `vlt`, `vge`, `vle`, `veq`), mask logic
(`vand`, `vor`, `vnot`), and `vwhere`.

`vwhere` is the important one: it is how a program turns a condition
into a value without a Python `if`, which is what keeps the whole tree
in vectorized code.

The protected operations replace a non-finite result with a fill, by
default `1.0`:

```python
gp.vdiv(numpy.array([1.0]), numpy.array([0.0]))  # -> array([1.])
gp.vdiv(numpy.array([1.0]), numpy.array([0.0]), fill=0.0)  # -> array([0.])
```

They only replace a value that the operation itself fabricated. A
`nan` that arrived in an operand comes back out as `nan`, so the warmup
written by the window primitives is never overwritten by a made-up
number.

## Causal window primitives

`add_window_primitives` registers `delay`, `diff`, `rolling_sum`,
`rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, and `ema`.
Each takes an `Array` and a `Window`.

Every one of them is **causal**: a sample of the output depends on that
sample and earlier ones, never on a later one. The rolling window is
`[t - n + 1, t]` inclusive.

```python
values = numpy.arange(10, dtype=numpy.float64)

gp.delay(values, 2)         # [nan nan 0. 1. 2. 3. 4. 5. 6. 7.]
gp.rolling_mean(values, 3)  # [nan nan 1. 2. 3. 4. 5. 6. 7. 8.]
```

Samples without enough history are `nan`, never zero. Zero would be a
value your fitness function could act on, and it would be a value the
program never actually had.

Window lengths are ephemeral constants, sampled once per tree:

```python
gp.add_window_ephemeral(pset, "window", 2, 64)
```

The bounds are inclusive. The name must be unique across the process,
because a primitive set stores ephemeral types by name.

After the shape of a tree is fixed, `tune_ephemerals` can polish
those numeric leaves with a few boxed CMA `generate` / `update`
steps. Walk order is prefix list order — on a `SlimTree`, `head`
then each delta. Window values are rounded and clamped back to
the inclusive `[low, high]` the ephemeral was registered with.
Evaluation stays on the caller: pass `evaluate`, or
`evaluate_batch` to score a pack of clones. This is a local
polish, not a second full ES run.

`affine_scale(predicted, target, *, valid=)` fits Keijzer
$a + b\,f(x)$ on the same mask `case_errors` uses. Prefer
`affine_case_errors(predicted, target, cases, *, valid=)` when
writing fitness or lexicase vectors — it applies the scaled series
without touching the tree (Darwinian). Call `write_affine_scale`
only when you explicitly want Lamarckian writeback. The tree shape
stays the same either way.

For memetic polish outside the policy loop, use
`gp.tune_ephemerals_budget` instead of raw `tune_ephemerals`.
It defaults to `MEMETIC_DEFAULT_N_GEN`, caps inner generations to
remaining `n_evals`, and requires a held-out judge when a
`CaseExamPool.held_out` exam is marked.

`add_pair_window_primitives` is an optional second kit for two series
and one `Window`: `rolling_corr`, `rolling_cov`, and `rolling_beta`.
Moments use the same population divisor as `rolling_std`. Beta is the
OLS slope of the first series on the second
($\mathrm{cov}(x, y) / \mathrm{var}(y)$). A window whose denominator
variance is zero is `nan`, not the protected-op fill.

```python
gp.add_pair_window_primitives(pset)
gp.rolling_beta(level, flow, 4)
```

`add_ts_primitives` is an optional kit for rank and arg-extremum
inside one trailing window: `ts_rank`, `ts_argmax`, and `ts_argmin`.
Each takes an `Array` and a `Window`. `ts_rank` is the average rank
of the current sample, scaled so a unique window low is $0$ and a
unique high is $1$. A window of 1 is `nan`. `ts_argmax` / `ts_argmin`
are how many samples ago the extreme occurred — $0$ means now — and
a tie keeps the most recent extreme.

```python
gp.add_ts_primitives(pset)
gp.ts_rank(level, 8)
gp.ts_argmax(flow, 8)
```

!!! attention
    A comparison hides the warmup. `numpy.nan > x` is `False`, so
    `vwhere(vgt(rolling_mean(level, 20), flow), a, b)` returns a real
    value at sample zero even though the average has not warmed up
    yet. Decide in the fitness function which prefix of the result to
    trust — for example by ignoring samples where the target or the
    prediction is not finite — instead of assuming the `nan` propagates
    all the way out.

## Evolving

`columnar_pset` is the one-call kit: `make_column_pset` plus the
NumPy and window primitives, and a window ephemeral. Pass
`pair_windows=True` / `ts=True` for those kits.

`register_gp` wires the aliases people forget — `clone_individual`
instead of `deepcopy`, `compile_tree`, half-and-half init, one-point
crossover, uniform mutation, and a height `static_limit`. Fitness
stays on you. `evaluate_columnar` is the `evaluate_batch` helper:
unique trees are lowered once, scored with `interpret_tapes`, and
warmup `nan` samples are dropped from the MSE. With the default
`static_filter=True`, `tape_flags` may skip identically-`nan`,
constant, or warmup-hiding `vwhere` programs before
`interpret_tapes` and write the `empty` sentinel instead.
Pass `static_filter=False` to score every tree. The runtime warmup
contract is unchanged — use `case_errors(..., valid=)` when a
comparison can still hide warmup that the static pass misses.

```python
from deap_er import Fitness, Toolbox, creator, gp, tools

pset = gp.columnar_pset(["level", "flow"], window=(2, 64))
creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = Toolbox()
gp.register_gp(toolbox, pset, individual=creator.Individual, backend="opcode")
toolbox.register(
    "evaluate_batch",
    gp.evaluate_columnar,
    pset=pset,
    matrix=matrix,
    target=target,
)
```

You can still register the pieces yourself. Tree nodes are immutable
once created, so an offspring only needs a new list and a fresh
fitness:

```python
from deap_er import Toolbox, gp, tools

toolbox = Toolbox()
toolbox.register("clone", tools.clone_individual)
toolbox.register("compile", gp.compile_tree, prim_set=pset)
```

A wide table has many more terminals than primitives, which pushes
`gen_grow` towards small trees. Reach for `gen_full`, or raise
`min_depth`, when the programs come out shorter than you want.

## Backends

`compile_tree` takes a `backend` argument. All three read the same
columns in the same order and agree on the same results.

| Backend | What it does |
|:--------|:-------------|
| `python` | evaluates the expression text. The default, and the reference |
| `opcode` | lowers the tree to a flat tape and runs it on a NumPy stack machine |
| `numba` | runs the same tape through a compiled interpreter, from the `numba` extra |

```python
func = gp.compile_tree(tree, pset)                     # python
func = gp.compile_tree(tree, pset, backend="opcode")
func = gp.compile_tree(tree, pset, backend="numba")

result = func(level, flow, noise)
```

The tape backends reject a tree while lowering it if any of its
primitives has no opcode, so an unsupported operator is an error before
the first sample is touched. Both accept the columns positionally or
as one prepacked `(n_rows, n_columns)` matrix, which avoids repacking
them on every call.

The Numba interpreter is compiled once per process, never once per
tree. Every compiled tape shares one process-wide workspace, so a
`bind_tape` callable must not be run from several threads at once.

To score a generation against one packed `(n_rows, n_columns)`
matrix, lower unique trees and call `interpret_tapes`. Cache by
`str(tree)` and lower the **tree object** —
`PrimitiveTree.from_string` cannot round-trip a `Window` ephemeral.

```python
def evaluate_batch(individuals):
    unique = {}
    tapes = []
    index = []
    for ind in individuals:
        key = str(ind)
        slot = unique.get(key)
        if slot is None:
            unique[key] = slot = len(tapes)
            tapes.append(gp.lower_tree(ind, pset))
        index.append(slot)
    predicted = gp.interpret_tapes(tapes, matrix, backend="numba")
    return [score(predicted[i], target) for i in index]
```

`interpret_tapes` hash-conses postfix subexpressions across the batch:
shared suffixes are evaluated once and stitched back. The return shape,
warmup `nan` contract, and per-tape `fill` semantics are unchanged.

When new rows arrive, grow the packed table and **rescore the full
matrix**. Do not score only the new rows: a causal window needs the
preceding samples, and a suffix-only call treats the first new row as
$t = 0$ (warmup `nan`s land on the wrong bars).

```python
matrix = numpy.vstack([matrix, new_rows])
for ind in population:
    del ind.fitness.values
# evaluate_batch calls interpret_tapes(tapes, matrix) on the full pack
```

New rows are the present. A program must not see a row that has not
arrived. After the append, prefix outputs of a causal tree match the
scores from before the append. Use `case_errors(..., valid=)` for
hidden warmup / `vwhere`, same as a static book. Persist the caller
loop with `Checkpoint.range`. There is no streaming daemon.

If `score` returns a vector of case errors, lexicase can filter on a
subset rebuilt each generation. Do not freeze `cases=` on the
toolbox. A case is solved when its value is exactly $0$. Continuous
residuals and maximize-only scores need an explicit `solved`
predicate — without one, every case looks unsolved and the subset
is a random fill.

```python
def select(individuals, sel_count):
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.sample_informed_cases(individuals, 20, matrix=matrix)
    return tools.sel_lexicase(individuals, sel_count, cases=cases, matrix=matrix)

toolbox.register("select", select)
```

For batch ε-lexicase, group cases into batches and filter on the
shorter matrix. For a down-sampled tournament, score each individual
on a case subset and run ordinary tournament selection on those
scalars:

```python
def select_batch(individuals, sel_count):
    matrix = tools.fitness_case_matrix(individuals)
    return tools.sel_batch_epsilon_lexicase(
        individuals, sel_count, batch_size=4, matrix=matrix
    )

def select_tournament(individuals, sel_count):
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.sample_informed_cases(individuals, 12, matrix=matrix)
    return tools.sel_tournament_cases(
        individuals, rounds=sel_count, contestants=3, cases=cases, matrix=matrix
    )
```

`next_downsample_cases` returns the next `cases=` list each generation
(`random`, `informed`, `cohort`, or `held_out`). Pass
``mode=`` on ``sel_epsilon_lexicase`` when the filter pool should
drive ε (`epsilon_semi`, ``epsilon_dynamic``).

```python
def select_scheduled(individuals, sel_count, generation):
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.next_downsample_cases(
        individuals, case_count=12, generation=generation, mode="informed"
    )
    return tools.sel_epsilon_lexicase(
        individuals,
        sel_count,
        cases=cases,
        matrix=matrix,
        mode="epsilon_dynamic",
    )
```

`sel_team` turns the same packed matrix into a covering ensemble:
greedy maximum coverage of cases solved at $0$. Member `fitness` is
not rewritten; score the team (vote, router, winner-take-regime) on
the caller.

```python
def select_team(individuals, sel_count):
    matrix = tools.fitness_case_matrix(individuals)
    return tools.sel_team(individuals, sel_count, matrix=matrix)
```

To co-evolve the exam with the programs, keep a cheap
`CaseExamPool` of catalog subsets and rebuild `cases=` from
elites each generation. Do not freeze that list on the
toolbox. A case is still solved when its value is exactly $0$.

```python
pool = tools.CaseExamPool(
    [tools.CaseExam.from_cases(range(8), 20)],
    held_out=tools.CaseExam.from_cases([19], 20),
)

def select(individuals, sel_count):
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.next_lexicase_cases(
        pool, individuals, matrix=matrix, case_count=8
    )
    return tools.sel_lexicase(individuals, sel_count, cases=cases, matrix=matrix)
```

The default path and `informed=False` both keep that varied
winner. `informed=True` only lets the guard fill an empty or
collapsed exam with `sample_informed_cases`. Walk-forward or
chronological splits stay on the caller. The library only
mutates given segments and guards an empty or all-solved exam.

## Case-structured fitness

When fitness is one error per segment — walk-forward folds, regimes,
or any case split you define — reduce the aligned series with
`tools.case_errors`. It returns one mean-squared error per case for
lexicase or a multi-objective fitness vector:

```python
predicted = func(*columns)
errors = tools.case_errors(predicted, target, [(0, 256), (256, 512)])
return errors  # assign directly to fitness.values for lexicase
```

Pass a one-dimensional `bool` mask instead of explicit ranges when
each contiguous run of `True` should be its own case. Integer arrays
are not accepted as case boundaries unless they are a `(n_cases, 2)`
table of half-open bounds. Use the optional `valid` mask when a
comparison or `vwhere` can hide warmup: it is intersected with the
finite check, so a sample must still be finite in both series even
when `valid` is `True`. Exclude hidden-warmup prefixes explicitly
rather than assuming `nan` propagates through the tree.

Segment boundaries and prediction–target alignment stay on the caller.
This helper does not choose a chronological split and does not ship
application-specific metrics beyond per-case MSE.

## Generalization path

The default case-structured recipe keeps a caller-marked held-out
exam, runs lexicase on train cases only, and optionally spends a
case budget with successive halving before full scoring. Chronological
meaning, embargo, and expanding windows stay on the caller — only
catalog indices are split here.

```python
N_CASES = matrix.shape[0]
recipe = tools.case_generalization_recipe(N_CASES, fraction=0.2)
toolbox.register("select", recipe.make_select(downsample=32))

def evaluate_cases(individual, cases):
    return gp.evaluate_columnar([individual], pset, matrix, target, cases=cases, reduce=False)[0]

halving = tools.evaluate_case_halving(
    offspring,
    evaluate_cases,
    recipe.train_cases,
    n_cases=N_CASES,
    evaluate_full=lambda ind: toolbox.evaluate(ind),
)
used += halving.nevals
```

`held_out_tail` and `case_generalization_pool` build the held-out
marker. `make_lexicase_train_select` never passes held-out indices
to lexicase. `evaluate_case_halving` ranks on prefixes of
`train_cases`, assigns full-catalog `fitness.values` on the final
rung, and returns `nevals` in case-eval units for a tight `n_evals=`
budget. For partial scoring without re-running the full tape batch,
pass an `evaluate_cases` that calls `evaluate_columnar(..., cases=)`.

The same `interpret_tapes` pack is also search geometry. Project it
into a behavior vector and `add` the result to a MAP-Elites archive
so the archive keeps different *functions*, not different strings.
`ind.fitness` still ranks the cell. Pass the same `valid=` warmup
mask you use for `case_errors` — a descriptor or distance that sees
warmup is a lookahead bug. `trust_matrix=True` means the pack is
row-aligned with the current individuals, the same footgun as
lexicase.

```python
predicted = gp.interpret_tapes(tapes, matrix, backend="numba")
descriptors = tools.semantic_moments(predicted, valid=warmup)
for individual, descriptor in zip(individuals, descriptors, strict=True):
    archive.add(individual, descriptor)
parent = individuals[tools.semantic_nearest(predicted[0], predicted, k=1)[0]]
surrogate = tools.SemanticSurrogate()
surrogate.update(predicted, [ind.fitness.wvalues[0] for ind in individuals])
guess = surrogate.predict(predicted[0], kind="nearest")
```

`semantic_solve_bits` is the lexicase convention: a case is solved
when its MSE is exactly $0$. Continuous residuals usually want
moments or a caller PCA / random projection (`semantic_pca_basis`,
`semantic_random_basis`, `semantic_project`). High-dimensional solve
bits fit `CvtArchive` or `UnstructuredArchive` better than a grid.
`SemanticSurrogate.predict` is last-generation nearest or linear
lookup, not a learned quality-diversity model.

### Team from archive

Keep a *book* of programs, not one hero per cell: project semantics,
`add` to the archive, then assemble a covering team from occupied
cells. `GridArchive.add` ranks on **single-objective** `fitness`; when
`fitness.values` is per-case, pass the case-solve matrix explicitly
to `sel_team` or `sel_team_archive`. Team scoring (vote, router,
held-out exam) stays on the caller — not `step_program_search`.

```python
predicted = gp.interpret_tapes(tapes, matrix, backend="numba")
descriptors = tools.semantic_project(pack, basis, valid=warmup, center=center)
for individual, descriptor in zip(individuals, descriptors, strict=True):
    archive.add(individual, descriptor)

case_matrix = tools.fitness_case_matrix(list(archive))  # or caller-built matrix=
team = tools.sel_team_archive(archive, sel_count, matrix=case_matrix)
```

`sel_team_archive` pools `list(archive)` (live elites, not
`random_elites` copies) and delegates to `sel_team`. Member
`fitness` is not rewritten.

`parallel=True` evaluates those tapes on several Numba threads. Each
thread keeps a workspace of shape `(depth + 1, n_rows)`, so a long
book costs `n_threads` full-length stacks. A consumer `dispatch`
kernel must be safe on those stacks at once — no process-global
buffer. Leave `parallel` off for a multi-year one-minute series;
turn it on for tens or hundreds of thousands of bars.
`evaluate_batch` still owns any process pool, as described in the
[multiprocessing tutorial](multiprocessing.md).

The Numba backend has no way to size a result for a primitive set
that takes no arguments, since every value it holds is one column
long. Use the `python` or `opcode` backend for a constant-only set.

!!! note
    A tree that is nothing but a column returns that column itself
    rather than a copy, and a tree that is nothing but a constant
    returns a scalar. Do not write into a result you did not build.

### Custom compiled kernels

A consumer can add its own compiled kernels without forking the
interpreter. Register the primitive as usual so that the default
backend and the type checks keep working, bind its name to an opcode at
or above `USER_BASE`, and pass one compiled dispatcher that implements
those opcodes.

```python
import numba

OP_SMOOTH = gp.USER_BASE + 1

pset.add_primitive(smooth, [gp.Array], gp.Array, "smooth")
gp.bind_numba_opcode("smooth", OP_SMOOTH)

@numba.njit(cache=True, error_model="numpy")
def dispatch(op, sp, stack, columns, constants, scratch):
    if op == OP_SMOOTH:
        ...
        return sp
    return -1

func = gp.compile_tree(tree, pset, backend="numba", dispatch=dispatch)
```

`gp.USER_DISPATCH_SIGNATURE` documents the stack layout the kernel has
to follow. Builtin window primitives fold their length into the
instruction, but a custom primitive receives every argument on the
stack, so a window arrives as a broadcast row. A kernel may freely
clobber `scratch` and the free row at `stack[sp]`.

Keep the two implementations honest against each other: treat the
Python primitive as the definition and compare the two backends on the
same trees. Persist the bindings from `gp.numba_opcodes()` with a run,
because a replayed tree decodes against whatever bindings exist at the
time.

!!! attention
    Fees, portfolio state, penalties, and anything else that allocates
    Python objects belong in the fitness function, not in a kernel.
    The dispatch signature only carries `float64` columns, constants,
    and scratch space.

### Growing the language

`promote_subtree` lifts a complete typed subtree into the same
primitive set as a generated name (`promo0`, …). The caller fires
it — from a fitness threshold, an archive cell, or a frequency
count. Do not auto-promote every generation: the language bloats
and the compile cache is dropped on each mutation.

Formals are the column arguments that appear in the subtree.
Constants and window lengths stay baked into the body. Later
`generate` and mutation can sample the new name like any other
primitive. `max_library` caps how many promoted names are kept;
the least-used promoted name is evicted, never a built-in kit
primitive. Evicted names are not reused.

On a columnar set the name is bound at or above `USER_BASE`.
`lower_tree` expands the body, so `interpret_tapes` stays on
builtin opcodes and does not need a consumer dispatcher. Persist
`gp.numba_opcodes()` with the run, the same as a hand-bound
kernel. `add_adf` remains the static “register this other pset”
path.

```python
name = gp.promote_subtree(pset, tree)
func = gp.compile_tree(
    gp.PrimitiveTree.from_string(f"{name}(level, flow)", pset),
    pset,
    backend="opcode",
)
```

## Limitations

- `PrimitiveTree.from_string` cannot round-trip an ephemeral of a custom
  type such as `Window`, because it parses the literal back as an `int`.
  Checkpoint the trees themselves rather than their text.
- `compile_adf_tree` runs on the default backend only.
- The Numba backend shares one workspace across every compiled tape,
  so a `bind_tape` callable must not be run from several threads at
  once. `interpret_tapes(..., parallel=True)` allocates thread-local
  stacks for that call only.

## Related

- [Columnar Programs example](../examples/genetic_programming/columnar_gp.md)
  (includes the [batch evaluation](../examples/genetic_programming/columnar_gp.md#batch-evaluation) script)
- [Multiprocessing](multiprocessing.md)
- [Genetic Programming reference](../reference/gp.md)
