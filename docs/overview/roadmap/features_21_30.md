# Features 21-30

Items in this decade from the [roadmap overview](index.md).
Status and surfaces live on that table. Empty slots stay empty
until an item is numbered into this range.

---

## 21. Growing primitive language

**What.** Library evolution on the existing prefix-tree + tape
surface. A helper (for example `promote_subtree`) lifts a typed
subtree into `PrimitiveSetTyped` as a new primitive: generated
name, the subtree’s argument types and return type, and — when
the set is columnar — an opcode binding so `lower_tree` /
`interpret_tapes` see it. Later `generate` / mutation can sample
that name like any other primitive. `add_adf` stays the static
“register this other pset” path; this item is *dynamic* accretion
from successful individuals.

**Today.** `promote_subtree` lifts a complete typed subtree into
the same `PrimitiveSetTyped` as a generated name (`promo0`, …).
Formals are the set's argument terminals that appear in the
subtree; constants and ephemerals stay in the body. Later
`generate` / mutation can sample that name. The library is
capped (`max_library`); the least-used promoted name is evicted,
not a built-in. Promotion clears the compile cache. On a
columnar set the name is bound at or above `USER_BASE` and
`lower_tree` expands the body so tapes stay on builtin opcodes.
`add_adf` / `compile_adf_tree` remain the static path.

**Benefit.** Search stops reshuffling the same kit and starts
building vocabulary. The thing you keep at the end can be a
small dialect plus shallow trees, not one giant expression.

**Design notes.**

- Promotion is an operator the caller fires (threshold on
  fitness, archive cell, or frequency). Do not auto-promote
  every generation — the language bloats and the compile cache
  dies.
- The extracted node list must be a complete typed tree. Reuse
  the same closure rules as `generate()`.
- Existing `compile_tree` / tape caches key on expression text
  and context identity. A pset mutation changes context: drop
  or namespace the cache; stale lambdas are wrong.
- Columnar path: bind at or above `USER_BASE` with
  `bind_numba_opcode`. Python is the definition; `lower_tree`
  expands the body so opcode and Numba follow without a
  consumer dispatcher.
- Cap library size. Evict the least-used promoted name, not a
  built-in kit primitive.

**Scope.** Feature construction on prefix trees. Not a catalog of
named domain indicators. Not a second genome. Not an LLM that
proposes names.

Related: [Genetic programming](../../reference/gp.md),
[item 9](features_1_10.md#9-compile-and-clone-path),
[Push GP P5](push_gp.md#p5-growing-primitive-language).

---

## 22. Semantic search space

**What.** Treat `interpret_tapes(...)`’s
$(n_{\mathrm{ind}}, n_{\mathrm{rows}})$ matrix as the search
geometry, not only as a fitness source. Two concrete pieces:

1. **Semantic descriptors** — project that matrix (per-individual
   moments, case-solve bits, a PCA / random projection the
   caller supplies) into a behavior vector and `add` it to
   `GridArchive` or the item-20 archive. Variation stays SlimGP
   / ordinary GP; the archive keeps *different functions*, not
   different strings.
2. **Semantic nearest-neighbor** — optional parent or surrogate
   lookup in that matrix (cosine / Euclidean on the finite
   mask). A cheap stand-in for “what does this program *do*.”
   Not a trained QD model.

`SlimTree` + `mut_slim` already move in output space. This item
hooks that geometry to the archive and to selection.

**Today.** `semantic_moments`, `semantic_solve_bits`, and
`semantic_project` turn an `interpret_tapes` pack into a
behavior vector (moments, lexicase solve bits, or a caller
PCA / random basis). `add` those descriptors to `GridArchive`,
`CvtArchive`, or `UnstructuredArchive` — variation stays SlimGP
or ordinary GP. `semantic_nearest` does cosine or Euclidean
lookup on the finite / `valid=` mask. `SemanticSurrogate` stores
last-generation semantics for nearest or linear predict.
`ind.fitness` is not replaced. `trust_matrix=True` is the same
row-alignment footgun as lexicase.

**Benefit.** Breeding and keeping happen in the space SlimGP
already mutates. You keep a zoo of competent specialists
instead of one tree that won a scalar.

**Design notes.**

- Warmup `nan`s: reuse the `valid=` contract from `case_errors`.
  A descriptor or distance that sees warmup is a lookahead bug.
- `trust_matrix=True` stays a footgun with the same meaning as
  on lexicase — the pack must match the current individuals.
- Do not replace `ind.fitness`. The archive still ranks a cell
  by fitness; the descriptor is *which* cell.
- A linear or nearest-neighbor surrogate of last generation’s
  semantics is in scope. A learned quality-diversity model is
  not (see [Not planned](not_planned.md)).

Related: [item 7](features_1_10.md#7-non-bloating-semantic-variation),
[item 8](features_1_10.md#8-quality-diversity-archive),
[item 4](features_1_10.md#4-batch-tape-evaluation),
[Push GP P6](push_gp.md#p6-semantic-search-space).

---

## 23. Co-evolving cases

**What.** A second, cheap population whose individuals are
**case subsets** — index ranges or boolean masks in the same
shape `case_errors` and `sel_lexicase(..., cases=)` already
consume. Each generation (or each island step):

1. Score programs on the current case subset (caller’s
   `evaluate` / `evaluate_batch`).
2. Score case subsets on the current elites: how many they
   still *fool*. The library needs a convention, not a domain
   metric. Match item 5: a case is solved when its value is
   $0$; a subset’s “difficulty” is the unsolved count or a
   Hamming distance from the all-solved vector.
3. Vary the subsets (mutate ranges, flip mask runs, informed
   resample via `sample_informed_cases`). Feed the next
   lexicase call.

POET is the reference *loop*, not the deliverable. No
environment simulator, no neural teacher.

**Today.** `CaseExam` stores a subset as ranges or a 1-D bool
mask. `CaseExamPool` is the cheap second population, with an
optional caller-marked `held_out` exam. `score_case_exams`
ranks exams on elites by unsolved count or Hamming distance
from the all-solved vector (solved ≡ $0$, same as item 5).
`mut_case_ranges` jitters bounds; `mut_case_mask` flips
contiguous runs. `guard_case_exams` repairs the empty exam
and the all-solved collapse (bump size or inject `held_out`).
`next_lexicase_cases` varies the pool and returns the mutated
or guarded winner as the next `cases=` list. `informed=True`
is a guard repair path only.
Program scoring stays on the caller. Chronological splits
stay on the caller. No environment simulator, no metric
catalog.

**Benefit.** The stand-in loss cannot sit still. Programs that
memorized last generation’s cases get a new test. This is the
machine-checkable replacement for a human staring at trees.

**Design notes.**

- Store subsets as data (`list[tuple[int, int]]` or a 1-D
  `bool` mask), not as a new genome type. `creator` can wrap
  them if someone wants a hall of fame of exams.
- Chronological / walk-forward splits stay on the caller. The
  library does not invent time. It shuffles or mutates *given*
  segments.
- Reuse `fitness_case_matrix` so program selection and exam
  scoring share one pack.
- Guard against the empty exam and the “every case always
  solved” collapse (bump subset size, or inject a held-out
  segment the caller marks).
- Do not put trading labels, Sharpe, or a metric catalog here.

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 6](features_1_10.md#6-case-structured-evaluation-helper),
[Push GP P7](push_gp.md#p7-co-evolving-cases).

---

## 24. Memetic constants

**What.** Split a generation into **shape** then **numbers**.
GP / SlimGP proposes or varies the tree. A helper extracts the
numeric leaves (ephemeral floats, `Window` ints) into a vector,
runs an existing `Strategy` (or item-19 Sep-CMA) for a few
`generate` / `update` steps, and writes the repaired values
back onto those nodes. Invalidate fitness and the compile
cache for that individual. Register as something like
`tune_ephemerals(ind, strategy, n_gen=...)`.

**Today.** `tune_ephemerals(ind, strategy, evaluate, n_gen=5)`
extracts ephemeral floats and `Window` ints in documented
prefix order (`SlimTree`: `head`, then each delta), runs a
short boxed `Strategy` or `StrategySeparable` `generate` /
`update` loop, writes the repaired centroid back, and
invalidates fitness plus the compile-cache entry for the old
expression. Trials are scored with the caller's `evaluate` on
clones, or `evaluate_batch` on a pack of clones. Window
values are rounded and clamped to the ephemeral's legal
range. `mut_ephemeral` is unchanged.

**Benefit.** Symbolic structure plus a real optimizer is how
you get a law instead of a mess that interpolates. Both halves
already exist; they do not meet.

**Design notes.**

- Walk the tree (and `SlimTree.head` / deltas if you support
  it) for `Ephemeral` nodes and `Window` terminals. Order is
  part of the contract — document it, keep it stable.
- `Window` is an inclusive integer length. After CMA, round
  and clamp to the ephemeral’s legal range. A non-integer
  window is a causality bug, not a style issue.
- Box the CMA strategy to those legal ranges (`low` / `up`,
  `bound_mode="clip"`).
- Evaluation of trial vectors is the caller’s `evaluate` on a
  clone with leaves written back — or `evaluate_batch` on a
  pack of clones. Do not add a domain fitness.
- Small inner budget. This is a local polish, not a second
  full ES run per offspring.
- No in-tree Autograd / Adam. CMA is the numeric engine
  unless a later profile says otherwise.

Related: [Strategies](../../reference/strategies.md),
[item 19](features_11_20.md#19-sep-cma),
[Push GP P8](push_gp.md#p8-memetic-constants).

---

## 25. Streaming and island ecology

**What.** Two thin algorithm pieces, not a runtime product.

1. **Append-only evaluation.** A packed `(rows, columns)`
   matrix grows by rows. Re-score with `interpret_tapes` on
   the new pack (or a dirty suffix if you can prove the
   opcode is causal and has a bounded window).
   `evaluate_invalid` already prefers `evaluate_batch`.
   `Checkpoint.range` already persists a run. Document the
   recipe; add a helper only if the dirty-row bookkeeping is
   easy to get wrong (row count vs warmup vs `valid=`).
2. **Heterogeneous islands.** Several demes, each with its
   *own* registered `select` / `vary` (lexicase on one,
   `sel_sms_emoa` on another, `ea_map_elites` on a third).
   `mig_ring` already moves individuals. A small
   `step_islands(demes, migrate=...)` loop is enough: evaluate
   → vary → select on each deme, then migrate. Different
   *pressures*, not different topologies.

**Today.** `step_islands(demes, migrate=...)` runs evaluate →
vary → select on each deme, then an optional `migrate`
(usually `mig_ring`). Each deme has its own toolbox, so
lexicase, SMS-EMOA, or a MAP-Elites `vary` / `select` pair
can apply different pressures on the same generation.
Append-only evaluation is a documented recipe: grow the
packed `(rows, columns)` matrix, invalidate fitness, and
rescore with `interpret_tapes` on the full pack. A
suffix-only score is not a library path — window warmup
would be wrong without history. Migrants keep fitness when
`eval_keys` agree; distinct keys clear immigrant fitness.
`Checkpoint.range` is the caller loop. No Ray/GPU daemon.

**Benefit.** Evolution can sit on a pipe, and a population
can disagree about what “good” means. Specialists survive
because some island is still selecting for them.

**Design notes.**

- Causality: new rows are the present. A program must not
  see a row that has not arrived. Window warmup on the new
  suffix is the same `nan` contract as the unary kit.
- Full-matrix rescore is the correct default. Incremental
  kernels (item 3) make that cheap; a custom dirty-suffix
  path is optional and must match the Python oracle.
- Migrants keep their fitness only if the destination’s
  cases / matrix are the same. Otherwise invalidate.
  Cross-island archives do not merge automatically.
- No Ray/GPU runtime, no daemon. Checkpoint + caller loop.

Related: [item 4](features_1_10.md#4-batch-tape-evaluation),
[Algorithms](../../reference/algorithms.md),
[Multiprocessing](../../tutorials/multiprocessing.md),
[Push GP P9](push_gp.md#p9-streaming-and-island-ecology).

---

## 26. Program teams

**What.** Selection of a **set** of programs that covers cases
together, plus an optional router individual. `sel_team(pool,
k, matrix=)` (name flexible) treats the case-solve matrix from
`fitness_case_matrix` as a set-cover / max-coverage problem:
greedy or lexicase-style, return $k$ members whose union of
solved cases is large. A team is a sequence of individuals.
Scoring the *team* (vote, mask-router, winner-take-regime)
stays on the caller — same rule as `evaluate`.

**Today.** `sel_team` builds a team of `sel_count` individuals by
greedy maximum coverage on the case-solve matrix: a case is
solved at $0$ (`isclose` $10^{-12}$). Optional `matrix=` /
`trust_matrix=` / `cases=` match lexicase. Members are unique
pool objects; `sel_count=1` is the widest cover and does not
crash. Member `fitness` is not rewritten. Team scoring (vote,
router) stays on the caller. Cooperative coevolution remains
an example-level recipe.

**Benefit.** The thing you ship is an ensemble that covers
regimes, which is what lexicase and MAP-Elites were already
pointing at.

**Design notes.**

- Do not overwrite member `fitness` with the team score. Team
  quality is a separate value the caller assigns if they want
  a hall of fame of teams.
- `matrix=` / `trust_matrix=` match lexicase. Solved ≡ $0$.
- Router-as-tree is just another individual the caller
  evaluates. Do not add a built-in gating primitive.
- $k=1$ reduces to “pick the best coverage individual” and
  must not crash.
- Cooperative coevolution of members (species per slot) is
  allowed as an example, not required in the operator.

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 8](features_1_10.md#8-quality-diversity-archive),
[item 22](#22-semantic-search-space),
[Push GP P10](push_gp.md#p10-program-teams).

---

## 27. Batch-epsilon-lexicase and down-sampled tournament

**What.** Two selectors that reuse `fitness_case_matrix` instead of
walking every case:

- **Batch-ε-lexicase** — collapse random groups of cases into one
  reduction per batch (MSE is the usual one), then run the existing
  ε-lexicase filter on the shorter matrix. Same `matrix=` /
  `cases=` / `trust_matrix=` contract.
- **Down-sampled tournament** — `sel_tournament_cases` (name
  flexible) scores each individual on a case subset (mean of those
  columns, or the caller's reduction) and tournaments. Informed
  down-sampling stays `sample_informed_cases`.

`sel_lexicase` and `sel_epsilon_lexicase` defaults do not change.

**Today.** Lexicase and MAD-ε already filter a packed
$(n, \textit{cases})$ matrix. `sample_informed_cases` builds a
subset. Tournament still reads a scalar `wvalues[0]`. There is no
batch reduction of cases and no tournament that sees a case
subset.

**Benefit.** Batch-ε-lexicase is the usual next lexicase variant
on noisy regression (Geiger et al.). Tournament plus down-sampling
is the fast path that recent symbolic-regression comparisons put
next to ε-lexicase. Both buy more individuals per evaluation
budget without plexicase.

**Scope.** Case-matrix reductions and one tournament wrapper. Not
a third ad-hoc lexicase family. Plexicase stays deferred (item 10)
until a profile shows selection still dominates after this and
[item 28](#28-dynamic-epsilon-and-downsample-schedule).

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 10](features_1_10.md#10-vectorized-lexicase-and-plexicase).

---

## 28. Dynamic epsilon and downsample schedule

**What.** Two small policies on the existing lexicase path:

1. **Dynamic / semi-dynamic ε** — recompute the elite error and/or
   per-case MAD on the *current filter pool*, not only on the whole
   population (La Cava). A `mode=` on the vectorized filter, not a
   third selector.
2. **Downsample schedule** — a helper that returns the next
   `cases=` list each generation: random, informed, cohort, or
   rotate-through-held-out. Chronological meaning stays on the
   caller; this is index policy.

**Today.** `sel_epsilon_lexicase` with `epsilon=None` uses static
per-case MAD on the packed matrix. `sample_informed_cases` and
`next_lexicase_cases` build one subset. There is no filter-pool
ε and no generation-indexed schedule object.

**Benefit.** Static MAD is elite on easy cases and slack on hard
ones in a way that does not track the remaining pool. A schedule
turns exams + lexicase into a one-liner instead of a hand-rolled
index dance.

**Scope.** Filter-pool ε and an index schedule. Not a split
algorithm, not timestamps, not a metric catalog.

Related: [item 5](features_1_10.md#5-down-sampled-and-informed-lexicase),
[item 23](#23-co-evolving-cases),
[item 27](#27-batch-epsilon-lexicase-and-down-sampled-tournament).

---

## 29. Novelty selection and iso+line

**What.** The minimum variation / selection pair that makes a
MAP-Elites archive *search* behavior space:

- `sel_novelty` ranks by distance to an archive (reuse
  `semantic_distance` / `UnstructuredArchive` neighbors). Fitness
  stays on `ind.fitness`; novelty is the selection key.
- `mut_iso_line` (name flexible): pick a donor elite, interpolate,
  add isotropic noise. Discrete / mixed encodings get the same
  recipe on the gene types they already have.

`ea_map_elites` can register them like any other `select` /
`mutate`. `random_elites` stays the parent source.

**Today.** `GridArchive`, `CvtArchive`, and `UnstructuredArchive`
store elites. `ea_map_elites` varies with generic `var_or`.
`semantic_nearest` looks up neighbors. There is no novelty
selector and no archive-aware variation operator.

**Benefit.** An archive that only `add`s and then runs ordinary
crossover is a hall of fame with bins. Iso+line and novelty are
how MAP-Elites papers actually move in descriptor space. Item 22's
semantic geometry gets a selector that lives in that space.

**Scope.** One selector and one mutator. Dominated novelty, a
Pareto-per-cell archive, and CMA-MAE thresholds stay in
[Under consideration](under_consideration.md) until this pair exists.

Related: [item 8](features_1_10.md#8-quality-diversity-archive),
[item 20](features_11_20.md#20-cvt-unstructured-map-elites),
[item 22](#22-semantic-search-space).

---

## 30. Causal lookback and suffix rescore

**What.** The correct form of the incremental path item 25
deferred. Each opcode declares a finite lookback (the `Window`
arg, `delay` steps, `ema` warmup). A helper
`tape_lookback(tape) -> int` returns the program's bound. A
second helper rescores only `lookback + n_new` trailing rows
after an append-only `vstack` and writes them back onto the
cached prefix so the full series matches a full-matrix
`interpret_tapes` oracle.

**Today.** Append-only evaluation is a documented full-matrix
rescore. A suffix-only score is not a library path — window
warmup would be wrong without history.

**Benefit.** Evolution can sit on a pipe without replaying the
whole history every generation. The lookback certificate is
what makes a dirty suffix legal; without it, incremental
rescore is a lookahead bug.

**Scope.** Lookback metadata plus a suffix rescore that matches
the Python oracle, including `nan` warmup. Not a Ray/GPU
daemon. Not a custom dirty-row protocol per caller.

Related: [item 3](features_1_10.md#3-incremental-window-kernels),
[item 4](features_1_10.md#4-batch-tape-evaluation),
[item 25](#25-streaming-and-island-ecology),
[Push GP P16](push_gp.md#p16-causal-lookback-and-suffix-rescore).

---
