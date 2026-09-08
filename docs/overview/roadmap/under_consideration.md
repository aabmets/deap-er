# Under consideration

Ideas that compose the same pieces as the numbered backlog, but
are not on it yet. They wait on a profile, a sequel to a planned
item, or a recipe the caller can write today. Promotion onto the
[overview](index.md) table needs a documented gap after the item
they depend on — not a catalog of every named algorithm.

The twelve items proposed as the next backlog are already
numbered 27–38. This page is the rest.

| Idea | Why later |
|:-----|:----------|
| [Plexicase](#plexicase) | Item 10 deferred it until a profile shows selection dominating after matrix lexicase, [item 27](features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament), and [item 28](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule). |
| [Structural meta-case regularization](#structural-meta-case-regularization) | Extra case columns. A recipe on `fitness_case_matrix` until the lexicase schedules exist. |
| [Dominated novelty search](#dominated-novelty-search) | Same family as [item 29](features_21_30.md#29-novelty-selection-and-isoline). |
| [Multi-objective MAP-Elites](#multi-objective-map-elites) | Sequel to items 29 and 33. `ParetoFront` already exists. |
| [CMA-MAE and MO-CMA-MAE](#cma-mae-and-mo-cma-mae) | Sequel to [item 33](features_31_40.md#33-archive-improving-cma). |
| [Phenotypic probe descriptors](#phenotypic-probe-descriptors) | `interpret_tapes` on a short probe plus `semantic_project`. |
| [Incremental ts_rank](#incremental-ts_rank) | Last $O(\textit{window})$ Numba scan. Kernel polish. |
| [Interval analysis on tapes](#interval-analysis-on-tapes) | Secondary to a legal suffix rescore ([item 30](features_21_30.md#30-causal-lookback-and-suffix-rescore)). |
| [Homologous and semantic crossover](#homologous-and-semantic-crossover) | After affine scaling and tape CSE. |
| [Index-only walk-forward builder](#index-only-walk-forward-builder) | Item 6 already refused to own splits. |
| [Stochastic ranking and epsilon-level](#stochastic-ranking-and-epsilon-level) | Second and third constraint rules after [item 32](features_31_40.md#32-constraint-dominance-on-remaining-selectors). |
| [IBEA and HypE](#ibea-and-hype) | Duplicates SMS-EMOA's indicator story. |
| [GDE3 and NSDE](#gde3-and-nsde) | `mut_de` plus NSGA survival is a recipe. |
| [AGE-MOEA-II+](#age-moea-ii) | Curvature tweak on a shipped selector. |
| [Extra DE trial recipes](#extra-de-trial-recipes) | Parameters of [item 35](features_31_40.md#35-adaptive-de-strategy). |
| [Active CMA and mirrored sampling](#active-cma-and-mirrored-sampling) | Flags on `Strategy`. |
| [Mixed-integer CMA](#mixed-integer-cma) | Sibling of shipped boxed CMA, not program search. |
| [SNES and CEM](#snes-and-cem) | Wait until a user hits a wall after sep-CMA. |
| [Adaptive operator rates](#adaptive-operator-rates) | After [item 37](features_31_40.md#37-evaluation-budget-and-eval-cache). |
| [Batched var_and uniforms](#batched-var_and-uniforms) | Housekeeping for the tiny `ea_simple` bar. |
| [Island topologies](#island-topologies) | `step_islands` already accepts any `migrate`. |
| [Noisy fitness resample](#noisy-fitness-resample) | After the eval cache so repeats are cheap. |
| [Persistent hall of fame](#persistent-hall-of-fame) | `Checkpoint` already dill-dumps arbitrary state. |
| [WFG and constrained DTLZ](#wfg-and-constrained-dtlz) | Benchmarks, not library surface. |
| [`step_program_search`](#step_program_search) | A wrapper that wants to become a framework. |

Ideas that are off the library entirely live on
[Not planned](not_planned.md).

---

## Plexicase

**What.** Ding, Chen, and Spector's tractable approximation of
lexicase's selection *distribution*: faster, and it yields a
probability that can be annealed or mixed.

**Today.** `sel_lexicase` and `sel_epsilon_lexicase` filter a
packed case matrix. Item 10 already said plexicase waits until
down-sampling and that path are not enough.

**Why later.** Ship [item 27](features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament)
and [item 28](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule)
first. If a columnar run with many cases still spends its time
in `lexicase_select_vectorized`, then plexicase.

---

## Structural meta-case regularization

**What.** Extra cheap cases — size, depth, unique opcodes,
promote-library hits — appended to the case matrix so lexicase
regularizes bloat without a second fitness weight.

**Today.** `fitness_case_matrix` packs `fitness.values`. Callers
can already concatenate columns.

**Why later.** A recipe on the matrix until the lexicase
schedules exist. Machine-checkable pressure, not
human-in-the-loop.

---

## Dominated novelty search

**What.** Rank by non-dominated novelty-plus-fitness (GECCO 2025)
instead of novelty alone.

**Today.** [Item 29](features_21_30.md#29-novelty-selection-and-isoline)
is the planned novelty selector. `semantic_distance` and
`UnstructuredArchive` already exist.

**Why later.** Same family as item 29. More mechanism after
`sel_novelty` exists.

---

## Multi-objective MAP-Elites

**What.** Keep a Pareto front *per cell* instead of one elite
(MOME). `add` inserts into that cell's front; `random_elites`
samples from occupied cells.

**Today.** Archives keep one individual per cell. `ParetoFront`
is a separate record.

**Why later.** Sequel to items 29 and 33. The archive has to
search before it needs a front per bin.

---

## CMA-MAE and MO-CMA-MAE

**What.** CMA-MAE's decaying improvement threshold, and
hypervolume improvement per cell as the CMA objective
(MO-CMA-MAE).

**Today.** [Item 33](features_31_40.md#33-archive-improving-cma)
is improvement-or-new-cell. `hypervolume` / `least_contrib`
already delegate to moocore.

**Why later.** Thresholds and HV-per-cell once the
`ArchiveStrategy` wrapper exists. Still not a learned QD model.

---

## Phenotypic probe descriptors

**What.** Apply each program to a short probe matrix and project
the outputs (`semantic_project`) into a behavior vector. QDGP's
phenotypic characterization, as a helper.

**Today.** `interpret_tapes` plus `semantic_moments` /
`semantic_project` already do this if the caller builds the
probe.

**Why later.** A named helper after archive search is real.
Still not a domain metric.

---

## Incremental ts_rank

**What.** Replace the per-window Numba scan for `ts_rank` with
a sliding-window rank structure. Same public opcode, same
Python oracle.

**Today.** Item 3 made the other rolling kernels $O(\text{rows})$.
`ts_rank` is still a per-window scan.

**Why later.** Kernel polish, not a new loop. Parity tests
against the existing oracle.

---

## Interval analysis on tapes

**What.** Given column bounds (or empirical min/max), propagate
intervals through the opcode kit. Flag programs that are
identically `nan`, constant, or that use `vwhere` to hide
warmup.

**Today.** Causality is a runtime `nan` contract. There is no
static range check.

**Why later.** Secondary to a legal suffix rescore
([item 30](features_21_30.md#30-causal-lookback-and-suffix-rescore)).
Not a domain fitness.

---

## Homologous and semantic crossover

**What.** Align similar subtrees (homologous) or prefer nodes
whose `interpret_tape` vectors are close (`semantic_nearest` on
subtrees). Type-matched one-point stays the default.

**Today.** `gp.cx_one_point` groups by return type. SlimGP
already moves in output space. Syntactic GP still swaps random
typed nodes.

**Why later.** After affine scaling and population tape CSE, when
subtree semantics are cheap to look up.

---

## Index-only walk-forward builder

**What.** `ranges_from_folds(n, k, embargo=0, purge=0)` — expanding
windows and embargo gaps as half-open index ranges. No
timestamps.

**Today.** `case_errors` accepts ranges or a mask.
`CaseExam` stores those shapes. Chronological splits stay on
the caller.

**Why later.** Item 6 already refused to own splits. Pure index
arithmetic is easy for the caller.

---

## Stochastic ranking and epsilon-level

**What.** Runarsson and Yao's stochastic ranking, and Takahama's
ε-constrained comparison. Both are comparison rules next to
`constraint_dominates`.

**Today.** Deb's rule is on `sel_nsga_2` and planned for the
other selectors ([item 32](features_31_40.md#32-constraint-dominance-on-remaining-selectors)).
`DeltaPenalty` remains the penalty path.

**Why later.** Second and third rules after the one already
chosen is on every selector.

---

## IBEA and HypE

**What.** Indicator-based environmental selection: binary ε
(IBEA) or estimated hypervolume (HypE) instead of crowding.

**Today.** `sel_sms_emoa` already searches with hypervolume
contribution via moocore.

**Why later.** Catalog creep. SMS-EMOA is the indicator selector
the library chose.

---

## GDE3 and NSDE

**What.** Differential-evolution trials with NSGA-II or NSGA-III
survival (GDE3 / NSDE).

**Today.** `mut_de` writes a trial. `sel_nsga_2` / `sel_nsga_3`
already survive. Item 17 scoped out `ea_de` on purpose.

**Why later.** A documented `var_or` recipe, not a new algorithm
family.

---

## AGE-MOEA-II+

**What.** Several curvature hypotheses; pick the closest. Aimed
at mixed concave/convex fronts.

**Today.** `sel_age_moea_2` already estimates front curvature
with Newton–Raphson.

**Why later.** A tweak on a selector that just shipped.

---

## Extra DE trial recipes

**What.** Current-to-pbest/1 and current-to-best/1 next to
DE/rand/1/bin.

**Today.** `mut_de` is rand/1/bin. [Item 35](features_31_40.md#35-adaptive-de-strategy)
is a SHADE-style generate/update object.

**Why later.** Parameters of that strategy, not a separate
backlog row.

---

## Active CMA and mirrored sampling

**What.** Active CMA uses unsuccessful steps. Mirroring draws
$x$ and $-x$. Flags on `Strategy`, not new classes.

**Today.** `Strategy` is full-matrix CMA with box constraints
and IPOP/BIPOP restarts. Sep-CMA is the high-dimension path.

**Why later.** Hansen-family polish after sep-CMA and restarts.

---

## Mixed-integer CMA

**What.** Mark some coordinates as integer: sample continuous,
round, box. Same `generate` / `update` surface.

**Today.** Boxed CMA and `mut_heterogeneous` /
`cx_heterogeneous` already handle mixed encodings if the caller
wires them.

**Why later.** High practical value, sibling of shipped
`Strategy`, not program search.

---

## SNES and CEM

**What.** Separable NES and the cross-entropy method as
$O(n)$ generate/update strategies next to `StrategySeparable`.

**Today.** Sep-CMA is the $O(n)$ CMA-shaped strategy.

**Why later.** Wait until a user hits a documented wall after
sep-CMA. xNES / full NES stay further out.

---

## Adaptive operator rates

**What.** Probability matching or a small MAB over
`{mate, mutate, slim_inflate, slim_deflate, tune}` using last
generation's archive or lexicase wins. Success-based `cx_prob`
/ `mut_prob` is the older version.

**Today.** `var_and` / `var_or` take fixed probabilities.

**Why later.** Policy, after
[item 37](features_31_40.md#37-evaluation-budget-and-eval-cache)
makes evaluation budget first-class.

---

## Batched var_and uniforms

**What.** Drain mate-or-skip and mutate-or-skip decisions with
`rng.take_floats`, the way flip-bit already does.

**Today.** Isolated `sel_tournament` is ahead of DEAP. Tiny
`ea_simple` still loses because variation draws one NumPy
uniform per decision.

**Why later.** Housekeeping. Only worth it if the golden stream
can be preserved or is deliberately versioned.

---

## Island topologies

**What.** `mig_fully_connected` / `mig_random` next to
`mig_ring`, plus a helper that computes `eval_keys` from the
current exam.

**Today.** `step_islands(..., migrate=)` already accepts any
migrate callable. `mig_ring` is the shipped topology.

**Why later.** A one-function recipe. Archives still do not
auto-merge.

---

## Noisy fitness resample

**What.** `resample(ind, evaluate, n)` and a racing stop
(F-Race-shaped) for noisy cases.

**Today.** Lexicase and exams treat one draw as truth.
`evaluate_invalid` scores each invalid once.

**Why later.** After the eval cache so repeats are cheap.
Time-series GP is the usual noisy caller.

---

## Persistent hall of fame

**What.** A first-class `hof` slot on `Checkpoint`, or
`HallOfFame.to_json` matching `Logbook`.

**Today.** `Checkpoint` dill-dumps arbitrary state. `Logbook`
already round-trips JSON. DEAP still has the persistency
request open.

**Why later.** The mechanism exists. A convenience API, not a
new record type.

---

## WFG and constrained DTLZ

**What.** WFG (irregular fronts) and constrained DTLZ as
`bm_*` functions.

**Today.** ZDT and DTLZ 1–7 are shipped. AGE-MOEA-II / RVEA
papers use WFG.

**Why later.** Test problems for
[item 32](features_31_40.md#32-constraint-dominance-on-remaining-selectors)
and [item 34](features_31_40.md#34-rvea-and-r-nsga-ii).
Not library surface.

---

## step_program_search

**What.** One generation that wires evaluate → exams →
lexicase → Slim → `tune_ephemerals` → semantic archive →
`sel_team`. Explicit callables; the library still does not own
fitness.

**Today.** Every piece exists as a toolbox function.
`ea_simple` / `step_islands` / `ea_map_elites` are the thin
loops.

**Why later.** A wrapper that wants to become a framework.
Mention it; do not ship it until someone hits a documented
composition bug that a helper would have prevented.
