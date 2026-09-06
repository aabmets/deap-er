# Computational Evolution, Genetic Programming, and deap-er

Research notes and a library-facing roadmap. Written against deap-er
3.0.0 (`main` at the time of writing) and published work through 2025–2026.
This file is **not** wired into the MkDocs nav; it is a planning artifact,
not a tutorial.

The library contract is unchanged: deap-er is a toolbox of operators and
algorithms. Evaluation, fees, portfolio state, and data loading stay on
the caller. Nothing below proposes turning the package into a backtester
or a trading product.

---

## 1. Crypto OHLC / metrics backtesting — what is actually pertinent

**Short answer:** the features that matter for evolving signals on crypto
OHLC already exist. The useful new work is not a built-in exchange
backtester. It is a richer *causal, columnar* primitive kit, faster
evaluation of many trees on long series, and selection that treats
folds / regimes / assets as fitness cases.

### 1.1 What a crypto backtest actually asks of an EA library

A typical OHLC research loop is:

1. Load bars (`open`, `high`, `low`, `close`, plus `volume` / `vwap`).
2. Evolve an *interpretable* formula or rule over those columns.
3. Turn the formula into a signal, then into fills, then into equity.
4. Score return, risk, turnover, and robustness on held-out time.

Steps 3–4 are application code. The library owns step 2, and the
plumbing that makes step 2 honest and fast. Published GP-for-trading
work agrees on that split: geometric semantic GP on lagged series
(Agapitos, Brabazon, O’Neill, EvoFIN 2014), vectorial GP on whole
windows (Azzali / Silva line; GECCO 2025 companion), multi-objective
GP + NSGA-II on return versus risk (Long et al., *Artificial Intelligence
Review*, 2025), and recent alpha-mining trees over OHLC+volume+vwap
(TreEvo, 2025). Continuous Program Search (2025 working paper) is
explicit that OHLCV evaluation, next-bar fills, fees, and walk-forward
splits live in the *evaluator*, not in the search language.

deap-er already documents the same boundary:

- Convert data **once** into C-contiguous `float64` columns
  (`docs/tutorials/columnar_gp.md`).
- Every window primitive is **causal**; warmup is `nan`, never zero.
- “Fees, portfolio state, penalties, and anything else that allocates
  Python objects belong in the fitness function, not in a kernel.”

So: **do not** add order books, fee schedules, position sizers, or
exchange connectors. Those would fight the toolbox model and the
Numba stack contract (`float64` columns only).

### 1.2 Already in the library — use these first

These are the pertinent pieces for crypto OHLC work *today*:

| Capability | Where | Why it matters on bars |
|:-----------|:------|:------------------------|
| Columnar, strongly typed GP (`Array`, `Mask`, `Window`) | `gp.make_column_pset` | Evolves a whole series at once; no Python loop over 1-minute bars. |
| Vectorized arith / compare / `vwhere` | `gp.add_numpy_primitives` | Signal = `vwhere(condition, long, flat)` without scalar `if`. |
| Causal `delay`, `diff`, `rolling_{sum,mean,std,min,max}`, `ema` | `gp.add_window_primitives` | The usual TA building blocks, without look-ahead. |
| Window ephemerals | `gp.add_window_ephemeral` | Per-tree lookbacks (e.g. 2–64). |
| Three eval backends | `compile_tree(..., backend=)` | `python` (reference), `opcode` (NumPy tape), `numba` (compiled interpreter). |
| Custom compiled kernels | `bind_numba_opcode` + `dispatch=` | RSI / ATR / decay / rank can be *user* opcodes, not a fork. |
| `evaluate_batch` | `algorithms` loop | Score a generation against one shared matrix. |
| `clone_individual` | `tools` | Avoids deepcopying immutable GP nodes. |
| NSGA-II / NSGA-III / SPEA-II / MO-CMA | `operators`, `strategies` | Return vs. risk vs. turnover as real objectives. |
| Lexicase and ε-lexicase | `sel_lexicase`, `sel_epsilon_lexicase` | Fitness cases = folds, assets, or regimes. |
| HARM | `gp.harm` | Bloat control; trading rules must stay readable. |
| Checkpoints | `Checkpoint` | Multi-hour runs on long histories. |

The columnar example (`examples/genetic_programming/columnar_gp.py`) is
already the right shape: rediscover
`rolling_mean(level, 4) - delay(flow, 2)` and ignore non-finite warmup
in fitness. Swap `level`/`flow` for `close`/`volume` and the same
program is an alpha.

### 1.3 New feature work that *is* pertinent for OHLC

Ranked by how much they help a crypto researcher without leaving the
toolbox model.

1. **Two-input causal windows** — `rolling_corr`, `rolling_cov`,
   `rolling_beta` of two `Array`s. OHLC research lives on pairs
   (close vs. volume, high vs. low, asset A vs. BTC). The current kit
   is unary. This is the single highest-leverage primitive addition.
2. **Alpha-mining unaries that stay causal** — `rank` (cross-sectional
   is out of scope unless the caller stacks assets as columns),
   time-series `ts_rank`, `ts_argmax` / `ts_argmin`, `decay` /
   `weighted_sum`, and a protected `delta / delay`. These are the
   operators WorldQuant-style GP and `gpquant` keep rediscovering.
   Implement as optional kit functions + opcodes, not as a “TA library.”
3. **O(n) window kernels** — the Numba path in
   `private/programming/numba/numba_window.py` recomputes each window
   from scratch (`O(rows × window)`). On a year of 1-minute crypto
   bars that dominates fitness. Incremental sum/mean/std and a
   deque-style min/max are the same API with a different kernel.
4. **Batch tape evaluation** — one compiled call over many tapes and
   one packed matrix. `evaluate_batch` exists; the interpreter still
   runs one tree at a time and copies `stack[0]` out. This is the
   difference between “columnar” and “columnar *and* population-wide.”
5. **Down-sampled / informed lexicase** — Helmuth/Spector down-sampling
   and Boldi et al. informed down-sampling. Crypto fitness is naturally
   many cases (walk-forward folds, per-asset, per-regime). Current
   `sel_lexicase` is correct but `O(sel_count × n × cases)` in Python
   and always uses every case.
6. **Non-bloating geometric semantic variation (SLIM / SLIMMER)** —
   current `cx_semantic` / `mut_semantic` wrap parents in linear
   combinations and grow without bound. Trading rules that cannot be
   read are not usable. Vanneschi’s SLIM (EuroGP 2024; GPEM 2026)
   keeps the unimodal semantic landscape and adds a *deflate* mutation.
7. **Quality-diversity archive (MAP-Elites-style)** — keep the best
   rule per *behavior* cell (trade frequency, holding time, volatility
   bucket). The 2025 VGP trading paper and the broader QD literature
   both treat diversity of *strategy shape* as the antidote to one
   overfit champion. This is a records/archive feature, not a broker.

### 1.4 What is *not* pertinent (or not the library’s job)

- A first-class walk-forward / purged-k-fold *algorithm*. The caller
  already owns the evaluator; documenting a recipe is enough.
- Sharpe, Sortino, max-drawdown, or Calmar as public fitness functions.
  They need fills, costs, and a return series the library does not have.
- LLM / transformer mutation (ELM, TSGP, AlphaEvolve). Heavy optional
  deps, unstable operators, and they do not compose with the tape.
- PushGP, Cartesian GP, or a second genome as a crypto prerequisite.
  Columnar prefix trees already match the published VGP-for-trading
  setup.
- Embedding any exchange, candle downloader, or paper-trading loop.

If the only question is “should we build a crypto backtester inside
deap-er?” the answer is **no**. If the question is “what library work
makes OHLC research better?” the answer is items 1–7 above, in that
order.

---

## 2. Field findings — computational evolution and GP

This section is a survey of *published* directions that a general EA
toolbox should know about. It is not a claim that deap-er should
implement all of them.

### 2.1 Semantic genetic programming

Standard GP varies syntax and hopes the semantics move. Geometric
semantic GP (GSGP; Moraglio, Krawiec, Johnson) makes crossover and
mutation *geometric* in the space of input–output vectors, which turns
the error surface of a convex loss into a cone. The historical cost is
bloat: each variation *wraps* the parent.

Recent work attacks that cost without giving up the landscape:

- **SLIM / SLIM_GSGP** (Vanneschi, EuroGP 2024; follow-ups 2025) —
  inflate mutation (classic GSM) plus a *deflate* mutation that
  shortens the genotype while remaining a ball mutation of radius
  `ms` in semantic space.
- **SLIMMER / XOBDn** (GPEM 2026) — Best-Donor geometric semantic
  crossover that keeps offspring size equal to the parents.
- **Transformer Semantic GP** (Wittenberg, Rothlauf; arXiv:2501.18479)
  — a generative transformer trained on synthetic problems proposes
  syntactically different, semantically close offspring. Competitive
  with SLIM and DSR on SRBench-style tasks. It is a *model*, not an
  operator you ship in a scientific wheel.

deap-er already has `cx_semantic` and `mut_semantic`. They are the
classical wrapping operators. They require primitives named `lf`,
`mul`, `add`, `sub`, and they grow trees. They are not SLIM.

### 2.2 Lexicase and case-based selection

Lexicase (Spector, Helmuth, and later ε-lexicase / batch / down-sampled
variants) filters parents by fitness *cases* in random order. It is the
default strong baseline for program synthesis and a strong one for
symbolic regression when the loss can be factored per point.

Recent results that matter for a toolbox:

- **ε-lexicase** — already in deap-er (`sel_epsilon_lexicase`),
  including MAD-estimated slack.
- **Down-sampled lexicase** (Helmuth & Spector) — evaluate on a
  random subset of cases per generation; more individuals per budget.
- **Informed down-sampling** (Boldi et al., 2023–2024) — build the
  subset from population statistics so synonymous cases are not
  over-sampled. Beats random down-sampling on PSB-style synthesis.
- **Plexicase** (Ding, Chen, Spector, GECCO 2023) — a tractable
  approximation of lexicase’s selection *distribution*. Faster, and
  it gives a probability you can anneal or mix.
- **Geiger et al. 2024** (extended analysis) — on 26 SR problems,
  choice of lexicase variant *and* whether you measure by evaluation
  budget or wall time changes the ranking. Informed down-sampling
  plus batches is the practical combination.

deap-er has the two classical operators and no down-sampling,
informed sampling, or plexicase. The implementation is a Python loop
over cases and list comprehensions (`sel_lexicase.py`).

### 2.3 Multi-objective and many-objective search

The field has three live families:

| Family | Representative | Status in deap-er |
|:-------|:---------------|:------------------|
| Pareto + crowding / archive | NSGA-II, SPEA-II | Present (`sel_nsga_2`, `sel_spea_2`) |
| Reference directions | NSGA-III, MOEA/D, RVEA | NSGA-III present; no MOEA/D, no RVEA |
| Indicator | SMS-EMOA, IBEA, HypE | Hypervolume *metric* via moocore; no indicator *selector* |
| Geometry-adaptive | AGE-MOEA / AGE-MOEA-II (Panichella, GECCO 2019 / 2022) | Absent |
| Quality-diversity | MAP-Elites, NSLC, Dominated Novelty Search; learned QD (LQD, 2025) | Absent |

AGE-MOEA-II estimates front curvature (Newton–Raphson) and uses
geodesic distances. pymoo and jMetal ship it; it is the current
“NSGA-III but the front is not a simplex” answer. SMS-EMOA is now
cheap to add *because* hypervolume already delegates to moocore
(closed issue: the old pure-Python HV was ~1000× slower). MOEA/D
is the standard decomposition method and is still missing.

Quality-diversity (Mouret & Clune, MAP-Elites; 2025 LQD via
meta-BBO) is the other large gap. QD is how you keep a *collection*
of competent solutions across a behavior space. For program search
and for trading-rule archives that is often more useful than a
single Pareto front.

### 2.4 Representations beyond prefix trees

Prefix trees (Koza / DEAP / deap-er) remain the research default.
Other encodings that have moved from papers into toolkits:

- **Linear GP** — register machines; natural for compiled interpreters.
  Rust crates (`lgp-core`) and many C systems. Fits deap-er’s tape
  more than the tree does.
- **Cartesian GP** — DAGs with reusable nodes; CRust_GP (2024) is a
  full Rust research kit.
- **PushGP** — stack language, autoconstruction; still the PSB
  vehicle. A Rust interpreter exists (`duo-pushr`).
- **Vectorial GP** (Azzali et al.; GECCO 2025 trading paper) —
  terminals are *vectors*, operators are vector→vector. deap-er’s
  columnar `Array` tag *is* this representation under another name.
- **Operon** (Burlacu et al., GECCO 2020; actively maintained) — C++
  linear-tree GP, thread-per-offspring, optional local search via
  dual numbers. Orders of magnitude faster than DEAP on scalar SR.
- **PySR / SymbolicRegression.jl** (Cranmer) — Julia JIT fuses
  operator combinations into SIMD kernels. The lesson is “compile
  the *expression*, not only the interpreter.”

deap-er should not grow a second genome family unless there is a
caller who cannot express the problem on trees or columnar arrays.
The tape is already a linearization of the tree.

### 2.5 Evolution strategies and continuous search

CMA-ES remains the default continuous solver. deap-er has the
standard, (1+λ), and multi-objective variants, now with box
constraints (`clip` / `resample`). Field movement since then is
mostly *around* CMA, not a replacement: Sep-CMA and LM-CMA for
high dimension, BIPOP restarts, and learned step-size controllers
(Shala et al.). Quality-diversity and neuroevolution (ELM — Evolution
Through Large Models, Lehman et al.) sit on top of MAP-Elites, not
inside CMA.

A BIPOP / IPOP restart wrapper around the existing `Strategy` would
be a small, high-value ES addition. A learned CMA controller would
not.

### 2.6 Evaluation, overfitting, and time series

The GP-for-finance literature is consistent on *failure modes*:

- Look-ahead in indicators (deap-er already forbids this).
- Training on one contiguous window until the rule memorizes a
  regime (VGP 2025: randomize the training origin each generation;
  Gonçalves & Silva 2013: split the series and hide parts).
- Bloat that looks like fitness (HARM, parsimony, SLIM).
- In-sample Sharpe that dies out of sample (MOO3: three objectives
  plus a modified Sharpe only *after* the search, as a preference
  filter — not as the sole fitness).

Walk-forward, embargoed splits, and next-bar fills are evaluator
policy. The library can help by making *case-structured* fitness
cheap (lexicase variants, batch eval) and by keeping programs
small (HARM, SLIM).

### 2.7 Compilation and native kernels

Three production answers exist:

1. **Stay in Python, vectorize** — NumPy ufuncs, `sliding_window_view`.
   deap-er’s default and `opcode` backends.
2. **JIT the interpreter** — one compiled stack machine, many tapes.
   deap-er’s `numba` extra. Same idea as a bytecode VM.
3. **JIT or AOT each expression** — Operon (static C++), PySR
   (Julia fusion), or a Rust/C kernel per tree. Fastest for *scalar*
   SR; more engineering per primitive.

A fourth answer, used for hypervolume, is **delegate the hot metric
to a maintained C wheel** (moocore). The differences page is explicit:
“deap-er stays a pure-Python package.”

---

## 3. Current state of deap-er

### 3.1 What the library is

A single-package, typed rewrite of DEAP for Python ≥ 3.12. Public
imports are `deap_er` (`Toolbox`, `Fitness`, `creator`, `Checkpoint`)
plus `tools`, `gp`, and `typedefs`. Operators and algorithms are
module-level functions. Runtime dependencies: `numpy`, `scipy`,
`dill`, `moocore`. Optional extra: `numba`. License: Apache-2.0.
Version: 3.0.0.

Layout (actual tree, not the older skill sketch):

```
deap_er/
  algorithms.py / operators.py / strategies.py / records.py /
  benchmarks.py / tools.py / gp.py     # public barrels
  private/
    toolbox.py, fitness.py, creator.py, checkpoint.py
    algorithms/   # ea_simple, μ+λ, μ,λ, generate-update, loop
    operators/    # cx_*, mut_*, mig_ring, sel_*
    strategies/   # CMA, (1+λ)-CMA, MO-CMA
    programming/  # GP trees, tape, numba, columnar, HARM
    records/      # logbook, statistics, hall of fame, history
    various/      # clone, metrics, HV, init, constraints, RNG
    benchmarks/
```

111 library modules, 69 test modules. The published docs
(https://aabmets.github.io/deap-er/) match this surface.

### 3.2 What it already does well

Relative to upstream DEAP, and verified in
`docs/overview/differences.md` plus the source:

- Snake_case, type hints, deprecated DEAP APIs removed.
- Bounded real variation that does not write NaN (`cx_simulated_binary_bounded`,
  `mut_polynomial_bounded`, `cx_blend_bounded`).
- Allele-keyed permutation crossovers; heterogeneous per-gene mutation.
- Box-constrained CMA (`clip` / `resample`) on every strategy;
  MO-CMA covariance-update fixes.
- Hypervolume, contributions, and Pareto ranking via moocore.
- GP: weighted primitive sampling, infix printer, typed generation
  that can close a leaf-only type (`Window`), columnar kit, three
  compile backends, `evaluate_batch`, `clone_individual`, HARM
  corrections, semantic-operator parent-snapshot fix.
- Records: JSON logbooks, targeted `MultiStatistics`, per-generation
  duration / logger / Pareto-front snapshots.
- Atomic checkpoint replace; creator keeps `array.array` typecodes.

This is a production-stable *research toolbox*, not a DEAP clone
with a coat of paint.

### 3.3 Gaps versus the field (library-shaped)

Present and modern:

- Columnar / vectorial GP, causal windows, Numba tape.
- NSGA-II/III, SPEA-II, MO-CMA, lexicase, ε-lexicase, HARM.
- Parallel `map` + documented shared-memory pattern.

Absent, and commonly expected in 2025 toolkits (pymoo, jMetal,
DEAP-adjacent papers):

- MOEA/D, SMS-EMOA / IBEA, AGE-MOEA-II.
- MAP-Elites or any QD archive.
- Down-sampled / informed / probabilistic lexicase.
- Non-bloating GSGP (SLIM).
- Binary window primitives; a broader causal kit.
- Batch (population × rows) tape interpreter.
- IPOP/BIPOP CMA restarts as a first-class wrapper.
- Linear / Cartesian / Push genomes (deliberately out of scope
  unless a user appears).

The GitHub tracker is almost empty (one closed HV issue). There is
no public roadmap; this file is that draft.

---

## 4. Feature development — what to undertake, and why

Each item is scoped to the existing public surface
(`operators`, `gp`, `records`, `strategies`, `algorithms`). No new
top-level package, no service layer, no extra runtime dependency
unless listed.

### 4.1 High benefit, fits the architecture

**A. Two-input and rank-style window kit (GP)**

Add optional helpers next to `add_window_primitives`, plus opcodes
so `numba` stays honest.

- Why: unary rolling stats cannot express “close vs. volume” or
  “this bar’s rank in the last n.” That is the daily language of
  OHLC research *and* of general time-series GP.
- Benefit: new users stop forking the primitive set; the type tags
  (`Array`, `Window`) already allow a second `Array` argument.
- Risk: every new opcode must have a Python reference, an opcode
  path, and a Numba kernel, with parity tests. That is already the
  house style.

**B. Incremental window kernels (optimization that looks like a feature)**

Keep the API. Change `roll_stats` / `roll_minmax` to running
aggregates. See §5.

- Why: this is the difference between “columnar is fast” and
  “columnar is fast on a million bars.”
- Benefit: every current window user, not only crypto.

**C. Down-sampled and informed lexicase (operators)**

`sel_lexicase_downsampled(individuals, sel_count, cases=…, rng=…)`
and an informed variant that takes a case-similarity hint.

- Why: this is the 2023–2025 lexicase literature’s main result, and
  deap-er already committed to lexicase.
- Benefit: program synthesis, symbolic regression, and any
  case-rich evaluator (including walk-forward folds) get more
  individuals per budget.
- Risk: must not change `sel_lexicase` defaults.

**D. SLIM-style semantic mutation (GP)**

A deflate mutation and a size-preserving semantic crossover that
reuse the existing `lf` / `add` / `mul` / `sub` requirement.

- Why: current semantic operators are correct and unusable for
  long runs; they are the feature researchers cite and then avoid.
- Benefit: puts deap-er on the 2024–2026 GSGP map without a
  transformer or a new representation.

**E. SMS-EMOA environmental selection (operators)**

Steady-state or generational: discard the individual with smallest
hypervolume contribution. `least_contrib` and `hypervolume` already
talk to moocore.

- Why: the expensive part is done; the algorithm is a loop.
- Benefit: indicator-based MOO without a new dependency. Completes
  the “we have HV, we should use it for search” story.

**F. MAP-Elites archive (records)**

A `GridArchive` (or similar) with a behavior descriptor supplied
by the caller, `add`, and `random_elites` for variation.

- Why: QD is the largest missing *family* in the toolbox. It is
  also how you keep a diverse set of GP programs or CMA solutions
  without pretending they share one Pareto front.
- Benefit: new algorithms (`ea_map_elites`) can be thin wrappers
  around `evaluate_invalid` + archive + `var_or`.
- Risk: keep it a data structure, not a framework.

### 4.2 Medium benefit, still in-scope

**G. MOEA/D and AGE-MOEA-II selection**

MOEA/D is a weight-vector loop plus Tchebycheff / PBI; NSGA-III
already has `uniform_reference_points`. AGE-MOEA-II is more code
(front geometry + geodesics) but is what pymoo users now reach for
when NSGA-III’s simplex assumption fails.

- Why: many-objective users currently stop at NSGA-III.
- Benefit: parity with pymoo on the algorithms researchers name
  in 2025 papers, without becoming pymoo.

**H. Batch tape runner (GP)**

`interpret_tapes(tapes, matrix) -> (n_individuals, n_rows)`.

- Why: `evaluate_batch` is a hook with no matching compiled
  implementation. The hook is waiting for this.
- Benefit: one kernel launch, one matrix, no per-tree
  `_as_matrix` / `stack[0].copy()` dance.

**I. IPOP / BIPOP restart around `Strategy`**

A small algorithm next to `ea_generate_update` that grows λ and
resets the CMA state on stagnation.

- Why: this is how CMA is actually used on hard landscapes.
- Benefit: ES users stop writing the same wrapper.

**J. Case-mask / fold helper for batch evaluation (utilities, thin)**

Not a backtester. A function that, given a boolean mask or a list
of index ranges, returns per-case errors from a predicted series
so lexicase has cases.

- Why: the 2025 VGP paper’s “three segments, hide two” recipe is
  five lines if the helper exists and fifty if everyone writes it
  wrong (leaking warmup, leaking the future).
- Benefit: documents the honest time-series fitness pattern next
  to the columnar tutorial.

### 4.3 Low priority or do not do

| Idea | Verdict |
|:-----|:--------|
| PushGP / CGP / linear GP as a second public genome | Do not, unless a user cannot use trees. The tape is enough linearization. |
| Transformer / LLM mutation | Do not put in-tree. Optional recipe in docs at most. |
| Native crypto TA library (RSI, MACD, Bollinger as named primitives) | Do not. Composable windows + user opcodes. Named TA ages badly. |
| Built-in Sharpe / drawdown | Do not. No fills in the library. |
| Learned QD / meta-BBO | Research paper, not a toolbox function. |
| Switching persistence off dill | Forbidden by project rules. |
| Replacing creator with dataclasses | Forbidden by project rules. |

---

## 5. Optimization analysis (including C and Rust)

### 5.1 Where time actually goes

For a *columnar* GP run on long series, the profile is not “Python
is slow at adding two numbers.” It is:

1. **Window kernels** — Numba `roll_stats` / `roll_minmax` are
   nested loops over `(t, j)` (`numba_window.py`). Complexity
   `O(rows × window)` per opcode. A tree with several rolling
   nodes on 500k bars and windows of 64–256 spends almost all
   of its time here. The Python/opcode path uses
   `sliding_window_view` + ufunc.reduce, which is better than
   a Python loop but still reads `O(rows × window)` data.
2. **Interpreter dispatch** — the Numba VM is one compiled
   `interpret` over a tape. That is already the right design
   (compile once per process, not per tree). Remaining waste:
   packing columns on every call unless the user pre-packs a
   matrix, and copying `stack[0]` out of a *shared* workspace
   (`bind_tape`).
3. **Workspace policy** — one process-wide stack. Safe and
   memory-cheap; forbids intra-process threads. Parallelism is
   `toolbox.map` across processes. That is documented and
   correct. It does mean you cannot hide latency with threads.
4. **Selection** — `sel_lexicase` is Python over cases;
   SPEA-II density / archive truncation is Python over pairs;
   NSGA-II crowding is already delegated in part to moocore
   via `sort_non_dominated`. Lexicase will hurt first when
   cases = bars or cases = assets × folds.
5. **Clone / `static_limit`** — `static_limit` still
   `deepcopy`s parents on every variation. GP toolboxes that
   forget `clone_individual` pay full tree copies. Easy
   documentation issue; also a possible `static_limit` fix
   that uses `clone_individual` when registered.
6. **`ema` on the Python path** — imports `scipy.signal.lfilter`
   on each call site’s first use (deferred import is
   intentional) and builds filter state every invocation.
   Fine for correctness; not a hot-path design.
7. **Compile cache** — `_COMPILE_CACHE_MAX = 1024` then
   **clears the whole dict**. A long run with diverse trees
   thrashes. An LRU would keep the working set.

For *scalar* symbolic regression (the `symb_regr.py` example),
evaluation is a Python `eval` per point. That path will never
match Operon or PySR. The columnar + tape path is the intended
answer; the scalar path is for teaching and for ADF/legacy trees.

### 5.2 C or Rust — recommendation

**Do not compile the library core to C or Rust.**

Evidence from this repo, not from fashion:

- The differences page states the policy: stay a pure-Python
  package; native work is a *dependency wheel* (moocore), not
  vendored C.
- The HV story is the template. A custom HV was 1000× slower;
  the fix was `moocore`, not a deap-er C module.
- Numba already compiles the only inner loop that is both hot
  and *owned* by this project (the tape interpreter and window
  kernels).
- A Rust/PyO3 or CPython extension implies: a rustc/C toolchain
  in every release, wheels for CPython 3.12–3.14 × manylinux /
  macOS / Windows, and a second implementation of every opcode
  that must bit-match `python` and `numba`. That is a product
  decision, not a weekend optimization.
- Operon and PySR win on *scalar* SR by compiling *expressions*.
  deap-er’s differentiating workload is *columnar* GP, where
  NumPy/Numba already run C loops over the bars. The remaining
  factor is algorithm (incremental windows, batch tapes), not
  language.

**When a native extra would be justified**

Only if *all* of these become true:

1. Incremental Numba windows and a batch interpreter are in
   and profiled.
2. A production user still spends most of the wall time inside
   `interpret` dispatch (not inside their own fitness).
3. The extra is optional (`deap-er[native]`), like `numba`,
   ships prebuilt wheels, and is a *second backend* for the
   same `Tape` — not a rewrite of `PrimitiveTree`.

If that day comes, **Rust + PyO3** is a better extra than C:
one ABI, no CMake, and the tape is a pair of `int32` buffers
plus a `float64` workspace — a natural `#[pyfunction]`. Cython
is the worse of both worlds (C toolchain *and* a new language
in-tree). **Do not** rewrite selection or CMA in Rust; those
are not the bottleneck and they must stay easy to read.

**What to do instead of C/Rust (ordered)**

1. Incremental `O(rows)` rolling sum/mean/std; monotonic-queue
   min/max; keep `nan` semantics identical (parity tests against
   the current Python functions, which remain the definition).
2. Accept a pre-packed matrix in the opcode backend the way
   Numba already does; stop `numpy.stack` per individual.
3. Batch interpreter: `n_tapes × rows` output, optional fused
   allocation.
4. LRU compile cache; optional `clone_individual` inside
   `static_limit`.
5. Vectorize lexicase case filters with a `(n, cases)` error
   matrix — this is also what informed down-sampling wants.
6. Numba `cache=True` for the interpreter (`cache=False` today
   in `_build`) so workers do not re-JIT every process.
7. Only then: measure. If dispatch remains the top bar, an
   optional Rust VM that consumes `Tape` is a reasonable extra.

### 5.3 Other optimization notes (not compilation)

- `duplicate_count` is `O(n²)` (`value not in unique` on a
  list). Fine for hall-of-fame sizes; not for a 10k population
  statistic. A `set` of a hashable key, or a sort, is enough.
- SPEA-II archive truncation is the classic expensive step;
  acceptable at typical μ. Do not rewrite it in C.
- `evaluate_batch` + shared memory is already the
  multiprocessing story. A GPU backend is a *user* batch
  operator, not a library feature.

---

## 6. Activity list

A backlog, not a schedule. Grouped so a maintainer can pick a
lane. “Pertinent to OHLC” is marked.

### Lane 1 — Columnar GP (highest overlap with crypto OHLC)

1. **Two-input causal primitives** (`rolling_corr`, `rolling_cov`,
   `rolling_beta`) with Python + opcode + Numba parity.
   *OHLC: yes.*
2. **Causal ts-rank / ts-argmax / ts-argmin / decay** as a second
   optional kit (`add_ts_primitives`). *OHLC: yes.*
3. **Rewrite Numba (and, if needed, NumPy) window kernels to
   O(rows).** Preserve `nan` warmup and “`nan` in window ⇒ `nan`
   out” for min/max. *OHLC: yes (1-minute bars).*
4. **Batch tape evaluation** (`interpret_tapes` + a documented
   `evaluate_batch` recipe that compiles once per unique tree
   string). *OHLC: yes.*
5. **LRU compile cache**; stop the 1024-then-wipe policy.
6. **`static_limit` should clone via `clone_individual` when the
   individual is a sequence of immutable nodes**, or document
   that decorated GP must register it (today it always
   `deepcopy`s).
7. **Numba interpreter `cache=True`** (or an explicit cache
   directory) so spawned workers are not re-JITing the VM.
8. **Opcode backend: accept a packed `(rows, cols)` matrix.**
9. **Do not** add named RSI/MACD/Bollinger primitives. Show in
   the columnar tutorial how a user opcode implements one.

### Lane 2 — Selection and archives

10. **Down-sampled lexicase** with an explicit case index list.
    *OHLC: yes (folds / assets as cases).*
11. **Informed down-sampling** (Boldi et al.) as a case-subset
    builder used *by* lexicase, not a third ad-hoc selector.
12. **Plexicase** if (10) is not enough and profiles say
    selection dominates.
13. **SMS-EMOA** on top of existing `least_contrib` / moocore.
14. **MOEA/D** (Tchebycheff + the existing reference points).
15. **AGE-MOEA-II** environmental selection (after MOEA/D;
    more code, more payoff on irregular fronts).
16. **MAP-Elites `GridArchive`** + a thin `ea_map_elites`.
    *OHLC: useful for a diverse rule book; not required for a
    first backtest.*
17. **Vectorize lexicase** over a dense error matrix.

### Lane 3 — Semantic GP and bloat

18. **SLIM inflate/deflate mutation** next to `mut_semantic`.
    *OHLC: yes (readable rules).*
19. **Size-preserving semantic crossover** (XOBDn or a minimal
    subset). Keep the current operators for compatibility.
20. **HARM + SLIM example** so users see a non-bloating
    semantic path that is not the wrapping operators.

### Lane 4 — Evolution strategies and algorithms

21. **IPOP/BIPOP restart** wrapper around `Strategy` /
    `ea_generate_update`.
22. No new CMA variant until a user hits a documented wall
    (high-dim Sep-CMA is the only plausible next one).

### Lane 5 — Time-series evaluation hygiene (docs + tiny helpers)

23. **Columnar tutorial subsection: “fitness on bars”** —
    ignore non-finite warmup, never score a `vwhere` that hid
    a `nan`, next-bar alignment is the caller’s job.
    *OHLC: yes, and it is documentation, not a feature.*
24. **Optional `case_errors(predicted, target, ranges)`**
    utility that returns per-fold losses for lexicase.
    *OHLC: yes; still not a backtester.*
25. **Do not** implement walk-forward, fees, or Sharpe.

### Lane 6 — Performance policy (C/Rust)

26. **Profile a columnar run** (window-heavy tree, 1e5–1e6
    rows, `backend="numba"`) and check items 3, 4, 5, 7
    before writing any native code.
27. **Keep the core pure Python.** Native work stays in
    optional extras (`numba` today, maybe `native` later)
    or in upstream wheels (`moocore`).
28. **Reject an in-tree C/Cython rewrite** of operators,
    CMA, or selection.
29. **Revisit Rust only** as an optional tape backend after
    Lane 1 items 3–4–7 are done and a profile still points
    at `interpret`. Same `Tape` layout, bit-identical tests.

### Lane 7 — Housekeeping that improves the library without new science

30. **`duplicate_count` in linear time** for hashable keys.
31. A **public roadmap issue** that points at this file, so
    the tracker is not empty.
32. **Optional:** add this page to MkDocs nav under Overview
    if maintainers want it published. Not done here.

---

## 7. Sources (selected)

Library evidence: `deap_er/gp.py`, `deap_er/operators.py`,
`deap_er/algorithms.py`, `deap_er/private/programming/**`,
`deap_er/private/algorithms/loop.py`,
`docs/tutorials/columnar_gp.md`,
`docs/overview/differences.md`, `README.md`, `pyproject.toml`,
GitHub issue “hypervolume is 1000x slower than moocore” (closed).

Field (primary):

- Vanneschi, SLIM_GSGP, EuroGP 2024; SLIM / SLIMMER, *Genetic
  Programming and Evolvable Machines*, 2026.
- Wittenberg & Rothlauf, “Transformer Semantic Genetic
  Programming,” arXiv:2501.18479, 2025.
- Ding, Chen, Spector, “Probabilistic Lexicase Selection,”
  GECCO 2023.
- Geiger, Sobania, Rothlauf, lexicase vs. traditional selection
  on SR, 2024 (arXiv:2407.21632).
- Boldi et al., informed down-sampled lexicase, 2023–2024.
- Panichella, AGE-MOEA (GECCO 2019), AGE-MOEA-II (GECCO 2022).
- Lange et al., “Discovering Quality-Diversity Algorithms via
  Meta-BBO,” arXiv:2502.02190, 2025.
- Azzali / Silva line; “Evolving Financial Trading Strategies
  with Vectorial Genetic Programming,” GECCO 2025 Companion
  (arXiv:2504.05418).
- Long et al., MOO3 multi-objective GP trading, *Artificial
  Intelligence Review*, 2025.
- Agapitos, Brabazon, O’Neill, GSGP for financial trading,
  EvoFIN 2014.
- Burlacu, Kronberger, Kommenda, Operon, GECCO 2020.
- Cranmer, PySR / SymbolicRegression.jl, arXiv:2305.01582.
- Lehman et al., Evolution Through Large Models, 2023.
- Continuous Program Search, SSRN 6169788, 2025 (OHLCV
  evaluator protocol; cited as application context only).

---

## 8. How to read this file

If you came for crypto: start at §1, then Lane 1 and items 10,
18, 23–25. If you came for “are we behind pymoo?”: §2.3 and
Lane 2. If you came for “should we rewrite hot loops in Rust?”:
§5 and Lane 6 — the answer is **not yet**, and maybe not ever
in core.
