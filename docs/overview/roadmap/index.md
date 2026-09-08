# Overview

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
| 1 | [Two-input causal windows](features_1_10.md#1-two-input-causal-windows) | `gp` | shipped |
| 2 | [Causal time-series unaries](features_1_10.md#2-causal-time-series-unaries) | `gp` | shipped |
| 3 | [Incremental window kernels](features_1_10.md#3-incremental-window-kernels) | `gp` (same API) | shipped |
| 4 | [Batch tape evaluation](features_1_10.md#4-batch-tape-evaluation) | `gp` | shipped |
| 5 | [Down-sampled and informed lexicase](features_1_10.md#5-down-sampled-and-informed-lexicase) | `operators` | shipped |
| 6 | [Case-structured evaluation helper](features_1_10.md#6-case-structured-evaluation-helper) | utilities + docs | shipped |
| 7 | [Non-bloating semantic variation](features_1_10.md#7-non-bloating-semantic-variation) | `gp` | shipped |
| 8 | [Quality-diversity archive](features_1_10.md#8-quality-diversity-archive) | `records` | shipped |
| 9 | [Compile and clone path](features_1_10.md#9-compile-and-clone-path) | `gp`, `tools` | shipped |
| 10 | [Vectorized lexicase and plexicase](features_1_10.md#10-vectorized-lexicase-and-plexicase) | `operators` | shipped |
| 11 | [SMS-EMOA](features_11_20.md#11-sms-emoa) | `operators` | shipped |
| 12 | [MOEA/D and AGE-MOEA-II](features_11_20.md#12-moead-and-age-moea-ii) | `operators` | shipped |
| 13 | [IPOP / BIPOP CMA restarts](features_11_20.md#13-ipop-bipop-cma-restarts) | `algorithms`, `strategies` | shipped |
| 14 | [Linear-time duplicate count](features_11_20.md#14-linear-time-duplicate-count) | `tools` | shipped |
| 15 | [Heterogeneous crossover](features_11_20.md#15-heterogeneous-crossover) | `operators` | shipped |
| 16 | [Bounded Gaussian mutation](features_11_20.md#16-bounded-gaussian-mutation) | `operators` | shipped |
| 17 | [Differential evolution operators](features_11_20.md#17-differential-evolution-operators) | `operators` | shipped |
| 18 | [Constraint-dominance selection](features_11_20.md#18-constraint-dominance-selection) | `operators` | shipped |
| 19 | [Sep-CMA](features_11_20.md#19-sep-cma) | `strategies` | shipped |
| 20 | [CVT / unstructured MAP-Elites](features_11_20.md#20-cvt-unstructured-map-elites) | `records` | shipped |
| 21 | [Growing primitive language](features_21_30.md#21-growing-primitive-language) | `gp` | shipped |
| 22 | [Semantic search space](features_21_30.md#22-semantic-search-space) | `gp`, `records` | shipped |
| 23 | [Co-evolving cases](features_21_30.md#23-co-evolving-cases) | `operators`, `records` | shipped |
| 24 | [Memetic constants](features_21_30.md#24-memetic-constants) | `gp`, `strategies` | shipped |
| 25 | [Streaming and island ecology](features_21_30.md#25-streaming-and-island-ecology) | `algorithms` | shipped |
| 26 | [Program teams](features_21_30.md#26-program-teams) | `operators` | shipped |
| 27 | [Batch-epsilon-lexicase and down-sampled tournament](features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament) | `operators` | planned |
| 28 | [Dynamic epsilon and downsample schedule](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule) | `operators` | planned |
| 29 | [Novelty selection and iso+line](features_21_30.md#29-novelty-selection-and-isoline) | `operators`, `records` | planned |
| 30 | [Causal lookback and suffix rescore](features_21_30.md#30-causal-lookback-and-suffix-rescore) | `gp` | planned |
| 31 | [Affine scaling and Lamarckian writeback](features_31_40.md#31-affine-scaling-and-lamarckian-writeback) | `gp`, `utilities` | planned |
| 32 | [Constraint-dominance on remaining selectors](features_31_40.md#32-constraint-dominance-on-remaining-selectors) | `operators` | planned |
| 33 | [Archive-improving CMA](features_31_40.md#33-archive-improving-cma) | `strategies` | planned |
| 34 | [RVEA and R-NSGA-II](features_31_40.md#34-rvea-and-r-nsga-ii) | `operators` | planned |
| 35 | [Adaptive DE strategy](features_31_40.md#35-adaptive-de-strategy) | `strategies` | planned |
| 36 | [Population tape CSE](features_31_40.md#36-population-tape-cse) | `gp` | planned |
| 37 | [Evaluation budget and eval cache](features_31_40.md#37-evaluation-budget-and-eval-cache) | `algorithms`, `utilities` | planned |
| 38 | [Parallel RNG streams](features_31_40.md#38-parallel-rng-streams) | `rng` | planned |

Shipping an item updates the matching feature page and tutorial or
reference stub. Items 15–20 are the toolbox-shaped holes after
the first backlog; they are shipped. Items 21–26 compose pieces
that already shipped (tapes, SlimGP, lexicase, archives, CMA)
into a longer program-search loop, and they are shipped.
Items 27–38 are the next backlog: case-selection schedules,
archive search, causal streaming, memetic symbolic regression,
and the remaining toolbox holes. Ideas that compose the same
pieces but wait on a profile, a sequel, or a recipe stay in
[Under consideration](under_consideration.md). Write-ups for the
numbered items are on [Features 1-10](features_1_10.md),
[Features 11-20](features_11_20.md),
[Features 21-30](features_21_30.md), and
[Features 31-40](features_31_40.md). Feature pages are
full decades even when the last decade is not full. Still not a second
genome family. A separate [Push GP](push_gp.md) page lists the planned
*policy* track — observation schema through a private Push individual.
That table is not part of items 1–38.

!!! note
    deap-er stays a pure-Python package. Native work remains an
    optional extra (`numba` today) or an upstream wheel
    ([moocore](https://pypi.org/project/moocore/)). An in-tree C,
    Cython, or Rust rewrite of operators, CMA, or selection is not
    on this list.

