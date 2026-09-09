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
| 27 | [Batch-epsilon-lexicase and down-sampled tournament](features_21_30.md#27-batch-epsilon-lexicase-and-down-sampled-tournament) | `operators` | shipped |
| 28 | [Dynamic epsilon and downsample schedule](features_21_30.md#28-dynamic-epsilon-and-downsample-schedule) | `operators` | shipped |
| 29 | [Novelty selection and iso+line](features_21_30.md#29-novelty-selection-and-isoline) | `operators`, `records` | shipped |
| 30 | [Causal lookback and suffix rescore](features_21_30.md#30-causal-lookback-and-suffix-rescore) | `gp` | shipped |
| 31 | [Affine scaling and Lamarckian writeback](features_31_40.md#31-affine-scaling-and-lamarckian-writeback) | `gp`, `utilities` | shipped |
| 32 | [Population tape CSE](features_31_40.md#32-population-tape-cse) | `gp` | shipped |
| 33 | [Evaluation budget and eval cache](features_31_40.md#33-evaluation-budget-and-eval-cache) | `algorithms`, `utilities` | shipped |
| 34 | [Parallel RNG streams](features_31_40.md#34-parallel-rng-streams) | `rng` | shipped |
| 35 | [Island topologies](features_31_40.md#35-island-topologies) | `operators` | shipped |
| 36 | [Persistent hall of fame](features_31_40.md#36-persistent-hall-of-fame) | `records` | shipped |
| 37 | [Interval analysis on tapes](features_31_40.md#37-interval-analysis-on-tapes) | `gp` | shipped |
| 38 | [Structural meta-case regularization](features_31_40.md#38-structural-meta-case-regularization) | `operators`, `utilities` | planned |
| 39 | [Homologous and semantic crossover](features_31_40.md#39-homologous-and-semantic-crossover) | `gp` | planned |
| 40 | [Noisy fitness resample](features_31_40.md#40-noisy-fitness-resample) | `utilities` | planned |
| 41 | [Case-structured generalization path](features_41_50.md#41-case-structured-generalization-path) | `utilities` + docs | planned |
| 42 | [Memetic and affine leash](features_41_50.md#42-memetic-and-affine-leash) | `gp`, `utilities` | shipped |
| 43 | [Team and archive recipe](features_41_50.md#43-team-and-archive-recipe) | `operators`, `records` | planned |

Shipping an item updates the matching feature page and tutorial or
reference stub. Items 15–20 are the toolbox-shaped holes after
the first backlog; they are shipped. Items 21–26 compose pieces
that already shipped (tapes, SlimGP, lexicase, archives, CMA)
into a longer program-search loop, and they are shipped.
Item 30 is the legal dirty-suffix path item 25 deferred.
Item 31 (Keijzer affine scaling next to `tune_ephemerals`) is
shipped. Items 27–36 are shipped. Items 37–40 promote
interval analysis, structural meta-cases, semantic crossover,
and noisy resample off
[Under consideration](under_consideration.md). Items 41–43 are
the thin generalization, memetic-leash, and team-archive
recipes that page left on the caller. Constraint-dominance
on the remaining selectors, archive-improving CMA, RVEA /
R-NSGA-II, and adaptive DE stay under consideration.
Write-ups for the numbered items are on
[Features 1-10](features_1_10.md),
[Features 11-20](features_11_20.md),
[Features 21-30](features_21_30.md),
[Features 31-40](features_31_40.md), and
[Features 41-50](features_41_50.md). Feature pages are
full decades even when the last decade is not full. Still not a second
genome family. A separate [Push GP](push_gp.md) page lists the planned
*policy* track — observation schema through a private Push individual.
That table is not part of items 1–43.

!!! note
    deap-er stays a pure-Python package. Native work remains an
    optional extra (`numba` today) or an upstream wheel
    ([moocore](https://pypi.org/project/moocore/)). An in-tree C,
    Cython, or Rust rewrite of operators, CMA, or selection is not
    on this list.

