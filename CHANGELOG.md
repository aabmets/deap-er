# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `gp.register_gp`, `gp.columnar_pset`, and `gp.evaluate_columnar`
  for the standard tree-GP / columnar toolbox wiring
- `tools.ea_policy` — `ea_simple` plus one policy observe / decide /
  apply step per generation. Policy-action evaluations count toward
  `n_evals` and the generation `nevals`; a policy step that meets
  the budget is recorded without variation
- `structural_meta_case_columns` and `structural_meta_case_weights`
  for cheap lexicase bloat and always-on regularization via appended
  meta-cases ([#116](https://github.com/aabmets/deap-er/pull/116))
- `resample`, `noisy_draw_key`, and `race_stop` for averaging noisy
  fitness draws through `EvalCache` and F-Race-shaped elimination
  ([#115](https://github.com/aabmets/deap-er/pull/115))

## [3.0.0] - 2026-09-08

Major release since GitHub tag `2.0`. This is not a drop-in from DEAP
or from 2.0.0: names, parameter order, and several contracts changed.
Roadmap features landed as PRs [#3](https://github.com/aabmets/deap-er/pull/3)–[#103](https://github.com/aabmets/deap-er/pull/103).
Correctness work landed as commits and later bughunt PRs
([#14](https://github.com/aabmets/deap-er/pull/14)–[#79](https://github.com/aabmets/deap-er/pull/79)).
The evolutionary algorithms DEAP already shipped are still DEAP's;
the items below are the 3.0 rework, new surfaces, and fixes.

### Removed

- `sort_log_non_dominated` and the `HyperVolume` class
- `ffo` on `sort_non_dominated`, `sorting` on `sel_nsga_2` /
  `sel_nsga_3`, `map_func` on `least_contrib`, and `mp_pool` on
  `StrategyMultiObjective`
- Deprecated and obsolete DEAP APIs
- Poetry, Sphinx / Read the Docs, Black, Flake8, isort, and Mypy
  (replaced below)

### Changed

- Requires Python 3.12 or newer; license is Apache-2.0
- Packaging and lockfile moved to uv; lint and types are ruff and ty
- Docs moved to MkDocs Material on GitHub Pages
- Public imports flattened (`base`, `creator`, `tools`, `gp`);
  implementations live under `deap_er.private`
- Algorithms, strategies, and benchmarks are imported from `tools`
- Hypervolume, hypervolume contributions, and Pareto ranking delegate
  to [moocore](https://pypi.org/project/moocore/)
- Process-wide RNG is one seedable NumPy facade (`rng.take_floats`
  for batched uniforms)
- Runtime dependencies are NumPy 2, SciPy, dill, and moocore;
  Numba is an optional extra
- Hot paths were sped up where DEAP still walks Python loops
  (NSGA metrics, NSGA-II, cached `compile_tree`, SPEA-II, clone,
  tournament)

### Added

Genetic programming (columnar and case-structured search):

- Columnar primitive set with NumPy, causal window, pair-window, and
  time-series kits; `interpret_tapes` batch scoring
  ([#3](https://github.com/aabmets/deap-er/pull/3),
  [#100](https://github.com/aabmets/deap-er/pull/100))
- Opcode / Numba tape backends, compile LRU, and `static_limit` clone
  ([#10](https://github.com/aabmets/deap-er/pull/10))
- SLIM non-bloating semantic variation
  ([#11](https://github.com/aabmets/deap-er/pull/11))
- Semantic descriptors, neighbors, and `SemanticSurrogate`
  ([#42](https://github.com/aabmets/deap-er/pull/42))
- Growing typed language via `promote_subtree`
  ([#43](https://github.com/aabmets/deap-er/pull/43))
- `tune_ephemerals` (boxed CMA on numeric leaves)
  ([#41](https://github.com/aabmets/deap-er/pull/41))
- Keijzer affine scaling and Lamarckian writeback
  ([#86](https://github.com/aabmets/deap-er/pull/86))
- Causal lookback and suffix rescore
  ([#85](https://github.com/aabmets/deap-er/pull/85))
- Private Push GP policy loop (observation schema, action applicator,
  held-out fitness, action guards)
  ([#89](https://github.com/aabmets/deap-er/pull/89)–[#95](https://github.com/aabmets/deap-er/pull/95))

Selection, teams, and quality-diversity:

- `case_errors`, vectorized lexicase, down-sampled / informed /
  epsilon-lexicase, and down-sampled tournament
  ([#4](https://github.com/aabmets/deap-er/pull/4),
  [#9](https://github.com/aabmets/deap-er/pull/9),
  [#99](https://github.com/aabmets/deap-er/pull/99),
  [#101](https://github.com/aabmets/deap-er/pull/101))
- `sel_team` and co-evolving case exams
  ([#39](https://github.com/aabmets/deap-er/pull/39),
  [#44](https://github.com/aabmets/deap-er/pull/44))
- `sel_sms_emoa`, MOEA/D, and AGE-MOEA-II
  ([#6](https://github.com/aabmets/deap-er/pull/6),
  [#7](https://github.com/aabmets/deap-er/pull/7))
- Deb constraint-dominance on `sel_nsga_2`
  ([#35](https://github.com/aabmets/deap-er/pull/35))
- `GridArchive`, `CvtArchive`, `UnstructuredArchive`, and
  `ea_map_elites`
  ([#8](https://github.com/aabmets/deap-er/pull/8),
  [#36](https://github.com/aabmets/deap-er/pull/36))
- `sel_novelty` and `mut_iso_line`
  ([#96](https://github.com/aabmets/deap-er/pull/96))

CMA, islands, and toolbox operators:

- Optional box bounds on every CMA strategy; `StrategySeparable`;
  IPOP / BIPOP restarts (`RestartStrategy`,
  `ea_generate_update_restarts`)
  ([#12](https://github.com/aabmets/deap-er/pull/12),
  [#32](https://github.com/aabmets/deap-er/pull/32))
- `cx_heterogeneous`, `mut_gaussian_bounded`, `mut_de`, and
  per-gene heterogeneous mutation
  ([#31](https://github.com/aabmets/deap-er/pull/31),
  [#33](https://github.com/aabmets/deap-er/pull/33),
  [#34](https://github.com/aabmets/deap-er/pull/34))
- `step_islands`, `mig_fully_connected`, `mig_random`, and
  `island_eval_keys`
  ([#40](https://github.com/aabmets/deap-er/pull/40),
  [#98](https://github.com/aabmets/deap-er/pull/98))
- `evaluate_batch`, evaluation budget / `EvalCache`, spawned RNG
  streams, persistent hall of fame (`hof_ind_cls=`), logbook JSON,
  linear-time `duplicate_count`, and `clone_individual`
  ([#5](https://github.com/aabmets/deap-er/pull/5),
  [#84](https://github.com/aabmets/deap-er/pull/84),
  [#87](https://github.com/aabmets/deap-er/pull/87),
  [#97](https://github.com/aabmets/deap-er/pull/97))

### Fixed

- 18 still-open [DEAP issues](https://github.com/DEAP/deap/issues)
  implemented here (not as upstream patches): bounded SBX / blend /
  polynomial mutation, allele-based PMX, CMA box bounds, weighted GP
  primitives, logbook JSON, DCD tournament `k`, and others listed in
  [deap_fixes](https://aabmets.github.io/deap-er/bugfixes/deap_fixes/)
- 84 correctness bugs on operators, GP, CMA, records, checkpoints,
  benchmarks, creator, utilities, and algorithms. Inventory:
  [differences](https://aabmets.github.io/deap-er/overview/differences/).
  Later clusters include `from_string` leftover tokens and Window
  ints ([#58](https://github.com/aabmets/deap-er/pull/58),
  [#71](https://github.com/aabmets/deap-er/pull/71)), `mig_ring`
  aliasing and unequal demes
  ([#55](https://github.com/aabmets/deap-er/pull/55),
  [#70](https://github.com/aabmets/deap-er/pull/70)), CMA λ=1 /
  restarts / unevaluated parents
  ([#49](https://github.com/aabmets/deap-er/pull/49),
  [#54](https://github.com/aabmets/deap-er/pull/54),
  [#69](https://github.com/aabmets/deap-er/pull/69)), Logbook
  chapter alignment
  ([#17](https://github.com/aabmets/deap-er/pull/17),
  [#23](https://github.com/aabmets/deap-er/pull/23),
  [#67](https://github.com/aabmets/deap-er/pull/67)), and empty or
  invalid fitness in HoF / Pareto / archives
  ([#21](https://github.com/aabmets/deap-er/pull/21),
  [#25](https://github.com/aabmets/deap-er/pull/25),
  [#46](https://github.com/aabmets/deap-er/pull/46))

## [2.0.0] - 2022-08-20

First GitHub release of deap-er (tag `2.0`). The library is a rewrite
of [DEAP](https://github.com/DEAP/deap): the evolutionary algorithms
are DEAP's. This cut is the reworked package, not a new algorithm suite.
The GitHub release had no notes; the items below are from the rewrite
commits that landed on that tag.

### Changed

- Relocated the package from `deap` to `deap_er` and split operators,
  algorithms, GP, strategies, and utilities into separate modules
- Added type hints, Google-style docstrings, and renamed public
  functions and datatypes
- Refactored checkpoints into a controller and changed range-function
  behavior
- Tightened the built-in `ea_simple`, `ea_mu_plus_lambda`,
  `ea_mu_comma_lambda`, and `ea_generate_update` loops
- Dropped the Ray runtime dependency
- Limited supported Python to 3.9 and 3.10

### Added

- NumPy-compatible crossover operators
- Examples and Sphinx / Read the Docs tutorials ported onto the new API

### Fixed

- Fitness comparison
- NSGA-III selection
- Hypervolume and least-contributed indicator
- CMA strategy keyword arguments

[Unreleased]: https://github.com/aabmets/deap-er/compare/3.0.0...HEAD
[3.0.0]: https://github.com/aabmets/deap-er/compare/2.0...3.0.0
[2.0.0]: https://github.com/aabmets/deap-er/releases/tag/2.0
