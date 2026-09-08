# Introduction

[DEAP-ER](https://github.com/aabmets/deap-er) is a typed
evolutionary-algorithm toolbox for Python 3.12 and newer. Register
operators, run an algorithm. The published API is what this
documentation describes: genetic algorithms and mixed encodings, tree
and columnar GP (including memetic and semantic search helpers), a
private Push policy loop, CMA (boxed, separable, and restarting),
multi-objective and quality-diversity search, and case-structured
selection.

The library started as a rewrite of [DEAP](https://github.com/DEAP/deap).
The toolbox model is the same; it is not a drop-in rename. Function
names, parameter order, and a few contracts changed. The
[differences page](differences/index.md) is the migration note and the
inventory of work that accumulated on top of the original toolbox.
A first complete program is on
[Getting started](getting_started.md#first-program).
Timed hot paths versus DEAP are on the
[performance page](performance.md). Planned library work is on the
[roadmap](roadmap/index.md).

## Capabilities

- Genetic algorithms on ordinary Python containers (list, array, set,
  dict, tree, NumPy array, and similar), including mixed encodings
  and differential-evolution variation
- Genetic programming: prefix trees (loosely typed, strongly typed,
  ADFs), SlimGP, columnar programs over named `float64` columns
  with vectorized, causal-window, and tape / Numba backends, and a
  private [Push GP](../tutorials/push_gp.md) policy loop that evolves
  the search around those tapes
- Program-search helpers: growing language, semantic descriptors,
  memetic ephemeral tuning, and Keijzer affine scaling
- Evolution strategies: CMA, boxed CMA, separable CMA, IPOP/BIPOP
  restarts, and MO-CMA
- Multi-objective search (SPEA-II, NSGA-II, NSGA-III, SMS-EMOA, MOEA/D,
  AGE-MOEA-II) with optional constraint-dominance on NSGA-II
- Quality-diversity search (MAP-Elites: grid, CVT, and unstructured
  archives)
- Case-structured selection (lexicase, ε-lexicase, informed
  down-sampling, program teams, co-evolving case exams)
- Cooperative and competitive co-evolution, plus heterogeneous island
  stepping
- Parallel evaluation with multiprocessing or with
  [Ray](https://github.com/ray-project/ray), plus spawned RNG streams
- Statistics, hall of fame, evaluation cache and budget, and a
  NetworkX-compatible genealogy
- Checkpoints that persist a run to disk
- Benchmarks against common test functions
- Worked examples of symbolic regression, particle swarm, differential
  evolution, MAP-Elites, mixed encoding, lexicase, columnar GP, and
  CMA (boxed, separable, restarting)
