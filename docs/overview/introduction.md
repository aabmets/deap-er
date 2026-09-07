# Introduction

[DEAP-ER](https://github.com/aabmets/deap-er) is a rewrite of
[DEAP](https://github.com/DEAP/deap) for Python 3.12 and newer. The
toolbox model is the same — register operators, run an algorithm —
and so are the families of methods listed below. The package is typed,
uses snake_case, and the published API is what this documentation
describes.

It is not a drop-in rename. Function names, parameter order, and a few
contracts changed. The [differences page](differences.md) is the
migration note and the inventory of work that accumulated on top of
the original toolbox. Timed hot paths versus DEAP are on the
[performance page](performance.md). Planned library work is on the
[roadmap](roadmap.md).

## Capabilities

- Genetic algorithms on ordinary Python containers (list, array, set,
  dict, tree, NumPy array, and similar)
- Genetic programming on prefix trees: loosely typed, strongly typed,
  and automatically defined functions
- Columnar genetic programming over named `float64` columns, with a
  vectorized kit, causal windows, and optional compiled backends
- Evolution strategies (covariance matrix adaptation)
- Multi-objective search (SPEA-II, NSGA-II, NSGA-III, SMS-EMOA, MOEA/D,
  AGE-MOEA-II, MO-CMA)
- Cooperative and competitive co-evolution
- Parallel evaluation with multiprocessing or with
  [Ray](https://github.com/ray-project/ray)
- Statistics, hall of fame, and a NetworkX-compatible genealogy
- Checkpoints that persist a run to disk
- Benchmarks against common test functions
- Worked examples of symbolic regression, particle swarm, differential
  evolution, and estimation of distribution
