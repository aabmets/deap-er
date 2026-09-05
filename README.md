# DEAP-ER

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deap-er)](https://pypi.org/project/deap-er/)
[![GitHub License](https://img.shields.io/github/license/aabmets/deap-er)](https://github.com/aabmets/deap-er/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/aabmets/deap-er/branch/main/graph/badge.svg?token=hEELibzJvq)](https://codecov.io/gh/aabmets/deap-er)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aabmets/deap-er/pytest-codecov.yml?label=tests)](https://github.com/aabmets/deap-er/actions/workflows/pytest-codecov.yml)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)<br/>
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=bugs)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)

DEAP-ER is a rewrite of [DEAP](https://github.com/DEAP/deap) for Python 3.12
and newer. The toolbox model is the same — register operators, run an
algorithm — and so are the families of methods listed below. The package is
typed, uses snake_case, and the published API is what the documentation
describes.

It is not a drop-in rename. Function names, parameter order, and a few
contracts changed. The
[differences page](https://aabmets.github.io/deap-er/overview/differences/)
is the migration note.

```bash
pip install deap-er
```
```bash
uv add deap-er
```

## Capabilities

- Genetic algorithms on ordinary Python containers (list, array, set,
  dict, tree, NumPy array, and similar)
- Genetic programming on prefix trees: loosely typed, strongly typed, and
  automatically defined functions
- Evolution strategies (covariance matrix adaptation)
- Multi-objective search (SPEA-II, NSGA-II, NSGA-III, MO-CMA)
- Cooperative and competitive co-evolution
- Parallel evaluation with multiprocessing or
  [Ray](https://github.com/ray-project/ray)
- Statistics, hall of fame, and a NetworkX-compatible genealogy
- Checkpoints that persist a run to disk
- Benchmarks against common test functions
- Worked examples of symbolic regression, particle swarm, differential
  evolution, and estimation of distribution

## Relative to DEAP

The original library is the research toolbox this one started from. What
follows is the work that accumulated on top of that base.

The rewrite is a single package with type hints on the public surface,
deprecated DEAP APIs removed, and algorithms, strategies, and benchmarks
collected under `tools`. Hypervolume, hypervolume contributions, and Pareto
ranking delegate to [moocore](https://pypi.org/project/moocore/). State
is persisted with `Checkpoint`. The license is Apache-2.0.

Several operators and bookkeeping tools that sat as open DEAP issues —
some for more than a decade — are implemented here. Bounded real-coded
variation no longer writes NaN or raises when a gene leaves the box;
permutation crossovers work on arbitrary alleles; CMA-ES can respect a
box; mixed encodings have a per-gene mutator; GP generation can weight
primitives and print trees in infix. Logbooks round-trip to JSON, and
the stock algorithms can record wall time, emit through `logging`, and
keep the Pareto front of each generation.

A separate pass went through operators, GP, records, CMA, checkpoints, and
the published benchmarks and corrected defects that change results: NumPy
crossovers that aliased a parent, a hall of fame that skipped a strictly
better individual, MO-CMA covariance updates gated on the sign of the
evolution path, checkpoints that truncated the good file before the new
dump finished, and several GP and benchmark formula errors inherited from
the original sources.

Genetic programming also has a columnar path that DEAP does not. A typed
primitive set is built from named `float64` columns; a vectorized NumPy
kit and causal window primitives (delay, difference, rolling statistics,
EMA) produce a whole array per tree. `compile_tree` caches the default
`eval` path, can lower a tree to an opcode tape, and, with the optional
`numba` extra, runs that tape in a process-wide compiled interpreter.
Algorithms call `evaluate_batch` when it is registered, so a generation
can be scored against one shared matrix, and `clone_individual` avoids
deepcopying every node. The contract is documented in the
[columnar GP tutorial](https://aabmets.github.io/deap-er/tutorials/columnar_gp/).

## Documentation

See the [documentation](https://aabmets.github.io/deap-er/) for the
complete guide.

## Acknowledgments

DEAP-ER is a rewrite of [DEAP](https://github.com/DEAP/deap),
originally developed at the Computer Vision and Systems Laboratory (CVSL)
at Université Laval.

The original DEAP authors and main contributors, in alphabetical order:

- François-Michel De Rainville
- Félix-Antoine Fortin
- Christian Gagné
- Olivier Gagnon
- Marc-André Gardner
- Simon Grenier
- Yannick Hold-Geoffroy
- Marc Parizeau

If you use this library in a scientific paper, please also cite the original
DEAP paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{DEAP_JMLR2012,
  author  = {F{\'e}lix-Antoine Fortin and Fran{\c{c}}ois-Michel {De Rainville}
             and Marc-Andr{\'e} Gardner and Marc Parizeau and Christian Gagn{\'e}},
  title   = {{DEAP}: Evolutionary Algorithms Made Easy},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  year    = {2012},
  pages   = {2171--2175}
}
```
