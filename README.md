# DEAP-ER

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deap-er)](https://pypi.org/project/deap-er/)
[![GitHub License](https://img.shields.io/github/license/aabmets/deap-er)](https://github.com/aabmets/deap-er/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/aabmets/deap-er/branch/main/graph/badge.svg?token=hEELibzJvq)](https://codecov.io/gh/aabmets/deap-er)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aabmets/deap-er/pytest-codecov.yml?label=tests)](https://github.com/aabmets/deap-er/actions/workflows/pytest-codecov.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/deap-er)](https://pypistats.org/packages/deap-er)


[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)<br/>
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=bugs)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=aabmets_deap-er&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=aabmets_deap-er)

DEAP-ER is a complete rewrite of the original DEAP library for Python 3.12 and up, which includes features such as:

  * Genetic algorithms using any imaginable containers like:
    * List, Array, Set, Dictionary, Tree, Numpy Array, etc.
  * Genetic programming using prefix trees
    * Loosely typed, Strongly typed
    * Automatically defined functions
  * Evolution Strategies (Covariance Matrix Adaptation)
  * Multi-objective optimisation (SPEA-II, NSGA-II, NSGA-III, MO-CMA)
  * Co-evolution (cooperative and competitive) of multiple populations
  * Parallelization of evolution processes using multiprocessing or with [Ray](https://github.com/ray-project/ray)
  * Records to track the evolution and to collect the best individuals
  * Checkpoints to persist the progress of evolutions to disk
  * Benchmarks to test evolution algorithms against common test functions
  * Genealogy of an evolution, that is also compatible with [NetworkX](https://github.com/networkx/networkx)
  * Examples of alternative algorithms: 
    * Symbolic Regression,
    * Particle Swarm Optimization, 
    * Differential Evolution, 
    * Estimation of Distribution Algorithm


## Documentation

See the [documentation](https://aabmets.github.io/deap-er/) for the complete guide to using this library.

## Contributing

Fork the repository, add tests for new code, and open a pull request against `main`.
