# DEAP-ER

DEAP-ER is a complete rework and refactor of the original DEAP evolutionary 
computation framework library for Python 3.9, 3.10 and up.
It seeks to make algorithms explicit and data structures transparent. 
It works in perfect harmony with parallelization mechanisms such as 
multiprocessing and [Ray](https://github.com/ray-project/ray).

DEAP includes the following features:

  * Genetic algorithm using any imaginable representation
    * List, Array, Set, Dictionary, Tree, Numpy Array, etc.
  * Genetic programming using prefix trees
    * Loosely typed, Strongly typed
    * Automatically defined functions
  * Evolution strategies (including CMA-ES)
  * Multi-objective optimisation (NSGA-II, NSGA-III, SPEA2, MO-CMA-ES)
  * Co-evolution (cooperative and competitive) of multiple populations
  * Parallelization of the evaluations (and more)
  * Hall of Fame of the best individuals that lived in the population
  * Checkpoints that take snapshots of a system regularly
  * Benchmarks module containing most common test functions
  * Genealogy of an evolution, that is also compatible with [NetworkX](https://github.com/networkx/networkx)
  * Examples of alternative algorithms : Particle Swarm Optimization, Differential Evolution, Estimation of Distribution Algorithm

## Documentation

See the [Documentation](http://deap-er.readthedocs.org/) for the user guide, tutorials and the reference manual.


## Installation
```bash
pip install deap-er
```


## Importing
```python
from deap_er import base, creator, tools, env, gp
```


## Contributing

Please read the CONTRIBUTING.md file before submitting pull requests.
