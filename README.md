# DEAP-er

DEAP-er is a fully refactored rework of the original DEAP library.
It's available for Python version 3.9 and the availability for 
version 3.10 depends on the Ray dependency library.


## Description

DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. 
It seeks to make algorithms explicit and data structures transparent. 
It works in perfect harmony with parallelization mechanisms such as 
multiprocessing and [SCOOP](https://github.com/soravux/scoop).

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

See the [Documentation](http://deap.readthedocs.org/) for the DEAP user guide, tutorials and the reference manual.


## Installation
```bash
pip install deap-er
```


## Importing
Change the import statement from `import deap` to `import deap_er`.
Everything should just work.


## Contributing

Please read the CONTRIBUTING.md file before submitting pull requests.


## Notes

The hypervolume submodule is currently pure-python only, because I could not get its C extension to compile.  
