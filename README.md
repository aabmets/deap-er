# DEAP-ER

DEAP-ER is a complete rewrite of the original DEAP library for Python 3.10 and up, which includes features such as:

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

See the [Documentation](http://deap-er.readthedocs.org/) for the complete guide to using this library.


## Contributing

Please read the CONTRIBUTING.md file before submitting pull requests.
