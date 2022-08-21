Introduction
============

`DEAP-ER <https://github.com/aabmets/deap-er>`_ is a complete rework of the original
`DEAP <https://github.com/DEAP/deap>`_ library, which includes features such as:

* Genetic algorithms using any imaginable containers like:

   * List, Array, Set, Dictionary, Tree, Numpy Array, etc.

* Genetic programming using prefix trees

   * Loosely typed, Strongly typed
   * Automatically defined functions

* Evolution Strategies (Covariance Matrix Adaptation)
* Multi-objective optimisation (SPEA-II, NSGA-II, NSGA-III, MO-CMA)
* Co-evolution (cooperative and competitive) of multiple populations
* Parallelization of evolution processes using multiprocessing or with `Ray <https://github.com/ray-project/ray>`_
* Records to track the evolution and to collect the best individuals
* Checkpoints to persist the progress of evolutions to disk
* Benchmarks to test evolution algorithms against common test functions
* Genealogy of an evolution, that is also compatible with `NetworkX <https://github.com/networkx/networkx>`_
* Examples of alternative algorithms:

   * Symbolic Regression,
   * Particle Swarm Optimization,
   * Differential Evolution,
   * Estimation of Distribution Algorithm
