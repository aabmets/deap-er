# Getting Started

## Installation

This library can be installed with:

```text
pip install deap-er
```

or if you're using the [uv](https://docs.astral.sh/uv/) package manager:

```text
uv add deap-er
```

The optional Numba compile backend for columnar GP is an extra:

```text
pip install deap-er[numba]
```

```text
uv add deap-er --extra numba
```

## Namespaces

The functionality of this library is divided into the following namespaces:

- **deap_er** — `Toolbox`, `Fitness`, `Checkpoint`, and the `creator` module.
- **tools** — algorithms (`ea_*`, MAP-Elites, islands, optional
  `n_evals=`), operators (lexicase, SMS-EMOA, MOEA/D, AGE-MOEA-II,
  mixed-gene variation, DE, constraint-dominance), CMA strategies
  (boxed, separable, restarting, MO-CMA), records (logbook, hall of
  fame, MAP-Elites archives), utilities (`EvalCache`, `spawn_rng`,
  `affine_scale`, `case_errors`, semantic helpers), and benchmarks.
- **gp** — prefix-tree GP (loosely typed, strongly typed, ADFs),
  SlimGP, growing language, memetic / affine writeback, and
  columnar kits with opcode / Numba backends.

These namespaces can be imported with:

```python
from deap_er import Checkpoint, Fitness, Toolbox, creator, gp, tools
```

## First program

Register operators, build a population, run `ea_simple`. This OneMax
run maximizes the number of ones in a 20-bit string:

```python
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)

creator.create_type("FitnessMax", Fitness, weights=(1.0,))
creator.create_type("Individual", list, fitness=creator.FitnessMax)

toolbox = Toolbox()
toolbox.register("attr_bool", tools.rng.randint, 0, 1)
toolbox.register(
    "individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, 20
)
toolbox.register("population", tools.init_repeat, list, toolbox.individual)
toolbox.register("mate", tools.cx_two_point)
toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
toolbox.register("select", tools.sel_tournament, contestants=3)
toolbox.register("evaluate", lambda ind: (sum(ind),))

pop = toolbox.population(size=60)
hof = tools.HallOfFame(1)
tools.ea_simple(
    toolbox, pop, generations=20, cx_prob=0.5, mut_prob=0.2, hof=hof, verbose=True
)
print(hof[0], hof[0].fitness.values)
```

A longer walkthrough of the same problem is the
[One Max example](../examples/genetic_algorithms/onemax.md).

## Published benchmarks

The [tools](../reference/benchmarks.md) barrel ships common test
functions as `bm_*` callables. Register one as `evaluate` instead of
writing a fitness function:

```python
toolbox.register("evaluate", tools.bm_sphere)      # unimodal
toolbox.register("evaluate", tools.bm_rastrigin)   # multimodal
toolbox.register("evaluate", tools.bm_zdt_1)       # two-objective
```

`bm_sphere` and `bm_rastrigin` return a one-float tuple. `bm_zdt_1`
returns two objectives — pair it with a multi-objective selector.

## Where next

Tutorials, in a useful reading order:

- [Using the Toolbox](../tutorials/using_the_toolbox.md)
- [Creating Individuals](../tutorials/creating_individuals.md)
- [Operators and Algorithms](../tutorials/operators_and_algorithms.md)
- [Genetic Programming](../tutorials/genetic_programming.md)
- [Columnar Programs](../tutorials/columnar_gp.md)
- [Push GP](../tutorials/push_gp.md)
- [Constraint Handling](../tutorials/constraints.md)
- [Island Models](../tutorials/islands.md)
- [Logging Statistics](../tutorials/logging_statistics.md)
- [Multiprocessing](../tutorials/multiprocessing.md)
- [Using Checkpoints](../tutorials/using_checkpoints.md)
