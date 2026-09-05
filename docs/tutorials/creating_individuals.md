# Creating Individuals

This tutorial introduces the fundamental concepts of the **fitness**, **individual**
and **population** objects and how they relate with each other. We will explore the
usage of the [`creator`](../reference/creator.md) module together with
[`Fitness`](../reference/base.md) and [`Toolbox`](../reference/base.md),
which are used to create and initialize these objects.

## Overview

### Fitness

Fitness represents the suitability of the solution values of an individual for the
given problem. Individuals with higher fitness have solutions which represent better
solutions to the problem.

[`Fitness`](../reference/base.md) types are created with the creator module and
require their *weights* attribute to be set in order to be functional. The *weights*
attribute is a sequence of either positive or negative real numbers. Negative
*weights* indicate minimizing fitness objectives while positive *weights* indicate
maximizing fitness objectives. A Fitness type is single-objective, when the length
of the *weights* tuple is **equal to one**, and multi-objective, when it's
**greater than one**. The weights can also be used to vary the importance of
objectives against each other by varying the sizes of the numbers. The type and
quantity of fitness *weights* should be chosen according to the nature of the
problem to be solved.

The following examples illustrate the various ways of creating different types of
Fitness. The `create_type()` function takes at least two arguments: a name for
the subclass and a base class to inherit from. All subsequent arguments, if there
are any, become the attributes of the new type.

**Single-objective**

```python
from deap_er import Fitness, creator

creator.create_type("FitnessMin", Fitness, weights=(-1.0,))  # Minimizing
creator.create_type("FitnessMax", Fitness, weights=(1.0,))  # Maximizing
```

**Multi-objective**

```python
creator.create_type("FitnessMulti", Fitness, weights=(-1.0, 1.0))  # Min and max
creator.create_type("FitnessVaried", Fitness, weights(0.5, 1.1, -1.7))  # Varied importance
```

### Individuals

Individuals are collections of solution values which mutate and mate with each other
in order to produce offsprings with altered solution values. Each individual has a
*fitness* attribute, which represents the suitability of the solution values for the
given problem. During evolution, the population of each generation is evaluated and
those individuals with better fitness scores pass their solution values to the next
generation.

Individuals are created by using the creator module and must inherit from mutable
types of `Collection`. In addition to the standard `list` class, it is also possible
to create individuals based on the `array.array` or `numpy.ndarray` classes:

```python
creator.create_type("Individual", list, fitness=creator.FitnessMax)
creator.create_type("Individual", numpy.ndarray, fitness=creator.FitnessMax)
creator.create_type("Individual", array.array, typecode="i", fitness=creator.FitnessMax)
```

After an **Individual** subclass with a *fitness* attribute has been created, it
must be registered into a [`Toolbox`](../reference/base.md). In the following
example, calling `toolbox.individual()` creates a single individual of type
`creator.Individual`.

```python
from deap_er import Toolbox, creator, tools
import random

toolbox = Toolbox()
toolbox.register("attr_float", random.random)      # alias and func
toolbox.register("individual", tools.init_repeat,  # alias and func
    container=creator.Individual,                  # passed to init_repeat
    func=toolbox.attr_float,                       # passed to init_repeat
    size=10                                        # passed to init_repeat
)
ind = toolbox.individual()  # creates a single individual
```

### Populations

Populations are collections of individuals, strategies or particles. Individuals
are usually not created directly one-by-one, but in bulk by a population generator
in the toolbox. Please refer to [Types of Populations](#types-of-populations) for
more details on generating different types of populations.

## Types of Individuals

### A List of Random Floats

This type of an individual can be created by using `random.random()`. The following
example creates a single individual, which is a list of **10** random floating-point
numbers and has a **fitness** attribute of the single-objective maximizing type.

```python
from deap_er import Fitness, Toolbox, creator, tools

creator.create_type("FitnessMax", Fitness, weights=(1.0,))
creator.create_type("Individual", list, fitness=creator.FitnessMax)

toolbox = Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.init_repeat,
    container=creator.Individual,
    func=toolbox.attr_float,
    size=10
)
ind = toolbox.individual()
```

### A List of Permutations

This type of an individual can be created by using `random.sample()`. The following
example creates a single individual, which is a list of **10** permuted values of
the integers **0** through **9** and has a **fitness** attribute of the
single-objective minimizing type.

```python
from deap_er import Fitness, Toolbox, creator, tools

creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
creator.create_type("Individual", list, fitness=creator.FitnessMin)

toolbox = Toolbox()
toolbox.register("indices", random.sample,
    population=range(10), k=10
)
toolbox.register("individual", tools.init_iterate,
    container=creator.Individual,
    generator=toolbox.indices
)
```

### A Tree of Expressions

This type of an individual is a prefix tree of mathematical expressions which is
used to solve symbolic regression problems. The prefix tree needs a collection of
all the available mathematical operators that the individual could use to solve
the problem in the form of a [`PrimitiveSet`](../reference/gp.md). The following
example creates a single individual of type `PrimitiveTree`, which can use `add`,
`sub`, and `mul` operators and has a **fitness** attribute of the single-objective
minimizing type.

!!! note
    Arity represents the number of arguments an operator takes.

```python
from deap_er import Fitness, Toolbox, creator, tools, gp
import operator

pset = gp.PrimitiveSet("MAIN", arity=1)
pset.add_primitive(operator.add, arity=2)
pset.add_primitive(operator.sub, arity=2)
pset.add_primitive(operator.mul, arity=2)

creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
creator.create_type("Individual", gp.PrimitiveTree,
    fitness=creator.FitnessMin, prim_set=pset
)
toolbox = Toolbox()
toolbox.register("expr", gp.gen_half_and_half,
    prim_set=pset, min_depth=1, max_depth=2
)
toolbox.register("individual", tools.init_iterate,
    container=creator.Individual,
    generator=toolbox.expr
)
ind = toolbox.individual()
```

### Evolution Strategies

Individuals with evolution strategies are slightly different as they generally
contain two lists: one for the fitness values and the other for the mutation
parameters. Since there are no builtin helper functions which generate two
different iterables at the same time, we must define this function ourselves.
The following example creates a single individual, which has an evolution
strategy and a **fitness** attribute of the single-objective minimizing type.

```python
from deap_er import Fitness, Toolbox, creator
import random

creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
creator.create_type("Individual", list, fitness=creator.FitnessMin, strategy=None)
creator.create_type("Strategy", list)

def init_evo_strat(individual, strategy, i_size, i_min, i_max, s_min, s_max):
    ind = individual(random.uniform(i_min, i_max) for _ in range(i_size))
    ind.strategy = strategy(random.uniform(s_min, s_max) for _ in range(i_size))
    return ind

toolbox = Toolbox()
toolbox.register("individual", init_evo_strat,
    individual=creator.Individual,
    strategy=creator.Strategy,
    i_size=10,
    i_min=-5, i_max=5,
    s_min=-1, s_max=1
)
ind = toolbox.individual()
```

### Moving Particles

A particle is an individual with a speed vector and can remember its best
position. Particles are used for solving particle swarm optimization problems.
Since there are no builtin helper functions to generate particles, we must
define this function ourselves. The following example creates a single
individual, which has a speed vector and a **fitness** attribute of the
multi-objective maximizing type.

```python
from deap_er import Fitness, Toolbox, creator
import random

creator.create_type("FitnessMax", Fitness, weights=(1.0, 1.0))
creator.create_type("Particle", list, fitness=creator.FitnessMax,
    speed=None, s_min=None, s_max=None, best=None
)
def init_particle(cr_cls, size, pos_min, pos_max, spd_min, spd_max):
    particle = cr_cls(random.uniform(pos_min, pos_max) for _ in range(size))
    particle.speed = [random.uniform(spd_min, spd_max) for _ in range(size)]
    particle.spd_min = spd_min
    particle.spd_max = spd_max
    return particle

toolbox = Toolbox()
toolbox.register("particle", init_particle, cr_cls=creator.Particle,
    size=2, pos_min=-6, pos_max=6, spd_min=-3, spd_max=3
)
ind = toolbox.particle()
```

### Custom Types

If your evolution problem can't be solved using the previously described
individuals, it's also possible to create individuals with custom behaviors
depending on your requirements. The following example creates a single
individual, which is a list of alternating integer and floating point
numbers `[int, float, int, float, ...]` and has a **fitness** attribute of
the multi-objective maximizing type.

```python
from deap_er import Fitness, Toolbox, creator, tools
import random

creator.create_type("FitnessMax", Fitness, weights=(1.0, 1.0))
creator.create_type("Individual", list, fitness=creator.FitnessMax)

INT_MIN, INT_MAX = 5, 10
FLT_MIN, FLT_MAX = -0.2, 0.8
N_CYCLES = 4

toolbox = Toolbox()
toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("attr_float", random.uniform, FLT_MIN, FLT_MAX)
toolbox.register("individual", tools.init_cycle,
    container=creator.Individual,
    funcs=(toolbox.attr_int, toolbox.attr_float),
    size=N_CYCLES
)
ind = toolbox.individual()
```

## Types of Populations

### Bags

A bag is the most commonly used type of a population. It has no particular
ordering and is usually implemented as a list. The population is initialized
using `init_repeat()` and created by calling `toolbox.population(size=num)`,
where **num** is the quantity of individuals in the population.

```python
toolbox.register("population", tools.init_repeat,  # alias and func
    container=list, func=toolbox.individual        # passed to init_repeat
)
pop = toolbox.population(size=100)  # creates a population of 100 individuals
```

### Grids

A grid is a special case of a structured population where neighbouring
individuals are associated with each other. The individuals are distributed
in a grid pattern, where each cell contains a single individual. It is usually
implemented as a list of rows, where each row is a list of individuals. The
length of the row determines the number of columns in the grid. The individuals
are accessible using two consecutive subscript operators `pop[i][j]`
(row, column).

```python
NUM_COLUMNS, NUM_ROWS = 50, 100

toolbox.register("row", tools.init_repeat,  # a row of columns
    container=list,
    func=toolbox.individual,
    size=NUM_COLUMNS
)
toolbox.register("population", tools.init_repeat,  # a population of rows
    container=list,
    func=toolbox.row,
    size=NUM_ROWS
)
pop = toolbox.population()  # size was already implicitly passed
```

### Swarms

Swarm-type populations are used for solving particle swarm optimization
problems. Please refer to [Moving Particles](#moving-particles) on how to
create particles for swarm-type populations. A particle swarm contains a
communication network between the particles. The simplest network is a
completely connected one, where each particle knows the best position that
has ever been visited by any particle. This can be implemented by recording
the best position and the best fitness as population attributes:

```python
creator.create_type("Swarm", list, g_best_pos=None, g_best_fit=creator.FitnessMax)
toolbox.register("swarm", tools.init_repeat, creator.Swarm, toolbox.particle)
pop = toolbox.swarm(size=100)  # creates a swarm of 100 particles
```

### Demes

A deme is a sub-population that is contained inside a population. In the
following example, a population of 3 demes is created, where each deme has
a different number of individuals:

```python
DEME_SIZES = [10, 50, 100]
toolbox.register("deme", tools.init_repeat, list, toolbox.individual)
population = [toolbox.deme(size=size) for size in DEME_SIZES]
```

## Seeding Populations

Sometimes, it could be useful to initialize the population from a list of
predefined first-guess individuals. This can be achieved by defining a
population initialization function that gets called when the population is
being created. In the following example, the population will be initialized
from the `first_guess.json` file that contains a list of first-guess
individuals. This can be combined with a regular population initializer to
create a population of part random and part non-random individuals (not
part of the example).

```python
from deap_er import Fitness, Toolbox, creator
import json

def init_population(pop_type, ind, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pop_type(ind(c) for c in contents)

creator.create_type("FitnessMax", Fitness, weights=(1.0, 1.0))
creator.create_type("Individual", list, fitness=creator.FitnessMax)

toolbox = Toolbox()
toolbox.register("population", init_population,
    list, creator.Individual, "first_guess.json")

population = toolbox.population()
```

**first_guess.json**

```json
[
    [1, 2, 3, 4, 5, 6],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
]
```
