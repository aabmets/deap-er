.. _ops_and_algos:

Operators and Algorithms
========================

This section describes the operators and algorithms that are available in the :code:`deap_er.tools` module.
For all subsequent examples, assume we have two individuals available as defined below:

.. code-block::

    from deap_er import base, tools
    import random

    IND_SIZE = 5

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.init_repeat,
        container=creator.Individual,
        func=toolbox.attr_float,
        size=IND_SIZE)

    ind1 = toolbox.individual()
    ind2 = toolbox.individual()

.. raw:: html

   <hr>


Operators
+++++++++

Evaluation
----------

Evaluation operators are responsible for generating fitness values from the solution values
of an individual. This is the only operator that the user has to implement themselves, as it's
specific to each optimization problem. A typical evaluation function takes an individual as the
only argument and returns its fitness value(s) as a tuple of floats. Please refer to the
:ref:`Creating Individuals <individuals>` chapter on information about Fitness.

The following example evaluates the previously created individual **ind1** and assigns
the resulting fitness values to the :code:`fitness.values` attribute of the individual:

.. code-block::

    def evaluate(individual) -> tuple[float]:  # Must be a tuple of float(s)
        # Compute the fitness value(s)
        return fit1, fit2

    print(ind1.fitness.is_valid())  # False
    print(ind1.fitness.values)      # ()

    ind1.fitness.values = evaluate(ind1)

    print(ind1.fitness.is_valid())  # True
    print(ind1.fitness.values)      # (float, float)

.. attention::
    All evaluation functions must return a **tuple of float(s)** for all types of fitness objectives.

.. raw:: html

   <br />


Mutation
--------

Mutation operators are responsible for mutating the solution values of an individual. There are a
variety of mutation operators available in the :code:`deap_er.tools` module. Each mutation operator
has its own characteristics and therefore it's recommended to read their documentation before use to
avoid undesirable behavior. Please refer to the :ref:`Mutation Operators <mut_ops>` section of the
:ref:`Reference Manual <reference>` to learn more.

In the following example, the original individual is cloned and the clone is mutated. This can be done
to preserve the original individual if needed, as mutation operators directly modify the input individuals.
After mutation, the fitness values of the mutant must be deleted, because they are no longer relevant
to the solution values of the mutant.

.. code-block::

    clone = toolbox.clone(ind1)
    tools.mut_flip_bit(clone, mut_prob=0.2)
    del clone.fitness.values

.. raw:: html

   <br />


Crossover
---------

Crossover operators are responsible for mating the solution values of two or more individuals. There are
a variety of crossover operators available in the :code:`deap_er.tools` module. Each crossover operator
has its own characteristics and therefore it's recommended to read their documentation before use to
avoid undesirable behavior. Please refer to the :ref:`Crossover Operators <cx_ops>` section of the
:ref:`Reference Manual <reference>` to learn more.

In the following example, the original individuals are cloned and the clones are mated with each other.
This can be done to preserve the original individuals if needed, as crossover operators directly modify
the input individuals. After mating, the fitness values of the offsprings must be deleted, because they
are no longer relevant to the solution values of the offspring.

.. code-block::

    clone1 = toolbox.clone(ind1)
    clone2 = toolbox.clone(ind2)
    tools.cx_blend(clone1, clone2, cx_prob=0.5)
    del clone1.fitness.values
    del clone2.fitness.values

.. raw:: html

   <br />


Selection
---------

Selection operators are responsible for selecting individuals for subsequent evolution processes. There are
a variety of selection operators available in the :code:`deap_er.tools` module. Each selection operator has
its own characteristics and therefore it's recommended to read their documentation before use to avoid
undesirable behavior. Please refer to the :ref:`Selection Operators <sel_ops>` section of the
:ref:`Reference Manual <reference>` to learn more.

In the following example, 10 individuals are selected from a population. The selected individuals can be
cloned after selection if needed, to preserve the original individuals for other processing. Please refer
to the :ref:`Creating Individuals <individuals>` chapter to learn more about populations.

.. code-block::

    selected = tools.sel_best(population, sel_count=10)
    selected = [toolbox.clone(ind) for ind in selected]

.. raw:: html

   <br />


Variation
---------

Variation functions are building blocks of evolution algorithms, which alter the application of crossover
and mutation operators depending on the given probabilities. To use a variation function, a **mate** and
**mutate** aliases must be registered into the toolbox with the necessary crossover and mutation operators.
Please refer to the :ref:`Algorithms <algos>` section of the :ref:`Reference Manual <reference>` to learn more.

.. code-block::

    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)

    offsprings = tools.var_and(toolbox, selected, CX_PROB, MUT_PROB)

.. raw:: html

   <br />
   <hr>


Algorithms
++++++++++

Evolutionary algorithms are the main workhorses of computational evolution, which alter the individuals
of the input population with the operators of a toolbox to solve optimization problems.

A few generic evolutionary algorithms have been built into this library, which can be used for various
different optimization problems and can accept any kind of individuals and operators as input. Please
refer to the :ref:`Algorithms <algos>` section of the :ref:`Reference Manual <reference>` to learn more.
The following examples demonstrate the most basic ways of solving optimization problems:

.. code-block::
   :caption: Using a builtin algorithm

    # toolbox and population setup is omitted for brevity

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=500,
        cx_prob=0.5,
        mut_prob=0.2
    )
    pop, log = tools.ea_simple(**args)

.. code-block::
   :caption: Using a variation function

    # toolbox and population setup is omitted for brevity

    for gen in range(GENS):
        selection = toolbox.select(pop, len(pop))
        offspring = map(toolbox.clone, selection)

        offspring = tools.var_and(toolbox, offspring, CX_PROB, MUT_PROB)

        invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitness):
            ind.fitness.values = fit

        pop[:] = offspring

.. code-block::
    :caption: Using custom crossover and mutation

    # toolbox and population setup is omitted for brevity

    for gen in range(GENS):
        offspring = toolbox.select(pop, len(pop))
        offspring = map(toolbox.clone, offspring)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitness):
            ind.fitness.values = fit

        pop[:] = offspring

.. raw:: html

   <br />
