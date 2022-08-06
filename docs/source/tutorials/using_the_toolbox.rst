Using the Toolbox
=================

The :class:`~deap_er.base.Toolbox` class is the main component of computational evolution, which is a
container for all the necessary tools to build genetic algorithms and solve evolutionary problems with.
Using a toolbox enables the separation of the algorithms from the operators, which makes it easy to
hot swap the operators during evolution in a plug-and-play manner as needed.

The two most important methods of a toolbox are :func:`~deap_er.base.Toolbox.register()` and
:func:`~deap_er.base.Toolbox.unregister()`, which are used to add or remove tools from the toolbox.
The :func:`~deap_er.base.Toolbox.register()` method takes at least two arguments: an alias and a
function that is going to be assigned to the alias. All subsequent arguments, if there are any,
will be implicitly passed into the registered function when the tool is called. Any name can be
registered into a toolbox as long as it's unique for that toolbox instance.



.. code-block::

    from deap_er import base

    def add_func(a, b, c):
        return a + b + c

    toolbox = base.Toolbox()
    toolbox.register("unique_name", add_func,   # alias and the function
        a=1, b=2, c=3  # implicitly passed to add_func when this tool is called
    )
    result = toolbox.unique_name()  # result == 6

.. raw:: html

   <br />

Tool Registration
-----------------

A toolbox is only as useful as the tools registered into it. When a function is registered into a toolbox,
it can be called as a method of the toolbox with the same alias that was used to register it. Because the
methods are added in a dynamic manner, IDE autocompletion is not available for them and IDEs incorrectly
mark the usage of these tools as erroneous due to lack of code references.

The most commonly used names for tools are **mate**, **mutate**, **select** and **evaluate**. For these
aliases and some others there are linting hints available, which makes them available for autocompletion
and IDEs don't complain about their usage. Nevertheless, all dynamically added methods are correctly
available through the toolbox instance as expected.

.. code-block::

    from deap_er import base, tools
    import random

    def evaluate(individual) -> tuple:  # Return type must be a tuple
        # Do some computation
        return result,

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_gaussian, mu=0, sigma=1, mut_prob=0.2)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("randomizer", random.randrange, start=0, stop=10)

    fit = toolbox.evaluate(individual)  # Each individual is passed explicitly
    rand = toolbox.randomizer()  # IDE complains, but works as expected

.. raw:: html

   <br />


Tool Decoration
---------------

Tool decoration is a powerful feature that allows the precise control of evolution operations without
changing the algorithm parameters or modifying operators. For example, in the case of a constrained
domain, one can apply a decorator to the mutation and crossover operators in order to keep the
individuals from growing out-of-bounds.

The following example defines a decorator that checks if the solution values of offsprings are out-of-bounds
and clamps them to the predefined limit values if this is the case. This decorator is defined using three
functions in order to pass the **MIN** and **MAX** arguments correctly to the decorated functions. Whenever
the mutation or crossover method is called, bounds will be checked on the resulting offsprings. This decorator
works for both crossover and mutation operators, because the return type of these operators must be an iterable.

.. code-block::

    def clamp(min, max):
        def wrapper(func):
            def wrapped(*args, **kwargs):
                offsprings = func(*args, **kwargs)
                for child in offsprings:
                    for i in range(len(child)):
                        if child[i] > max:
                            child[i] = max
                        elif child[i] < min:
                            child[i] = min
                return offspring
            return wrapped
        return wrapper

    toolbox.register("mate", tools.cx_blend, alpha=0.2)
    toolbox.register("mutate", tools.mut_gaussian,
        mu=0, sigma=2, mut_prob=0.2
    )
    toolbox.decorate("mate", clamp(MIN, MAX))
    toolbox.decorate("mutate", clamp(MIN, MAX))

.. raw:: html

   <br />
