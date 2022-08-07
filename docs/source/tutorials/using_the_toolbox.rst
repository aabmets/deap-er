.. _toolbox:

Using the Toolbox
=================

The :class:`~deap_er.base.Toolbox` class is the main component of computational evolution, which is a
container for all the necessary tools to build genetic algorithms and solve evolutionary problems with.
Using a toolbox enables the separation of algorithms from the operators, which makes it easy to hot swap
the operators during evolution in a plug-and-play manner as needed.

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
    toolbox.register("first_alias", add_func, a=1, b=2, c=3)  # args are passed implicitly
    toolbox.register("other_alias", add_func)                 # args are passed explicitly

    result = toolbox.first_alias()         # result == 6
    result = toolbox.other_alias(1, 2, 3)  # result == 6

The preceding code registers two aliases in the toolbox: **first_alias** and **other_alias**, which
point to the same function **add_func**. In the case of :code:`toolbox.first_alias()`, the arguments
were already passed during registration, so the method needs to be called without arguments. In the
case of :code:`toolbox.other_alias(1, 2, 3)`, the arguments must be passed explicitly, because the
method was registered without arguments. The difference between the two methods is that the arguments
of **first_alias** are static, while the arguments of the **other_alias** are dynamic.

.. raw:: html

   <br />

Tool Registration
-----------------

A toolbox is only as useful as the tools registered into it. When a function is registered into a toolbox,
it can be called as a method of the toolbox with the same alias that was used to register it. Because the
methods are added in a dynamic manner, linting hints are not available for them and IDEs incorrectly mark
their usage as erroneous due to lack of code references.

.. note::
    Some commonly used tool aliases, such as **mate**, **mutate**, **select** and **evaluate**, do
    have linting hints available in Toolbox objects and IDEs don't complain about their code references.

The **tools** module contains a number of useful tools that can be registered into a toolbox. While all
tools in the module can be registered into a toolbox, some of them, such as algorithms or statistics, are
equally useful independently. More on tools in the :ref:`Operators and Algorithms <ops_and_algos>` chapter.

.. code-block::

    from deap_er import base, tools

    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", tools.bm_sphere)

    pop, log = tools.ea_simple(**args)

.. raw:: html

   <br />


Tool Decoration
---------------

Tool decoration is a powerful feature that allows the precise control of parameters during the
evolution process. For example, in the case of constrained domains, a tool decorator can be used
on mutation and crossover operators to prevent the individuals from growing out-of-bounds.

The following example defines a decorator that checks if the solution values of offsprings are
out-of-bounds and clamps them to the predefined limit values if this is the case. Whenever either
of the decorated tools is called, bounds will be checked on the resulting offsprings.

.. note::
    This decorator works for both crossover and mutation operators,
    because the return type of these operators must be a **tuple**.

.. code-block::

    def clamp(min, max):
        def wrapper(func):
            def wrapped(*args, **kwargs):
                offsprings: tuple = func(*args, **kwargs)
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
