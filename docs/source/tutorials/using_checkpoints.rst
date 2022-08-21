Using Checkpoints
=================

In this tutorial, we will describe how persistence can be achieved for evolution algorithms.
This library has a helper class named :class:`~deap_er.persistence.checkpoint.Checkpoint`, which can
be used to save the current state of an evolution algorithm to disk and restore it later to resume
the computation. You can read more about the Checkpoint object from the :ref:`State Persistence
<persistence>` section of the Reference Manual.

Checkpoint objects use the `dill <https://pypi.org/project/dill/>`_ library for object (de-)serialization,
because it supports more Python types like lambdas, than the default :mod:`pickle` library. Checkpoints can
be used either manually with the :func:`save()` and :func:`load()` methods or automatically with the custom
:func:`range()` generator. The builtin algorithms don't implement automatic checkpointing due to their
simplistic nature, but the user is able to implement manual checkpointing around them.

In the following example, we will use the :func:`range()` generator to save the progress to disk every
**save_freq** seconds. If one should wish to resume the computation later, they would only have to pass
the name or path of the checkpoint file to the checkpoint constructor, as the data is automatically
loaded from the disk on object initialization by default.


.. code-block::

    from deap_er import base, creator, tools, env

    # setup() definition is omitted for brevity

    def main(file=None):
        toolbox, stats = setup()
        cp = env.Checkpoint(file)
        cp.save_freq = 10  # every 10 seconds

        if not cp.is_loaded():  # skip if loaded
            cp.pop = toolbox.population(size=300)
            cp.hof = tools.HallOfFame(maxsize=1)
            cp.log = tools.Logbook()
            fields = stats.fields if stats else []
            cp.log.header = ['gen', 'nevals'] + fields

        for gen in cp.range(1000):
            # evolve new offspring from parent pop
            offspring = tools.var_and(
                toolbox=toolbox,
                population=cp.pop,
                cx_prob=0.5,
                mut_prob=0.2
            )
            # update fitness values of individuals
            invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
            fitness = toolbox.map(toolbox.evaluate, invalids)
            for ind, fit in zip(invalids, fitness):
                ind.fitness.values = fit

            # persist the hof, log and offspring
            cp.hof.update(offspring)
            record = stats.compile(offspring)
            cp.log.record(gen=gen, nevals=len(invalids), **record)
            cp.pop = toolbox.select(offspring, sel_count=len(offspring))

            # the range() generator persists the cp to disk
            # if saving conditions are fulfilled

.. attention::
    Only those objects that are attributes of the checkpoint object will be saved to disk.

.. raw:: html

   <br />
