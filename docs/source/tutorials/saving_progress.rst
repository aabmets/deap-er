Saving Progress
===============

In this tutorial, we will describe how persistence can be achieved for evolution algorithms.
This library has a helper class named :class:`~deap_er.other.Checkpoint`, which can be
used to save the current state of an evolution algorithm to disk and to restore it later to
resume the computation.

:class:`~deap_er.other.Checkpoint` objects use the `dill <https://pypi.org/project/dill/>`_ library for
object (de-)serialization, because it supports more Python types, like lambdas, than the default :mod:`pickle`.
Checkpoint objects can be used either manually by using the :func:`~deap_er.other.Checkpoint.save()` and
:func:`~deap_er.other.Checkpoint.load()` methods or automatically by using the custom
:func:`~deap_er.other.Checkpoint.range()` method. Automatic checkpointing is also available
for the standard evolution algorithms of this library.

In the following partial example, we will use the :func:`~deap_er.other.Checkpoint.range()` method to save
the progress to disk every **SAVE_FREQ** generations. If one should want to resume the computation later, they
would only have to pass the name or path of the checkpoint file to the :class:`~deap_er.other.Checkpoint`
constructor, as the data is automatically loaded from disk on initialization by default.


.. code-block::

    # Toolbox setup is omitted for brevity

    SAVE_FREQ = 100
    NUM_GENS = 1000

    def main(file=None):
        toolbox, stats = setup()
        cp = tools.Checkpoint(file)

        if not file:
            cp.pop = toolbox.population(size=300)
            cp.hof = tools.HallOfFame(maxsize=1)
            cp.log = tools.Logbook()

        for gen in cp.range(NUM_GENS, SAVE_FREQ):
            population = algorithms.var_and(
                toolbox=toolbox, population=cp.pop,
                cx_prob=0.5, mut_prob=0.2
            )
            invalids = [ind for ind in population if not ind.fitness.is_valid()]
            fitness = toolbox.map(toolbox.evaluate, invalids)
            for ind, fit in zip(invalids, fitness):
                ind.fitness.values = fit

            cp.hof.update(population)
            record = stats.compile(population)
            cp.log.record(gen=gen, evals=len(invalids), **record)

            cp.pop = toolbox.select(population, sel_count=len(population))

.. attention::
    Only those objects that are in the :code:`__dict__` attribute of the checkpoint object will be saved.
