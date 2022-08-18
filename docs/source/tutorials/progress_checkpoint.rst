Saving and Loading Progress
===========================

In this tutorial, we will describe how persistence can be achieved for evolution algorithms.
This library has a helper class named :class:`~deap_er.controllers.checkpoint.Checkpoint`, which can
be used to save the current state of an evolution algorithm to disk and restore it later to resume
the computation. The checkpoint controller is available from the ``env`` sub-package of this library.

Checkpoint objects use the `dill <https://pypi.org/project/dill/>`_ library for object (de-)serialization, because
it supports more Python types like lambdas, than the default :mod:`pickle` library. Checkpoints can be used either
manually with the :func:`save()` and :func:`load()` methods or automatically with the custom :func:`range()` method.
Automatic checkpointing is also available for the standard evolution algorithms of this library.

In the following partial example, we will use the :func:`range()` method to save the progress to disk every
**save_freq** generations. If one should wish to resume the computation later, they would only have to pass
the name or path of the checkpoint file to the checkpoint constructor, as the data is automatically loaded
from the disk on object initialization by default.

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
            population = tools.var_and(
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

.. raw:: html

   <br />
