# Using Checkpoints

In this tutorial, we will describe how persistence can be achieved for evolution
algorithms. This library has a helper class named
[`Checkpoint`](../reference/persistence.md), which can be used to save the
current state of an evolution algorithm to disk and restore it later to resume
the computation.

Checkpoint objects use the [dill](https://pypi.org/project/dill/) library for
object (de-)serialization, because it supports more Python types like lambdas
than the default `pickle` library. Checkpoints can be used either manually with
the `save()` and `load()` methods or automatically with the custom `range()`
generator. The builtin algorithms don't implement automatic checkpointing due
to their simplistic nature, but the user is able to implement manual
checkpointing around them.

`save()` writes a sibling `.tmp` file and replaces the destination, so
a dump that fails part-way through does not truncate a good file. The
process-wide `tools.rng` state is persisted with the attributes you
set on the checkpoint. Child streams from `spawn_rng` are not — those
are derived again from the run seed. `save_freq = -1` disables
automatic saves during `range()`. `last_op` is one of `none`,
`load_success`, `load_error`, `save_success`, or `save_error`.

The default file is a UUID with a `.dcpf` extension under
`<cwd>/deap-er`. Pass `file_name` (and optionally `dir_path`) to
choose the path. `autoload=True` (the default) calls `load()` during
construction.

In the following example, we will use the `range()` generator to save the
progress to disk every **save_freq** seconds. If one should wish to resume
the computation later, they would only have to pass the name of the
checkpoint file to the constructor.

```python
from deap_er import Checkpoint, tools

# setup() definition is omitted for brevity

def main(file=None):
    toolbox, stats = setup()
    cp = Checkpoint(file)
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
        cp.pop = toolbox.select(offspring, len(offspring))

        # the range() generator persists the cp to disk
        # if saving conditions are fulfilled
```

!!! attention
    Only those objects that are attributes of the checkpoint object will be saved to disk.
