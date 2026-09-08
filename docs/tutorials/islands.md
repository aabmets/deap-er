# Island Models

A deme is a sub-population that evolves on its own toolbox, then
exchanges individuals with other demes. [`step_islands`](../reference/algorithms.md)
runs one evaluate → vary → select step on each deme. A migration
callable then moves emigrants.

The contract is easy to get wrong. Each deme toolbox must register
**`vary`**, **`select`**, and **`evaluate`** (or `evaluate_batch`).
`mate` and `mutate` alone are not enough — wrap them with `var_and`
or `var_or` and register that as `vary`.

A complete three-deme OneMax run is the
[islands example](../examples/genetic_algorithms/islands.md).
Creating the list of demes is also in
[Creating Individuals](creating_individuals.md#demes).

## One step

```python
from deap_er import tools

def vary(population):
    return tools.var_and(toolbox, population, cx_prob=0.5, mut_prob=0.2)

toolbox.register("vary", vary)
toolbox.register("select", tools.sel_tournament, contestants=3)
toolbox.register("evaluate", evaluate)

demes = [(toolbox, pop_a), (toolbox, pop_b), (toolbox, pop_c)]
tools.step_islands(demes)
```

Each pair is `(toolbox, population)`. Populations are modified in
place. After the step, `select(offspring, len(population))` replaces
the deme. Unlike toolboxes are allowed — one deme can use a different
`vary` or `evaluate`.

## Migration

Pass `migrate` to move individuals after every deme has stepped.
`mig_ring` sends each deme to the next. `mig_fully_connected` sends
along every directed edge. `mig_random` picks one destination per
source. All three take a `selection` callable and a `mig_count`:

```python
def migrate(populations):
    tools.mig_ring(populations, mig_count=5, selection=tools.sel_best)

tools.step_islands(demes, migrate=migrate)
```

Emigrants are placed by object identity. When a replacement operator
is omitted, a source individual that would otherwise be aliased into
two demes is cloned.

## Different exams per deme

When demes evaluate on different case subsets or matrices, a migrant's
fitness is stale. Pass `eval_keys=` so `step_islands` clears fitness
on arrival. `island_eval_keys` hashes each deme's `CaseExam` and an
optional matrix identity:

```python
keys = tools.island_eval_keys(exams, n_cases=20)
tools.step_islands(demes, migrate=migrate, eval_keys=keys)
```

Omit `eval_keys` when every deme scores the same data — migrants keep
their fitness.

Persist the list of demes on a
[`Checkpoint`](using_checkpoints.md). There is no island daemon and
no extra process pool; `evaluate_batch` still owns parallelism.
See [Multiprocessing](multiprocessing.md) and the
[Algorithms](../reference/algorithms.md) reference.
