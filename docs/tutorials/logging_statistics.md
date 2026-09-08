# Logging Statistics

## The Statistics Object

Often, there is a need to understand what is going on in the optimization
process. This can be achieved with the [`Statistics`](../reference/records.md)
object, which is able to compile statistics on arbitrary attributes of any
designated objects. The statistics object constructor takes a single argument
of a callable, which should return the data that the statistics will be
computed on. Usually, the statistics are computed on the fitness values of
individuals, as in the following example:

```python
from deap_er import tools

# using a func

def source(ind):
    return ind.fitness.values

stats = tools.Statistics(key=source)

# using a lambda equivalent

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
```

After the statistics object has been created, it is necessary to register the
needed statistical functions into the **stats** object. The register function
expects a name alias as the first argument and for the second argument it
expects a function which is able to operate on a vector of values. Any
subsequent arguments are passed to this function when it's called. Usually,
numpy statistical functions are used:

```python
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
```

### Builtin Algorithms

When using [builtin algorithms](../reference/algorithms.md), a statistics
object can be given as an argument to the function. During the optimization
process, statistics will be automatically computed on the population every
generation. If the verbose argument is **True**, the statistics are also
printed to the console. At the end of the optimization process, the builtin
algorithms return the final population and the logbook.

```python
pop, log = tools.ea_simple(
    toolbox=toolbox,
    population=pop,
    generations=100,
    cx_prob=0.5,
    mut_prob=0.2,
    stats=stats,
    hof=hof,
    verbose=True,
    logger=logger,
    log_time=True,
    fronts=fronts,
    n_evals=50_000,
)
```

`log_time` adds a per-generation `duration`. `logger` receives the
stream instead of `print` when `verbose` is True. `fronts` appends
a new `ParetoFront` of the current population each generation.
`n_evals` stops after the generation that meets the evaluation
budget. `hof` is a `HallOfFame` or `ParetoFront` updated in place.

### Custom Algorithms

When writing custom evolution algorithms, any statistics can be obtained by
compiling it on the desired objects. The argument to the compile function
must be an iterable of elements on which the key function will be called.
In the next example, we will compile statistics on a population of
individuals. The statistics object will call the key function on every
individual to retrieve their `fitness.values` attribute. The resulting
array of values is finally given to each registered statistical function
and the results are put into the record dictionary under the key associated
with the function. Usually, the records are computed on each generation of
individuals and the resulting records are inserted into the logbook.

```python
# inside the main evolution loop

record = stats.compile(population)
```

```text
>>> print(record)
{
    'std': 4.96,
    'max': 63.0,
    'avg': 50.2,
    'min': 39.0
}
```

### Multi-Objective Statistics

When statistics are computed directly on the values with numpy functions,
all the objectives are combined together, which is the default behaviour of
numpy. In order to separate the objectives, an axis of operation needs to
be specified. This is done by providing the axis as an additional argument
to the registration function. The axis can also be specified for
single-objective statistics, which causes the output to be in the format
of numpy arrays.

```python
stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

record = stats.compile(population)
```

```text
>>> print(record)
{
    'std': array([4.96]),
    'max': array([63.0]),
    'avg': array([50.2]),
    'min': array([39.0])
}
```

### Multiple Statistics

It is also possible to compute statistics on multiple attributes of an
individual by using a [`MultiStatistics`](../reference/records.md) object.
For instance, it is quite common in genetic programming to compile
statistics on the height of the trees as well as their fitness values.
In order to achieve this, we define multiple regular statistics objects
with the required key functions and then pass them into a
`MultiStatistics` constructor. The statistical functions can be
registered only once either in the multi-statistics object or
individually in each statistics objects. `mstats.register(...,
chapters=)` can target a subset of chapter names when one function
should not run on every chapter. The multi-statistics object
can be then given to a builtin algorithm or they can be compiled using
the exact same procedure as for the simple statistics objects. A
compiled multi-statistics record is a dictionary of statistics objects.
`compile` materializes the input once so a generator is not consumed
by the first chapter.

```python
stats_size = tools.Statistics(key=len)
stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_dups = tools.Statistics()
stats_dups.register("dups", tools.duplicate_count)
mstats = tools.MultiStatistics(
    fitness=stats_fit, size=stats_size, variety=stats_dups
)

mstats.register("avg", numpy.mean, chapters=("fitness", "size"))
mstats.register("std", numpy.std, chapters=("fitness", "size"))
mstats.register("min", numpy.min, chapters=("fitness", "size"))
mstats.register("max", numpy.max, chapters=("fitness", "size"))

record = mstats.compile(population)
```

```text
>>> print(record)
{
    'fitness': {
        'std': 1.64,
        'max': 6.86,
        'avg': 1.71,
        'min': 0.166
    },
    'size': {
        'std': 1.89,
        'max': 7,
        'avg': 4.54,
        'min': 3
    }
}
```

## The Logbook Object

The logbook is intended to be a chronological sequence of dictionaries. It
is directly compliant with, but not limited to, the type of data returned
by statistics and multi-statistics objects. Anything can be incorporated
into an entry of a logbook.

### Logging Data

Once a record has been compiled by a statistics object, it can be recorded
into a Logbook by unpacking the record dictionary into the `record()`
method. The `record()` method takes a variable number of arguments, each
of which is data to be recorded. The records of a logbook can be later
retrieved with the `select()` method, which accepts a variable number of
string arguments of either the keywords that the data was recorded under
or name aliases of statistical functions.

```python
# record creation omitted for brevity

logbook = tools.Logbook()
logbook.record(gen=0, nevals=30, **record)

# later when analyzing the evolution
gen, avg = logbook.select("gen", "avg")
```

### Printing to Screen

A logbook can be printed to screen or a file. Its `__str__()` method
returns a header of each key inserted in the first record and then the
complete logbook for each of these keys. The rows are in chronological
order of insertion while the columns are in an undefined order, which
can be made specific by setting the header attribute of the Logbook
object to a list of strings with the column names.

```python
logbook.header = "gen", "food", "avg"
```

If an entry is missing from a row, the space in the column will be left
blank for that entry. An empty logbook that already has a `header`
still prints that header.

`Logbook.to_json` / `from_json` round-trip entries, chapters, and
the header. NumPy scalars become Python numbers.

```text
>>> print(logbook)
gen   food    avg
0     apple   50.2
1             45.9
3     meat    42.4
```

A logbook also contains a stream property, which returns only the yet
unprinted entries.

```text
>>> print(logbook.stream)
gen   avg      spam
0     50.2
>>> logbook.record(gen=1, nevals=15, **record)
>>> print(logbook.stream)
1     50.2
```

### Dealing with Multi-statistics

Logbooks are able to cope with the return type of MultiStatistics objects
by logging the data in chapters for each sub dictionary contained in the
record. Therefore, a multi-record can be used exactly like a single-record.

```python
logbook = tools.Logbook()
logbook.record(gen=0, nevals=30, **record)
```

One difference is in the column ordering, where we can specify the order
of the chapters and their content as follows:

```python
logbook.header = "gen", "evals", "fitness", "size"
logbook.chapters["fitness"].header = "min", "avg", "max"
logbook.chapters["size"].header = "min", "avg", "max"
```

```text
>>> print(logbook)
                     fitness                 size
            -------------------------  ---------------
gen   nevals  min       avg      max      min   avg   max
0     30      0.165572  1.71136  6.85956  3     4.54  7
```

Retrieving the data is also done through the chapters. The generations,
minimum fitness and average size are obtained in a chronological order.
If some data is not available, a None appears in the vector.

```python
gen = logbook.select("gen")
fit_mins = logbook.chapters["fitness"].select("min")
size_avgs = logbook.chapters["size"].select("avg")
```

## Hall of fame and archives

`HallOfFame` keeps the `maxsize` best individuals seen so far.
`ParetoFront` keeps the non-dominated set. Both skip an individual
whose fitness is missing, invalid, or non-finite. Pass either as
`hof=` to a builtin algorithm. `HallOfFame.to_json` /
`from_json` round-trip `maxsize` and members as `genes` plus
`fitness` values; pair with `Checkpoint(..., hof_ind_cls=)` to
persist the archive as JSON instead of dill.

```python
hof = tools.HallOfFame(maxsize=1)
front = tools.ParetoFront()
```

MAP-Elites archives sit next to that pair. `GridArchive` bins a
caller-supplied behavior descriptor. `CvtArchive` and
`UnstructuredArchive` are the centroid and nearest-neighbor
variants. `ea_map_elites` is the matching loop. Register
`tools.sel_novelty` for parent selection by archive distance and
`tools.mut_iso_line` for iso+line variation toward a donor elite;
fitness stays on `ind.fitness`. See the
[MAP-Elites example](../examples/genetic_algorithms/map_elites.md).

`History` records a NetworkX-compatible genealogy. Call
`history.update` on the seed population and after each variation,
or wrap `mate` / `mutate` with `toolbox.decorate`. `update` records
every member of a batch:

```python
history = tools.History()
history.update(pop)
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)
# ... run ea_simple or a custom loop ...
tree = history.get_genealogy(hof[0])
print(len(history.genealogy_tree), len(tree))
```

A complete OneMax run is the
[genealogy example](../examples/genetic_algorithms/history.md).
