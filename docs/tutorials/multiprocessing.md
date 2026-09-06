# Multiprocessing

In this section of the tutorial we will learn how to speed up the computation of
evolutionary algorithms by using multiprocessing. The distribution of workloads
over multiple cores or computing clusters requires the serialization of data
objects, which is usually done by pickling, therefore all objects that are to
be distributed (e.g. functions and their arguments) must be pickleable.

The correct way of using multiprocessing with **DEAP-ER** is to override the
default `map` function in the toolbox with one that supports parallel execution.
The only requirement of this `map` function is that its signature and return
type must match with the regular `map` function. This enables the use of any
third-party distributed computing libraries, such as
[Ray](https://docs.ray.io/en/latest/ray-more-libs/multiprocessing.html),
that implement the same interface.

**With context manager**

```python
# Using the multiprocessing library
with multiprocessing.Pool() as pool:
    toolbox.register('map', pool.map)
    # Execute the evolution


# Using the concurrent.futures library
with concurrent.futures.ProcessPoolExecutor() as executor:
    toolbox.register('map', executor.map)
    # Execute the evolution
```

**Without context manager**

```python
# Using the multiprocessing library
pool = multiprocessing.Pool()
toolbox.register('map', pool.map)
# Execute the evolution
pool.close()
pool.join()


# Using the concurrent.futures library
executor = concurrent.futures.ProcessPoolExecutor()
toolbox.register('map', executor.map)
# Execute the evolution
executor.shutdown()
```

!!! note
    It is also suggested to take a look at the
    [full multiprocessing example](../examples/genetic_algorithms/onemax.md#using-multiprocessing).

## Sharing a large dataset

Every item a parallel `map` hands to a worker has to be pickled. When
the evaluation reads a large array, the way that array reaches the
worker decides whether parallelism helps at all.

!!! attention
    Do not bind the dataset into the evaluation operator:

    ```python
    toolbox.register("evaluate", fitness, data=huge_array)  # anti-pattern
    ```

    The array becomes part of the registered callable, so it is pickled
    and sent again for **every individual**. On a dataset of any real
    size that cost dwarfs the evaluation itself.

Build the arrays once, before the pool starts, and let the evaluation
read them from the enclosing scope. The individual then stays the only
thing that travels:

```python
columns = load_columns()  # built before the pool exists

def evaluate(individual):
    func = toolbox.compile(expr=individual)
    return (score(func(*columns)),)

toolbox.register("evaluate", evaluate)

with multiprocessing.Pool() as pool:
    toolbox.register("map", pool.map)
    # Execute the evolution
```

On platforms that start workers by forking, which is the default on
Linux, the child inherits those pages copy-on-write and never copies the
data at all. On platforms that start workers by spawning, which is the
default on Windows and macOS, place the arrays in
[shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
or a `numpy.memmap` and have each worker attach to them once at import.

## Evaluating a whole generation at once

Some evaluations are faster when the entire generation is handed over
in a single call, for instance when the work is dispatched to a GPU or
to a vectorized kernel that amortizes its setup.

Register an `evaluate_batch` operator for that. It takes the list of
individuals whose fitness is invalid and returns their fitness values
in the same order. When it is present, `ea_simple`, `ea_mu_plus_lambda`,
`ea_mu_comma_lambda`, and `harm` call it instead of going through `map`
and `evaluate`.

```python
def evaluate_batch(individuals):
    unique = {}
    tapes = []
    index = []
    for ind in individuals:
        key = str(ind)
        slot = unique.get(key)
        if slot is None:
            unique[key] = slot = len(tapes)
            tapes.append(gp.lower_tree(ind, pset))
        index.append(slot)
    predicted = gp.interpret_tapes(tapes, matrix, backend="numba")
    return [score(predicted[i], target) for i in index]

toolbox.register("evaluate_batch", evaluate_batch)
```

Key unique programs by `str(ind)` and lower the tree object. Do not
pass the string to `lower_tree`: `PrimitiveTree.from_string` cannot
round-trip a `Window` ephemeral. The fitness list must have one
entry per individual — `evaluate_invalid` zips without `strict`.

`parallel=True` is in-process Numba threading (one workspace per
thread, each as long as the book). It does not replace a process
pool. Leave it off when the series is millions of bars, or when a
pool is already running.

!!! note
    `evaluate_batch` replaces `map` for evaluation, so a batch operator
    is responsible for its own parallelism. Leave it unregistered to
    keep the ordinary per-individual path.

!!! attention
    When using multiprocessing on Windows, the main function needs to be guarded
    with the `if __name__ == '__main__'` statement.

!!! tip
    An excellent third-party tutorial about multiprocessing in Python is available
    [here](https://superfastpython.com/multiprocessing-in-python).
    The reference manuals of
    [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) and
    [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
    should also prove useful.
