Logging Statistics
==================


The Statistics Object
+++++++++++++++++++++

Often, there is a need to understand what is going on in the optimization process. This can be
achieved with the :class:`~deap_er.records.statistics.Statistics` object, which is able to compile
statistics on arbitrary attributes of any designated objects. The statistics object constructor takes
a single argument of a callable, which should return the data that the statistics will be computed on.
Usually, the statistics are computed on the fitness values of individuals, as in the following example:

.. code-block:: python

    # using a func

    def source(ind):
        return ind.fitness.values

    stats = tools.Statistics(key=source)

    # using a lambda equivalent

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

After the statistics object has been created, it is necessary to register the needed statistical functions
into the **stats** object. The register function expects a name alias as the first argument and for the
second argument it expects a function which is able to operate on a vector of values. Any subsequent
arguments are passed to this function when it's called. Usually, numpy statistical functions are used:

.. code-block:: python

    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

.. raw:: html

   <br />


Builtin Algorithms
------------------

When using :ref:`builtin algorithms <algos>`, a statistics object can be given as an argument to the function.
During the optimization process, statistics will be automatically computed on the population every generation.
If the verbose argument is **True**, the statistics are also printed to the console. At the end of the optimization
process, the builtin algorithms return the final population and the logbook.

.. code-block:: python
   :emphasize-lines: 7

    pop, log = algorithms.eaSimple(
        toolbox=toolbox,
        population=pop,
        generations=100,
        cx_prob=0.5,
        mut_prob=0.2,
        stats=stats,
        verbose=True
    )

.. raw:: html

   <br />


Custom Algorithms
-----------------

When writing custom evolution algorithms, any statistics can be obtained by compiling it on the desired objects.
The argument to the compile function must be an iterable of elements on which the key function will be called.
In the next example, we will compile statistics on a population of individuals. The statistics object will call
the key function on every individual to retrieve their ``fitness.values`` attribute. The resulting array of values
is finally given to each registered statistical function and the results are put into the record dictionary under
the key associated with the function. Usually, the records are computed on each generation of individuals and the
resulting records are inserted into the logbook, which we will learn to do in the Logbook chapter of this tutorial.

.. code-block:: python

    # inside the main evolution loop

    record = stats.compile(population)

.. code-block:: text

    >>> print(record)
    {
        'std': 4.96,
        'max': 63.0,
        'avg': 50.2,
        'min': 39.0
    }

.. raw:: html

   <br />


Multi-Objective Statistics
--------------------------

When statistics are computed directly on the values with numpy functions, all the objectives are
combined together, which is the default behaviour of numpy. In order to separate the objectives,
an axis of operation needs to be specified. This is done by providing the axis as an additional
argument to the registration function. The axis can also be specified for single-objective
statistics, which causes the output to be in the format of numpy arrays.

.. code-block:: python

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    record = stats.compile(population)

.. code-block:: text

    >>> print(record)
    {
        'std': array([4.96]),
        'max': array([63.0]),
        'avg': array([50.2]),
        'min': array([39.0])
    }

.. raw:: html

   <br />


Multiple Statistics
-------------------

It is also possible to compute statistics on multiple attributes of an individual by using a
:class:`~deap_er.records.statistics.MultiStatistics` object. For instance, it is quite common in
genetic programming to compile statistics on the height of the trees as well as their fitness values.
In order to achieve this, we define multiple regular statistics objects with the required key functions
and then pass them into a :class:`~deap_er.records.statistics.MultiStatistics` constructor.
The statistical functions can be registered only once either in the multi-statistics object or
individually in each statistics objects. The multi-statistics object can be then given to a builtin
algorithm or they can be compiled using the exact same procedure as for the simple statistics objects.
A compiled multi-statistics record is a dictionary of statistics objects.

.. code-block:: python

    stats_size = tools.Statistics(key=len)
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    record = mstats.compile(population)

.. code-block:: text

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

.. raw:: html

   <br />
   <hr>


The Logbook Object
++++++++++++++++++

The logbook is intended to be a chronological sequence of dictionaries. It is directly compliant with,
but not limited to, the type of data returned by statistics and multi-statistics objects.
Anything can be incorporated into an entry of a logbook.

Logging Data
------------

Once a record has been compiled by a statistics object, it can be recorded into a Logbook by unpacking
the record dictionary into the :func:`record()` method. The :func:`record()` method takes a variable number
of arguments, each of which is data to be recorded. The records of a logbook can be later retrieved with the
:func:`select()` method, which accepts a variable number of string arguments of either the keywords that the
data was recorded under or name aliases of statistical functions.

.. code-block:: python

   # record creation omitted for brevity

   logbook = tools.Logbook()
   logbook.record(gen=0, nevals=30, **record)

   # later when analyzing the evolution
   gen, avg = logbook.select("gen", "avg")


.. raw:: html

   <br />


Printing to Screen
------------------

A logbook can be printed to screen or a file. Its :func:`__str__()` method returns a header of each
key inserted in the first record and then the complete logbook for each of these keys. The rows are in
chronological order of insertion while the columns are in an undefined order, which can be made specific
by setting the header attribute of the Logbook object to a list of strings with the column names.

.. code-block::

    logbook.header = "gen", "food", "avg"

If an entry is missing from a row, the space in the column will be left blank for that entry.

.. code-block::

    >>> print(logbook)
    gen   food    avg
    0     apple   50.2
    1             45.9
    3     meat    42.4


A logbook also contains a stream property, which returns only the yet unprinted entries.

.. code-block::

    >>> print(logbook.stream)
    gen   avg      spam
    0     50.2
    >>> logbook.record(gen=1, nevals=15, **record)
    >>> print(logbook.stream)
    1     50.2

.. raw:: html

   <br />


Dealing with Multi-statistics
-----------------------------

Logbooks are able to cope with the return type of MultiStatistics objects by logging
the data in chapters for each sub dictionary contained in the record. Therefore, a
multi-record can be used exactly like a single-record.

.. code-block::

    logbook = tools.Logbook()
    logbook.record(gen=0, nevals=30, **record)

One difference is in the column ordering, where we can specify
the order of the chapters and their content as follows:

.. code-block::

    logbook.header = "gen", "evals", "fitness", "size"
    logbook.chapters["fitness"].header = "min", "avg", "max"
    logbook.chapters["size"].header = "min", "avg", "max"

.. code-block::

    >>> print(logbook)
                         fitness                 size
                -------------------------  ---------------
    gen   nevals  min       avg      max      min   avg   max
    0     30      0.165572  1.71136  6.85956  3     4.54  7

Retrieving the data is also done through the chapters. The generations,
minimum fitness and average size are obtained in a chronological order.
If some data is not available, a None appears in the vector.

.. code-block::

    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    size_avgs = logbook.chapters["size"].select("avg")

.. raw:: html

   <br />
