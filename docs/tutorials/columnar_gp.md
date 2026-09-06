# Columnar Programs

The [genetic programming](../reference/gp.md) package can evolve programs
over a whole table of numeric columns instead of one scalar at a time. A
compiled tree takes one array per column and returns one array, so a
population of a few hundred programs can be scored against hundreds of
thousands of samples without a Python loop over rows.

This tutorial covers the columnar primitive set, the vectorized and
window primitive kits, the three evaluation backends, and the rules you
have to respect to keep the results honest.

## The data contract

Convert your data **once**, before the evolution starts, into one
`float64` array per column:

- one-dimensional, C-contiguous, and all of the same length
- missing values as `numpy.nan`, not as zero or a sentinel
- integer counts promoted to `float64`
- identifiers, labels, and timestamps left out of the primitive set

Anything else — loading, joining, resampling, and the fitness function
itself — stays on your side. The library only owns the step from a tree
to an array.

## Building the primitive set

`make_column_pset` creates a strongly typed set with one input per
column. Argument order *is* column order, so `pset.arguments[i]` names
the array you pass in position `i`.

```python
import numpy
from deap_er import gp

names = ["level", "flow", "noise"]

pset = gp.make_column_pset(names)
gp.add_numpy_primitives(pset)
gp.add_window_primitives(pset)
gp.add_window_ephemeral(pset, "window", 2, 64)
```

Column names have to be plain Python identifiers, because a compiled
tree becomes a lambda over them. Keywords, duplicates, and names that
look like the default `ARG0` prefix are rejected.

!!! attention
    A column may not share a name with a primitive. `level` is fine,
    `vadd` is not: the lambda parameter would shadow the operator and
    the tree would quietly compute the wrong thing. Both
    `add_numpy_primitives` and `add_window_primitives` raise instead of
    letting that happen.

### Type tags

Three tag classes drive the strongly typed search. They are markers for
the type system only — the values that flow at runtime are ordinary
NumPy arrays and Python integers.

| Tag | Runtime value | Role |
|:----|:--------------|:-----|
| `Array` | 1D `float64` | a column, or anything derived from one |
| `Mask` | 1D `bool` | a condition |
| `Window` | `int` | a rolling length or a delay |

Keeping `Mask` apart from `Array` is what makes `vwhere` well defined
and stops a condition from being added to a series. No primitive returns
a `Window`, so a window is always a leaf.

## The vectorized kit

`add_numpy_primitives` registers arithmetic (`vadd`, `vsub`, `vmul`,
`vneg`, `vabs`, `vsin`, `vcos`), protected operations (`vdiv`, `vlog`,
`vsqrt`), comparisons (`vgt`, `vlt`, `vge`, `vle`, `veq`), mask logic
(`vand`, `vor`, `vnot`), and `vwhere`.

`vwhere` is the important one: it is how a program turns a condition
into a value without a Python `if`, which is what keeps the whole tree
in vectorized code.

The protected operations replace a non-finite result with a fill, by
default `1.0`:

```python
gp.vdiv(numpy.array([1.0]), numpy.array([0.0]))  # -> array([1.])
gp.vdiv(numpy.array([1.0]), numpy.array([0.0]), fill=0.0)  # -> array([0.])
```

They only replace a value that the operation itself fabricated. A
`nan` that arrived in an operand comes back out as `nan`, so the warmup
written by the window primitives is never overwritten by a made-up
number.

## Causal window primitives

`add_window_primitives` registers `delay`, `diff`, `rolling_sum`,
`rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, and `ema`.
Each takes an `Array` and a `Window`.

Every one of them is **causal**: a sample of the output depends on that
sample and earlier ones, never on a later one. The rolling window is
`[t - n + 1, t]` inclusive.

```python
values = numpy.arange(10, dtype=numpy.float64)

gp.delay(values, 2)         # [nan nan 0. 1. 2. 3. 4. 5. 6. 7.]
gp.rolling_mean(values, 3)  # [nan nan 1. 2. 3. 4. 5. 6. 7. 8.]
```

Samples without enough history are `nan`, never zero. Zero would be a
value your fitness function could act on, and it would be a value the
program never actually had.

Window lengths are ephemeral constants, sampled once per tree:

```python
gp.add_window_ephemeral(pset, "window", 2, 64)
```

The bounds are inclusive. The name must be unique across the process,
because a primitive set stores ephemeral types by name.

`add_pair_window_primitives` is an optional second kit for two series
and one `Window`: `rolling_corr`, `rolling_cov`, and `rolling_beta`.
Moments use the same population divisor as `rolling_std`. Beta is the
OLS slope of the first series on the second
($\mathrm{cov}(x, y) / \mathrm{var}(y)$). A window whose denominator
variance is zero is `nan`, not the protected-op fill.

```python
gp.add_pair_window_primitives(pset)
gp.rolling_beta(level, flow, 4)
```

!!! attention
    A comparison hides the warmup. `numpy.nan > x` is `False`, so
    `vwhere(vgt(rolling_mean(level, 20), flow), a, b)` returns a real
    value at sample zero even though the average has not warmed up
    yet. Decide in the fitness function which prefix of the result to
    trust — for example by ignoring samples where the target or the
    prediction is not finite — instead of assuming the `nan` propagates
    all the way out.

## Evolving

Register `clone_individual` rather than leaving the default `deepcopy`.
Tree nodes are immutable once created, so an offspring only needs a new
list and a fresh fitness:

```python
from deap_er import Toolbox, gp, tools

toolbox = Toolbox()
toolbox.register("clone", tools.clone_individual)
toolbox.register("compile", gp.compile_tree, prim_set=pset)
```

A wide table has many more terminals than primitives, which pushes
`gen_grow` towards small trees. Reach for `gen_full`, or raise
`min_depth`, when the programs come out shorter than you want.

## Backends

`compile_tree` takes a `backend` argument. All three read the same
columns in the same order and agree on the same results.

| Backend | What it does |
|:--------|:-------------|
| `python` | evaluates the expression text. The default, and the reference |
| `opcode` | lowers the tree to a flat tape and runs it on a NumPy stack machine |
| `numba` | runs the same tape through a compiled interpreter, from the `numba` extra |

```python
func = gp.compile_tree(tree, pset)                     # python
func = gp.compile_tree(tree, pset, backend="opcode")
func = gp.compile_tree(tree, pset, backend="numba")

result = func(level, flow, noise)
```

The tape backends reject a tree while lowering it if any of its
primitives has no opcode, so an unsupported operator is an error before
the first sample is touched. Both accept the columns positionally; the
Numba backend also accepts a single prepacked `(n_rows, n_columns)`
matrix, which avoids repacking them on every call.

The Numba interpreter is compiled once per process, never once per
tree, and it is deliberately single threaded. Every compiled tape
shares one process-wide workspace, so parallelize across individuals
with `toolbox.map`, as described in the
[multiprocessing tutorial](multiprocessing.md), rather than by calling
compiled programs from several threads.

It also has no way to size a result for a primitive set that takes no
arguments, since every value it holds is one column long. Use the
`python` or `opcode` backend for a constant-only set.

!!! note
    A tree that is nothing but a column returns that column itself
    rather than a copy, and a tree that is nothing but a constant
    returns a scalar. Do not write into a result you did not build.

### Custom compiled kernels

A consumer can add its own compiled kernels without forking the
interpreter. Register the primitive as usual so that the default
backend and the type checks keep working, bind its name to an opcode at
or above `USER_BASE`, and pass one compiled dispatcher that implements
those opcodes.

```python
import numba

OP_SMOOTH = gp.USER_BASE + 1

pset.add_primitive(smooth, [gp.Array], gp.Array, "smooth")
gp.bind_numba_opcode("smooth", OP_SMOOTH)

@numba.njit(error_model="numpy")
def dispatch(op, sp, stack, columns, constants, scratch):
    if op == OP_SMOOTH:
        ...
        return sp
    return -1

func = gp.compile_tree(tree, pset, backend="numba", dispatch=dispatch)
```

`gp.USER_DISPATCH_SIGNATURE` documents the stack layout the kernel has
to follow. Builtin window primitives fold their length into the
instruction, but a custom primitive receives every argument on the
stack, so a window arrives as a broadcast row. A kernel may freely
clobber `scratch` and the free row at `stack[sp]`.

Keep the two implementations honest against each other: treat the
Python primitive as the definition and compare the two backends on the
same trees. Persist the bindings from `gp.numba_opcodes()` with a run,
because a replayed tree decodes against whatever bindings exist at the
time.

!!! attention
    Fees, portfolio state, penalties, and anything else that allocates
    Python objects belong in the fitness function, not in a kernel.
    The dispatch signature only carries `float64` columns, constants,
    and scratch space.

## Limitations

- `PrimitiveTree.from_string` cannot round-trip an ephemeral of a custom
  type such as `Window`, because it parses the literal back as an `int`.
  Checkpoint the trees themselves rather than their text.
- `compile_adf_tree` runs on the default backend only.
- The Numba backend shares one workspace across every compiled tape,
  so its callables must not be run from several threads at once.

## Related

- [Columnar Programs example](../examples/genetic_programming/columnar_gp.md)
- [Multiprocessing](multiprocessing.md)
- [Genetic Programming reference](../reference/gp.md)
