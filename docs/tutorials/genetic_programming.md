# Genetic Programming

Genetic programming evolves programs. The usual representation is a
syntax tree: internal nodes are primitives (functions), and leaves are
terminals (constants and arguments). The [gp](../reference/gp.md)
package builds those trees, compiles them, and varies them.

This page covers loosely typed and strongly typed primitive sets,
ephemeral constants, compilation, and bloat control. Full loops live
in the [symbolic regression](../examples/genetic_programming/symb_regr.md),
[even-parity](../examples/genetic_programming/even_parity.md), and
[artificial ant](../examples/genetic_programming/art_ant.md) examples.
Strongly typed search is the
[2-4 multiplexer](../examples/genetic_programming/multiplexer.md) and
[typed classify](../examples/genetic_programming/typed_classify.md)
examples. Columnar and Push programs have their own tutorials.

## Loosely typed GP

A loosely typed set does not enforce types between nodes. Any primitive
may take any other primitive or terminal as an argument.

```python
import operator
from deap_er import gp

pset = gp.PrimitiveSet("main", 2)
pset.add_primitive(max, 2)
pset.add_primitive(operator.add, 2)
pset.add_primitive(operator.mul, 2)
pset.add_terminal(3)
pset.rename_arguments(ARG0="x", ARG1="y")
```

`PrimitiveSet` takes the procedure name and the number of inputs.
Default argument names are `ARG0`, `ARG1`, … — rename them before you
compile. A unary primitive is an arity of 1:

```python
pset.add_primitive(operator.neg, 1)
```

`gen_full`, `gen_grow`, and `gen_half_and_half` return a valid prefix
expression. Wrap that list in `PrimitiveTree`:

```python
expr = gp.gen_full(pset, min_depth=1, max_depth=3)
tree = gp.PrimitiveTree(expr)
```

## Strongly typed GP

Every primitive and terminal has argument types and a return type. The
output of one node must match the input of the next.

```python
import operator
from deap_er import gp

def if_then_else(cond, output1, output2):
    return output1 if cond else output2

pset = gp.PrimitiveSetTyped("main", [bool, float], float)
pset.add_primitive(operator.xor, [bool, bool], bool)
pset.add_primitive(operator.mul, [float, float], float)
pset.add_primitive(if_then_else, [bool, float, float], float)
pset.add_terminal(3.0, float)
pset.add_terminal(True, bool)
pset.rename_arguments(ARG0="x", ARG1="y")
```

Generation still respects the type table. If a primitive asks for a
type that no terminal can supply, `generate` raises `IndexError`.

## Ephemeral constants

An ephemeral is a terminal whose value is drawn when the node is
inserted, then stays fixed until another ephemeral replaces it.

```python
from deap_er import gp, tools

pset = gp.PrimitiveSet("main", 1)
pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
```

On a typed set, pass the return type as the third argument:

```python
pset.add_ephemeral_constant("randi", lambda: tools.rng.randint(-10, 10), int)
```

## Individuals and compilation

Trees are not yet individuals. Combine the creator and the toolbox:

```python
from deap_er import Fitness, Toolbox, creator, gp, tools

creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = Toolbox()
toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile_tree, prim_set=pset)
```

`gp.register_gp(toolbox, pset, individual=creator.Individual)`
registers that block plus `clone_individual`, one-point crossover,
uniform mutation, a height `static_limit`, and tournament selection.
It does not register `evaluate`.

```python
ind = toolbox.individual()
print(ind)                 # prefix expression as text
func = toolbox.compile(expr=ind)
func(1.0)                  # one input, because the set has arity 1
```

`str(tree)` is readable Python. `compile_tree` turns that string (or
the tree) into a callable. A program with no inputs compiles the same
way — see the artificial ant example.

`gp.build_tree_graph(expr)` returns `(nodes, edges, labels)` for a
NetworkX-style plot. This library does not pull in a graph drawer:

```python
nodes, edges, labels = gp.build_tree_graph(ind)
print(len(nodes), len(edges), labels[0])
```

## Bloat control

Python's parser stack limits tree depth (usually around 90). Deep
trees also stagnate search. Decorate variation with `static_limit` so
an oversized child is replaced by a parent:

```python
import operator

toolbox.register("mate", gp.cx_one_point)
toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))
```

`register_gp` applies that decoration by default (`height_limit=17`).
`HARM-GP` (`gp.harm`) and SlimGP are other size-control paths — see
the symbolic regression example variants.

## How to evolve programs

Register `evaluate`, `select`, `mate`, and `mutate`, then call
`ea_simple` or `harm`. The examples under
[Genetic Programming](../examples/genetic_programming/symb_regr.md)
are complete recipes. Semantic variation (`cx_semantic`,
`mut_semantic`) is the
[semantic GP example](../examples/genetic_programming/semantic_gp.md).
