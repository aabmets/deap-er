# Using the Toolbox

The [`Toolbox`](../reference/base.md) class is the main component of computational evolution,
which is a container for all the necessary tools to build genetic algorithms and solve
evolutionary problems with. Using a toolbox enables the separation of algorithms from the
operators, which makes it easy to hot swap the operators during evolution in a plug-and-play
manner as needed.

The two most important methods of a toolbox are `register()` and `unregister()`, which are
used to add or remove tools from the toolbox. The `register()` method takes at least two
arguments: an alias and a function that is going to be associated with the alias. All
subsequent arguments, if there are any, will be implicitly passed into the associated
function when the registered method is called.

!!! attention
    Alias names must be valid Python identifiers. Registering the same
    alias again overwrites the previous callable.

```python
from deap_er import Toolbox

def add_func(a, b, c):
    return a + b + c

toolbox = Toolbox()
toolbox.register("first_alias", add_func, a=1, b=2, c=3)  # args are passed implicitly
toolbox.register("other_alias", add_func)                 # args are passed explicitly

result = toolbox.first_alias()         # result == 6
result = toolbox.other_alias(1, 2, 3)  # result == 6
```

The preceding code registers two aliases in the toolbox: **first_alias** and **other_alias**,
which point to the same function **add_func**. In the case of `toolbox.first_alias()`, the
arguments were already passed during registration, so the method needs to be called without
arguments. In the case of `toolbox.other_alias(1, 2, 3)`, the arguments must be passed
explicitly, because the method was registered without arguments. The difference between
the two is that the arguments of **first_alias** are static, while the arguments of
**other_alias** are dynamic.

## Tool Registration

A toolbox is only as useful as the tools registered into it. The **tools** module contains
a number of useful tools that can be registered into a toolbox. While all tools in the
module can be registered into a toolbox, some of them such as algorithms or statistics,
are equally useful independently. More on tools in the
[Operators and Algorithms](operators_and_algorithms.md) chapter.

Aliases are attached at runtime, so type checkers do not see
`toolbox.mate` until you register it. A new toolbox already has
`clone` (`copy.deepcopy`) and `map` (`map`). For list or
`array.array` individuals, register `tools.clone_individual`
instead — it copies the genes and the fitness without a full
deepcopy. Register `evaluate_batch` when a generation should be
scored in one call; the builtin loops use it in place of
`map` + `evaluate`. See
[Operators and Algorithms](operators_and_algorithms.md) and
[Multiprocessing](multiprocessing.md).

```python
from deap_er import Toolbox, tools

toolbox = Toolbox()
toolbox.register("clone", tools.clone_individual)
toolbox.register("mate", tools.cx_two_point)
toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.2)
toolbox.register("select", tools.sel_tournament, contestants=3)
toolbox.register("evaluate", tools.bm_sphere)

pop, log = tools.ea_simple(**args)
```

## Tool Decoration

Tool decoration is a powerful feature that allows the precise control of parameters during
the evolution process. For example, in the case of constrained domains, a tool decorator
can be used on mutation and crossover operators to prevent the individuals from growing
out-of-bounds.

The following example defines a decorator that checks if the solution values of offsprings
are out-of-bounds and clamps them to the predefined limit values if this is the case.
Whenever either of the decorated tools is called, bounds will be checked on the resulting
offsprings.

!!! note
    This decorator works for both crossover and mutation operators, because the return
    type of these operators must be a **tuple**.

```python
def clamp(min, max):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            offsprings: tuple = func(*args, **kwargs)
            for child in offsprings:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offsprings
        return wrapped
    return wrapper

toolbox.register("mate", tools.cx_blend, alpha=0.2)
toolbox.register("mutate", tools.mut_gaussian,
    mu=0, sigma=2, mut_prob=0.2
)
toolbox.decorate("mate", clamp(MIN, MAX))
toolbox.decorate("mutate", clamp(MIN, MAX))
```

For Gaussian mutation alone, `tools.mut_gaussian_bounded` clamps each
mutated gene into `[low, up]` inside the operator. Blend has a boxed
form too (`cx_blend_bounded`). The decorator remains the generic wrap
for any other tool.
