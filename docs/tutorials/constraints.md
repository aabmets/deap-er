# Constraint Handling

Evolutionary algorithms are unconstrained by default. This page shows
how to add feasibility to a run: a penalty on `evaluate`, and Deb
constraint-dominance on NSGA-II. The walkthrough follows Coello
Coello's survey of penalty methods. A complete script is the
[constraints example](../examples/genetic_algorithms/constraints.md).

## Penalty function

A penalty gives an infeasible individual a worse fitness, constant or
growing with distance to the valid region. For a scalar $x$ constrained
to $3 < x < 7$, the assigned fitness is $g(x)$ inside the interval and
$h(x)$ outside. $h(x)$ may be a constant $\Delta$, $\Delta$ plus
Euclidean distance to a feasible point, or $\Delta$ plus a quadratic
distance.

`DeltaPenalty` wraps any evaluation function. The first argument is a
feasibility predicate. The second is $\Delta$. The optional third is a
distance to the valid region:

```python
from math import sin
from deap_er import Toolbox, tools

def evaluate(individual):
    x = individual[0]
    return ((x - 5) ** 2 * sin(x) * (x / 3),)

def feasible(individual):
    return 3 < individual[0] < 7

def distance(individual):
    return (individual[0] - 5.0) ** 2

toolbox = Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 7.0, distance))
```

Valid individuals keep $g(x)$. Invalid ones receive $\Delta$ adjusted
by the distance term (the sign follows `fitness.weights`).

`ClosestValidPenalty` is the other decorator: an invalid individual
is scored as the closest feasible individual plus $\alpha$ times a
distance. See the [MO-CMA example](../examples/evolution_strategies/mo_cma_strat.md)
and the [utilities](../reference/utilities.md) reference.

## Constraint-dominance

Deb's rule compares feasibility first, then violation, then ordinary
Pareto dominance. `constraint_dominates` is that comparison.
`sel_nsga_2` accepts optional `feasible=` and `violation=` callables
and ranks the pool with that order:

```python
def violation(individual):
    return max(0.0, individual[0] ** 2 + individual[1] ** 2 - 1.0)

toolbox.register(
    "select",
    tools.sel_nsga_2,
    feasible=lambda ind: violation(ind) == 0.0,
    violation=violation,
)
```

Use this when the run is already multi-objective and you want
infeasible points to lose to feasible ones before crowding. Pair it
with a penalty if `evaluate` must still return a number for an
invalid genome, or return the raw objectives and let selection
enforce the constraint.

See the [Operators](../reference/operators.md) reference for the
full `feasible` / `violation` contract.
