import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = [x**4 + x**3 + x**2 + x for x in POINTS]
CASE_COUNT = 8


def two():
    return 2


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - y) ** 2 for x, y in zip(POINTS, TARGET, strict=True))


def select(individuals, sel_count):
    # Rebuild the case subset every generation. Do not freeze cases= on the toolbox.
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.sample_informed_cases(individuals, CASE_COUNT, matrix=matrix)
    # Use tools.sel_lexicase(...) for the strict filter.
    return tools.sel_epsilon_lexicase(individuals, sel_count, cases=cases, matrix=matrix)


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2, weight=2.0)
    pset.add_primitive(operator.sub, 2, weight=2.0)
    pset.add_primitive(operator.mul, 2, weight=2.0)
    pset.add_primitive(safe_div, 2, weight=0.5)
    pset.add_primitive(operator.neg, 1, weight=0.5)
    pset.add_primitive(math.cos, 1, weight=0.25)
    pset.add_primitive(math.sin, 1, weight=0.25)
    pset.add_terminal(two, call_zero=True)
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * len(POINTS))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("clone", tools.clone_individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("select", select)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))

    stats = tools.Statistics(lambda ind: float(numpy.mean(ind.fitness.values)))
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    return toolbox, stats


def print_results(best_ind):
    mse = float(numpy.mean(best_ind.fitness.values))
    print(f"\nBest program: {best_ind}")
    print(f"Infix: {gp.tree_to_infix(best_ind)}")
    print(f"Mean squared error: {mse:.6g}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=120)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=15,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
