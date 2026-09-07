import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization


def lf(x):
    if x > 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, points, toolbox):
    func = toolbox.compile(individual)
    sq_errors = ((func(x) - x**4 - x**3 - x**2 - x) ** 2 for x in points)
    result = math.fsum(sq_errors) / len(points)
    return (result,)


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_primitive(operator.neg, 1)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type(
        "Individual",
        gp.SlimTree,
        fitness=creator.FitnessMin,
    )

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_slim_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_slim_donor, prim_set=pset)
    toolbox.register("mutate", gp.mut_slim, prim_set=pset, inflate_prob=0.3)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register(
        "evaluate", evaluate, points=[x / 10.0 for x in range(-10, 10)], toolbox=toolbox
    )
    toolbox.register("clone", tools.clone_individual)

    stats_size = tools.Statistics(len)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    return toolbox, mstats


def print_results(best_ind, initial_fitness):
    if best_ind.fitness.values[0] >= initial_fitness:
        raise RuntimeError("Evolution failed to improve the population.")
    if best_ind.fitness.values[0] >= 0.1:
        raise RuntimeError("Evolution failed to reach the demo fitness target.")
    if len(best_ind) > 250:
        raise RuntimeError("SLIM elite grew too large for a readable model.")
    print("\nEvolution converged correctly.")


def main():
    toolbox, mstats = setup()
    pop = toolbox.population(size=300)
    hof = tools.HallOfFame(1)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    initial_fitness = min(ind.fitness.values[0] for ind in pop)
    args = {
        "toolbox": toolbox,
        "population": pop,
        "generations": 80,
        "cx_prob": 0.5,
        "mut_prob": 0.2,
        "hof": hof,
        "stats": mstats,
        "verbose": True,
    }
    gp.harm(**args)
    print_results(hof[0], initial_fitness)


if __name__ == "__main__":
    main()
