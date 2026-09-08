import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]


def lf(x):
    if x > 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    result = math.fsum((func(x) - x**4 - x**3 - x**2 - x) ** 2 for x in POINTS) / len(POINTS)
    return (result,)


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2, name="add")
    pset.add_primitive(operator.sub, 2, name="sub")
    pset.add_primitive(operator.mul, 2, name="mul")
    pset.add_primitive(lf, 1, name="lf")
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_semantic, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("mutate", gp.mut_semantic, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=12))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=12))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values[0] >= 0.5:
        raise RuntimeError("Semantic GP failed to reach the seed-fixed MSE floor.")
    print(f"\nBest MSE: {best_ind.fitness.values[0]:.6f}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=12,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
