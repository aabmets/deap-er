from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
from deap_er import gp
import operator
import random
import numpy
import math


random.seed(1234)  # disables randomization


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, points, toolbox):
    func = toolbox.compile(expr=individual)
    sq_errors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    result = math.fsum(sq_errors) / len(points)
    return result,  # The comma is essential here.


def setup():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_primitive(operator.neg, 1)
    pset.add_primitive(math.cos, 1)
    pset.add_primitive(math.sin, 1)
    pset.add_ephemeral_constant("rand101", lambda: random.randint(-1, 1))
    pset.rename_arguments(ARG0='x')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", utils.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate, points=[x / 10. for x in range(-10, 10)], toolbox=toolbox)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))

    stats_size = records.Statistics(len)
    stats_fit = records.Statistics(lambda ind: ind.fitness.values)
    mstats = records.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    return toolbox, mstats


def print_results(best_ind):
    if not best_ind.fitness.values < (1.0e-3,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, mstats = setup()
    pop = toolbox.population(size=300)
    hof = records.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=40,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=mstats,
        verbose=True  # prints stats
    )
    algos.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
