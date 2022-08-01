from deap_er import creator
from deap_er import tools
from deap_er import base
from deap_er import gp
import random
import numpy


random.seed(1234)  # disables randomization


def safe_div(left, right):
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x


def evaluate(individual, toolbox, samples, values):
    func = toolbox.compile(expr=individual)
    diff = numpy.sum((func(samples) - values)**2)
    return diff,


def setup():
    samples = numpy.linspace(-1, 1, 10000)
    values = samples ** 4 + samples ** 3 + samples ** 2 + samples

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(numpy.add, 2, name="vadd")
    pset.add_primitive(numpy.subtract, 2, name="vsub")
    pset.add_primitive(numpy.multiply, 2, name="vmul")
    pset.add_primitive(safe_div, 2)
    pset.add_primitive(numpy.negative, 1, name="vneg")
    pset.add_primitive(numpy.cos, 1, name="vcos")
    pset.add_primitive(numpy.sin, 1, name="vsin")
    pset.add_ephemeral_constant("rand101", lambda: random.randint(-1, 1))
    pset.rename_arguments(ARG0='x')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, samples=samples, values=values)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not best_ind.fitness.values < (1.0e-3,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = tools.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=40,
        cx_prob=0.5,
        mut_prob=0.1,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    tools.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
