from deap_er import creator
from deap_er import tools
from deap_er import base
from deap_er import gp
import operator
import random
import numpy
import math


random.seed(1234)  # disables randomization

CX_PROB = 0.5
MUT_PROB = 0.2
GENS = 40


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox, points):
    func = toolbox.compile(expr=individual)
    sq_errors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    result = math.fsum(sq_errors) / len(points)
    return result,  # The comma is essential here.


def add_primitives(prim_set):
    prim_set.add_primitive(operator.add, 2)
    prim_set.add_primitive(operator.sub, 2)
    prim_set.add_primitive(operator.mul, 2)
    prim_set.add_primitive(safe_div, 2)
    prim_set.add_primitive(operator.neg, 1)
    prim_set.add_primitive(math.cos, 1)
    prim_set.add_primitive(math.sin, 1)


def setup():
    adf_set_2 = gp.PrimitiveSet("ADF2", 2)
    add_primitives(adf_set_2)

    adf_set_1 = gp.PrimitiveSet("ADF1", 2)
    add_primitives(adf_set_1)
    adf_set_1.add_adf(adf_set_2)

    adf_set_0 = gp.PrimitiveSet("ADF0", 2)
    add_primitives(adf_set_0)
    adf_set_0.add_adf(adf_set_1)
    adf_set_0.add_adf(adf_set_2)

    pset = gp.PrimitiveSet("MAIN", 1)
    add_primitives(pset)
    pset.add_ephemeral_constant("rand101", lambda: random.randint(-1, 1))
    pset.add_adf(adf_set_0)
    pset.add_adf(adf_set_1)
    pset.add_adf(adf_set_2)
    pset.rename_arguments(ARG0='x')

    prim_sets = (pset, adf_set_0, adf_set_1, adf_set_2)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Tree", gp.PrimitiveTree)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('adf_expr_0', gp.gen_full, prim_set=adf_set_0, min_depth=1, max_depth=2)
    toolbox.register('adf_expr_1', gp.gen_full, prim_set=adf_set_1, min_depth=1, max_depth=2)
    toolbox.register('adf_expr_2', gp.gen_full, prim_set=adf_set_2, min_depth=1, max_depth=2)
    toolbox.register('main_expr', gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)

    toolbox.register('ADF0', tools.init_iterate, creator.Tree, toolbox.adf_expr_0)
    toolbox.register('ADF1', tools.init_iterate, creator.Tree, toolbox.adf_expr_1)
    toolbox.register('ADF2', tools.init_iterate, creator.Tree, toolbox.adf_expr_2)
    toolbox.register('MAIN', tools.init_iterate, creator.Tree, toolbox.main_expr)

    func_cycle = [toolbox.MAIN, toolbox.ADF0, toolbox.ADF1, toolbox.ADF2]

    toolbox.register('individual', tools.init_cycle, creator.Individual, func_cycle)
    toolbox.register('population', tools.init_repeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile_adf_tree, prim_sets=prim_sets)
    toolbox.register('mate', gp.cx_one_point)
    toolbox.register('expr', gp.gen_full, min_depth=1, max_depth=2)
    toolbox.register('mutate', gp.mut_uniform, expr=toolbox.expr)
    toolbox.register('select', tools.sel_tournament, contestants=3)
    toolbox.register('evaluate', evaluate, toolbox=toolbox, points=[x / 10. for x in range(-10, 10)])
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=17))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    return toolbox, stats, logbook, prim_sets


def print_results(best_ind):
    if not best_ind.fitness.values < (0.5,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats, logbook, psets = setup()
    hof = tools.HallOfFame(1)

    def log_stats(ngen=0):
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=ngen, evals=len(pop), **record)
        print(logbook.stream)

    pop = toolbox.population(size=100)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    log_stats()

    for generations in range(1, GENS):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            for tree1, tree2 in zip(ind1, ind2):
                if random.random() < CX_PROB:
                    toolbox.mate(tree1, tree2)
                    del ind1.fitness.values
                    del ind2.fitness.values

        for ind in offspring:
            for tree, pset in zip(ind, psets):
                if random.random() < MUT_PROB:
                    toolbox.mutate(individual=tree, prim_set=pset)
                    del ind.fitness.values

        invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = offspring

        log_stats(generations)

    print_results(hof[0])


if __name__ == "__main__":
    main()
