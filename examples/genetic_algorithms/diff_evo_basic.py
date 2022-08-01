from deap_er import utilities as utils
from deap_er import evaluators as evals
from deap_er import operators as ops
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import array
import numpy


random.seed(1234)  # disables randomization

NDIM = 10
CR = 0.25
F = 1
MU = 300
NGEN = 200


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -3, 3)
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_float, NDIM)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("select", ops.sel_random, sel_count=3)
    toolbox.register("evaluate", evals.sphere)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = records.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    return toolbox, stats, logbook


def print_results(best_ind):
    if not best_ind.fitness.values < (1e-3,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats, logbook = setup()
    pop = toolbox.population(size=MU)
    hof = records.HallOfFame(1)

    def log_stats(ngen=0):
        record = stats.compile(pop)
        logbook.record(gen=ngen, evals=len(pop), **record)
        print(logbook.stream)

    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    log_stats()

    for gen in range(1, NGEN):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)
        log_stats(gen)

    print_results(hof[0])


if __name__ == "__main__":
    main()
