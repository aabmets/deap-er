from deap_er import creator
from deap_er import tools
from deap_er import base
import itertools
import random
import array
import numpy
import math


# Disable randomization to guarantee reproducibility
random.seed(1234)

# Define constants, objects and functions.
REG_POP_SIZE = 4
RAND_POP_SIZE = 2
TOTAL_POP_SIZE = 6

NDIMS = 5
NPOPS = 10
CR = 0.6
F = 0.4

SCENARIO = tools.MPConfigs.ALT1
BOUNDS = [SCENARIO["min_coord"], SCENARIO["max_coord"]]
MPB = tools.MovingPeaks(dimensions=NDIMS, **SCENARIO)

AVG_OE_MEASURE_INTERVAL = 100
AVG_OE_THRESHOLD = 3
VERBOSE = True


def brown_ind(iter_, best, sigma):
    return iter_(random.gauss(x, sigma) for x in best)


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, BOUNDS[0], BOUNDS[1])
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_float, NDIMS)
    toolbox.register("brownian_individual", brown_ind, creator.Individual, sigma=0.3)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("select", random.sample, k=4)
    toolbox.register("best", tools.sel_best, sel_count=1)
    toolbox.register("evaluate", MPB)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "error", "offline_error", "avg", "max"

    return toolbox, stats, logbook


def stop_condition(logbook):
    interval = AVG_OE_MEASURE_INTERVAL
    if len(logbook) >= 5e+5:
        raise RuntimeError('Evolution failed to converge.')
    elif len(logbook) % interval == 0:
        err_sum = 0
        for i in range(interval, 0, -1):
            val = logbook.select("offline_error")[-i]
            err_sum += val
        avg_err = err_sum / interval
        if avg_err <= AVG_OE_THRESHOLD:
            print_results(avg_err)
            return 1
    return 0


def print_results(avg_err):
    print(f'\nAverage offline error: {avg_err:.3f} (<={AVG_OE_THRESHOLD}).')
    print(f'\nEvolution converged correctly.')


class Logger:
    def __init__(self, logbook, stats):
        self.logbook = logbook
        self.stats = stats

    def log(self, ngen, pops):
        chain = itertools.chain(*pops)
        record = self.stats.compile(chain)
        args = dict(
            gen=ngen,
            evals=MPB.nevals,
            error=MPB.current_error,
            offline_error=MPB.offline_error
        )
        self.logbook.record(**args, **record)
        if VERBOSE:
            print(self.logbook.stream)


def get_best_and_invalidate(toolbox, populations) -> list:
    bests = [toolbox.best(subpop)[0] for subpop in populations]
    if any(b.fitness.values != toolbox.evaluate(b) for b in bests):
        for individual in itertools.chain(*populations):
            del individual.fitness.values
    return bests


def exclude_best_inds(toolbox, bests, populations) -> None:
    rex_cl = (BOUNDS[1] - BOUNDS[0]) / (2 * NPOPS ** (1.0 / NDIMS))
    for i, j in itertools.combinations(range(NPOPS), 2):
        if bests[i].fitness.is_valid() and bests[j].fitness.is_valid():
            d = sum((bests[i][k] - bests[j][k]) ** 2 for k in range(NDIMS))
            d = math.sqrt(d)
            if d < rex_cl:
                k = i if bests[i].fitness < bests[j].fitness else j
                populations[k] = toolbox.population(size=TOTAL_POP_SIZE)


def regular_diff_evo(toolbox, subpop, xbest, new_pop) -> None:
    for individual in subpop[:REG_POP_SIZE]:
        x1, x2, x3, x4 = toolbox.select(subpop)
        offspring = toolbox.clone(individual)
        index = random.randrange(NDIMS)
        for i, value in enumerate(individual):
            if i == index or random.random() < CR:
                offspring[i] = xbest[i] + F * (x1[i] + x2[i] - x3[i] - x4[i])
        offspring.fitness.values = toolbox.evaluate(offspring)
        if offspring.fitness >= individual.fitness:
            new_pop.append(offspring)
        else:
            new_pop.append(individual)


def brownian_diff_evo(toolbox, xbest, new_pop) -> None:
    inds = []
    for _ in range(RAND_POP_SIZE):
        ind = toolbox.brownian_individual(xbest)
        inds.append(ind)
    new_pop.extend(inds)


def main():
    toolbox, stats, logbook = setup()
    logger = Logger(logbook, stats)

    # Generate the initial populations.
    populations = [toolbox.population(size=TOTAL_POP_SIZE) for _ in range(NPOPS)]

    # Evaluate the initial populations.
    for idx, subpop in enumerate(populations):
        fitness = toolbox.map(toolbox.evaluate, subpop)
        for ind, fit in zip(subpop, fitness):
            ind.fitness.values = fit

    logger.log(0, populations)

    generation = 1

    # Define the main evolution loop.
    while not stop_condition(logbook):

        # Detect changes and invalidate fitness if necessary.
        bests = get_best_and_invalidate(toolbox, populations)

        # Apply exclusionary pressure to the best individuals.
        exclude_best_inds(toolbox, bests, populations)

        # Evaluate the individuals with an invalid fitness.
        chain = itertools.chain(*populations)
        invalid_ind = [ind for ind in chain if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        logger.log(generation, populations)

        # Evolve the subpopulations.
        for idx, subpop in enumerate(populations):
            new_pop = []
            xbest, = toolbox.best(subpop)

            # Apply regular DE to the first part of the population.
            regular_diff_evo(toolbox, subpop, xbest, new_pop)

            # Apply brownian DE to the last part of the population.
            brownian_diff_evo(toolbox, xbest, new_pop)

            # Evaluate the brownian individuals.
            for individual in new_pop[-RAND_POP_SIZE:]:
                individual.fitness.value = toolbox.evaluate(individual)

            # Replace the population with the new one.
            populations[idx] = new_pop

        # Update iteration counter.
        generation += 1


if __name__ == "__main__":
    main()
