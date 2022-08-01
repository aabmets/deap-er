from deap_er import utilities as utils
from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy


random.seed(1234)  # disables randomization

INPUTS = 6
NGEN = 100
CX_PROB = 0.5
MUT_PROB = 0.2
ADD_PROB = 0.01
DEL_PROB = 0.01


def eval_network(individual, dimension):
    network = utils.SortingNetwork(dimension, individual)
    return network.evaluate(), network.length, network.depth


def gen_wire(dimension):
    wire1 = random.randrange(dimension)
    wire2 = random.randrange(dimension)
    return wire1, wire2


def gen_network(dimension, min_size, max_size):
    size = random.randint(min_size, max_size)
    network = [gen_wire(dimension) for _ in range(size)]
    return network


def mut_wire(individual, dimension, mut_prob):
    for index, elem in enumerate(individual):
        if random.random() < mut_prob:
            individual[index] = gen_wire(dimension)


def mut_add_wire(individual, dimension):
    index = random.randint(0, len(individual))
    individual.insert(index, gen_wire(dimension))


def mut_del_wire(individual):
    index = random.randrange(len(individual))
    del individual[index]


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("network", gen_network, dimension=INPUTS, min_size=9, max_size=12)
    toolbox.register("individual", utils.init_iterate, creator.Individual, toolbox.network)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_network, dimension=INPUTS)
    toolbox.register("mate", ops.cx_two_point)
    toolbox.register("mutate", mut_wire, dimension=INPUTS, mut_prob=0.05)
    toolbox.register("addwire", mut_add_wire, dimension=INPUTS)
    toolbox.register("delwire", mut_del_wire)
    toolbox.register("select", ops.sel_nsga_2)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = records.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    return toolbox, stats, logbook


def print_results(best_network):
    print('\nBest sorting network schematic:')
    print(best_network.draw())
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats, logbook = setup()
    population = toolbox.population(size=300)
    hof = records.ParetoFront()

    def log_stats(ngen=0):
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=ngen, evals=len(population), **record)
        print(logbook.stream)

    fitness = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit

    log_stats()

    for generation in range(1, NGEN):
        offspring = [toolbox.clone(ind) for ind in population]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for ind in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(ind)
                del ind.fitness.values
            if random.random() < ADD_PROB:
                toolbox.addwire(ind)
                del ind.fitness.values
            if random.random() < DEL_PROB:
                toolbox.delwire(ind)
                del ind.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        population = toolbox.select(population+offspring, len(offspring))

        log_stats(generation)

    best_network = utils.SortingNetwork(INPUTS, hof[0])
    print_results(best_network)


if __name__ == "__main__":
    main()
