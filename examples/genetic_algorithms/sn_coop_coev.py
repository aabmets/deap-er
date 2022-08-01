from deap_er import creator
from deap_er import tools
from deap_er import base
import random
import numpy


random.seed(1234)  # disables randomization

INPUTS = 12
MAXGEN = 100
H_CX_PROB, H_MUT_PROB = 0.6, 0.3
P_CX_PROB, P_MUT_PROB = 0.6, 0.3


def gen_wire(dimension):
    wire1 = random.randrange(dimension)
    wire2 = random.randrange(dimension)
    return wire1, wire2


def gen_network(dimension, min_size, max_size):
    size = random.randint(min_size, max_size)
    network = [gen_wire(dimension) for _ in range(size)]
    return network


def eval_network(host, parasite, dimension):
    network = tools.SortingNetwork(dimension, host)
    return network.evaluate(parasite),  # The comma is essential here.


def mut_network(individual, dimension, mutpb, addpb, delpb, indpb):
    if random.random() < mutpb:
        for index, elem in enumerate(individual):
            if random.random() < indpb:
                individual[index] = gen_wire(dimension)
    if random.random() < addpb:
        index = random.randint(0, len(individual))
        individual.insert(index, gen_wire(dimension))
    if random.random() < delpb:
        index = random.randrange(len(individual))
        del individual[index]
    return individual,  # The comma is essential here.


def clone_network(individual):
    clone = individual.__class__(individual)
    clone.fitness.values = individual.fitness.values
    return clone


def gen_parasite(dimension):
    return [random.choice((0, 1)) for _ in range(dimension)]


def mut_parasite(individual, mut_prob):
    for i in individual:
        if random.random() < mut_prob:
            tools.mut_flip_bit(i, mut_prob)
    return individual,  # The comma is essential here.


def clone_parasite(individual):
    clone = individual.__class__(list(seq) for seq in individual)
    clone.fitness.values = individual.fitness.values
    return clone


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Host", list, fitness=creator.FitnessMin)
    creator.create("Parasite", list, fitness=creator.FitnessMax)

    h_toolbox = base.Toolbox()
    h_toolbox.register("host", gen_network, dimension=INPUTS, min_size=9, max_size=12)
    h_toolbox.register("individual", tools.init_iterate, creator.Host, h_toolbox.host)
    h_toolbox.register("population", tools.init_repeat, list, h_toolbox.individual)
    h_toolbox.register("evaluate", eval_network, dimension=INPUTS)
    h_toolbox.register("mate", tools.cx_two_point)
    h_toolbox.register("mutate", mut_network, dimension=INPUTS, mutpb=0.2, addpb=0.01, delpb=0.01, indpb=0.05)
    h_toolbox.register("select", tools.sel_tournament, contestants=3)
    h_toolbox.register("clone", clone_network)

    p_toolbox = base.Toolbox()
    p_toolbox.register("parasite", gen_parasite, dimension=INPUTS)
    p_toolbox.register("individual", tools.init_repeat, creator.Parasite, p_toolbox.parasite, 20)
    p_toolbox.register("population", tools.init_repeat, list, p_toolbox.individual)
    p_toolbox.register("mate", tools.cx_two_point)
    p_toolbox.register("mutate", mut_parasite, mut_prob=0.05)
    p_toolbox.register("select", tools.sel_tournament, contestants=3)
    p_toolbox.register("clone", clone_parasite)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    return h_toolbox, p_toolbox, stats, logbook


def print_results(best_network):
    print('\nBest sorting network schematic:')
    print(best_network.draw())
    print(f'\nEvolution converged correctly.')


def main():
    h_toolbox, p_toolbox, stats, logbook = setup()
    hof = tools.HallOfFame(1)

    def evaluate_fitness():
        fits = h_toolbox.map(h_toolbox.evaluate, hosts, parasites)
        for host, parasite, fit in zip(hosts, parasites, fits):
            host.fitness.values = parasite.fitness.values = fit

    def log_stats(ngen=0):
        hof.update(hosts)
        record = stats.compile(hosts)
        logbook.record(gen=ngen, evals=len(hosts), **record)
        print(logbook.stream)

    hosts = h_toolbox.population(size=300)
    parasites = p_toolbox.population(size=300)

    evaluate_fitness()
    log_stats()

    for gen in range(1, MAXGEN):
        hosts = h_toolbox.select(hosts, len(hosts))
        parasites = p_toolbox.select(parasites, len(parasites))

        hosts = tools.var_and(h_toolbox, hosts, H_CX_PROB, H_MUT_PROB)
        parasites = tools.var_and(p_toolbox, parasites, P_CX_PROB, P_MUT_PROB)

        evaluate_fitness()
        log_stats(gen)

    best_network = tools.SortingNetwork(INPUTS, hof[0])
    print_results(best_network)


if __name__ == "__main__":
    main()
