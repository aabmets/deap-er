import array

import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

NDIM = 10
CR = 0.25
F = 1
MU = 300
NGEN = 200


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", array.array, typecode="d", fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.uniform, -3, 3)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_float, NDIM)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("select", tools.sel_random, sel_count=3)
    toolbox.register("evaluate", tools.bm_sphere)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    return toolbox, stats, logbook


def print_results(best_ind):
    if best_ind.fitness.values >= (1e-3,):
        raise RuntimeError("Evolution failed to converge.")
    print("\nEvolution converged correctly.")


def main():
    toolbox, stats, logbook = setup()
    pop = toolbox.population(size=MU)
    hof = tools.HallOfFame(1)

    def log_stats(ngen=0):
        record = stats.compile(pop)
        logbook.record(gen=ngen, evals=len(pop), **record)
        print(logbook.stream)

    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness, strict=False):
        ind.fitness.values = fit

    log_stats()

    for gen in range(1, NGEN):
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = tools.rng.randrange(NDIM)
            for i, _value in enumerate(agent):
                if i == index or tools.rng.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)
        log_stats(gen)

    print_results(hof[0])


if __name__ == "__main__":
    main()
