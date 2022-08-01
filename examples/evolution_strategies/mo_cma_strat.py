from deap_er import creator
from deap_er import tools
from deap_er import base
import numpy


numpy.random.seed(1234)  # disables randomization


SIZE = 5
MU, LAMBDA = 10, 10
MIN_BOUND = numpy.zeros(SIZE)
MAX_BOUND = numpy.ones(SIZE)
EPS_BOUND = 2.e-5
NGEN = 500


def validity(individual):
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True


def feasible(individual):
    feasible_ind = numpy.array(individual)
    feasible_ind = numpy.maximum(MIN_BOUND, feasible_ind)
    feasible_ind = numpy.minimum(MAX_BOUND, feasible_ind)
    return feasible_ind


def distance(feasible_ind, original_ind):
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.decorate("evaluate", tools.ClosestValidPenalty(validity, feasible, 1.0e+6, distance))

    pop = [creator.Individual(x) for x in (numpy.random.uniform(0, 1, (MU, SIZE)))]
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    strategy = tools.StrategyMultiObjective(
        population=pop,
        sigma=1.0,
        offsprings=LAMBDA,
        survivors=MU
    )
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    return toolbox, strategy, stats, logbook


def print_results(valid, parents):
    hv = tools.hypervolume(parents, [11.0, 11.0])
    if not hv > 110 and valid != len(parents):
        raise RuntimeError('Evolution failed to converge.')
    print(f"\nNumber of valid individuals is {valid}/{len(parents)}"
          f" with a hypervolume of {hv:.2f}.")
    print(f'Evolution converged correctly.')


def main():
    toolbox, strategy, stats, logbook = setup()

    fitness_history = []
    for gen in range(NGEN):
        population = toolbox.generate()
        fitness = toolbox.map(toolbox.evaluate, population)

        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit
            fitness_history.append(fit)

        toolbox.update(population)
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        print(logbook.stream)

    num_valid = 0
    for ind in strategy.parents:
        dist = distance(feasible(ind), ind)
        if numpy.isclose(dist, 0.0, rtol=1.e-5, atol=1.e-5):
            num_valid += 1

    print_results(num_valid, strategy.parents)


if __name__ == "__main__":
    main()
