from deap_er import creator
from deap_er import tools
from deap_er import base
import random
import numpy
import array


random.seed(1234)  # disables randomization


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda x: sum(x))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not all(gene == 1 for gene in best_ind):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = tools.HallOfFame(maxsize=1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=50,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    tools.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
