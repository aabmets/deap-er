from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy


random.seed(1234)  # ensure reproducibility


def setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("mate", ops.cx_two_point_copy)
    toolbox.register("mutate", ops.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("evaluate", lambda x: sum(x))

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not all(gene == 1 for gene in best_ind):
        print('Evolution failed to converge.')
    else:
        print(f'\nThe best individual is: [1, 1, 1, ..., 1] '
              f'with a fitness score of 100.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = records.HallOfFame(maxsize=1, similar=numpy.array_equal)

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=50,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True  # print stats
    )
    algos.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
