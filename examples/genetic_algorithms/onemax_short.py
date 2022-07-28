from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy
import array


def eval_one_max(individual):
    return sum(individual)


def main():
    random.seed()
    toolbox = base.Toolbox()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", utils.init_repeat, creator.Individual, toolbox.attr_bool, 100)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_one_max)
    toolbox.register("mate", ops.cx_two_point)
    toolbox.register("mutate", ops.mut_flip_bit, mut_prob=0.05)
    toolbox.register("select", ops.sel_tournament, contestants=3)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop = toolbox.population(count=300)
    hof = records.HallOfFame(maxsize=1)

    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=50,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=False
    )
    algos.ea_simple(**args)

    for gene in hof[0]:
        if gene != 1:
            raise RuntimeError('Evolution failed to converge.')
    print(f'The best individual is: [1, 1, 1, ..., 1] '
          f'with a fitness score of 100.')


if __name__ == "__main__":
    main()
