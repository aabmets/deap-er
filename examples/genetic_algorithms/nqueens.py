from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy


random.seed(1234)  # disables randomization

BOARD_SIZE = 20


def evaluate(individual):
    size = len(individual)
    left_diagonal = [0] * (2 * size - 1)
    right_diagonal = [0] * (2 * size - 1)

    for i in range(size):
        l_idx = i + individual[i]
        left_diagonal[l_idx] += 1
        r_idx = size - 1 - i + individual[i]
        right_diagonal[r_idx] += 1

    sum_ = 0
    for i in range(2*size-1):
        if left_diagonal[i] > 1:
            sum_ += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            sum_ += right_diagonal[i] - 1

    return sum_,  # The comma is essential here.


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("permutation", random.sample, range(BOARD_SIZE), BOARD_SIZE)
    toolbox.register("individual", utils.init_iterate, creator.Individual, toolbox.permutation)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("mate", ops.cx_partially_matched)
    toolbox.register("mutate", ops.mut_shuffle_indexes, mut_prob=2.0 / BOARD_SIZE)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not best_ind.fitness.values == (0.0,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nRow numbers for each queen on each column of the chessboard: \n{best_ind}')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = records.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=400,
        cx_prob=0.6,
        mut_prob=0.3,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    algos.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
