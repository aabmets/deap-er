from deap_er import creator
from deap_er import tools
from deap_er import base
import numpy


numpy.random.seed(1234)  # disables randomization


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    strategy = tools.Strategy(
        centroid=[5.0] * 30,
        sigma=5.0,
        offsprings=600,
        survivors=30
    )
    toolbox = base.Toolbox()
    toolbox.register("evaluate", tools.bm_rastrigin)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not best_ind.fitness.values < (20,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    hof = tools.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        generations=250,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    tools.ea_generate_update(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
