from deap_er import algorithms as algos
from deap_er import operators as ops
from deap_er import utilities as utils
from deap_er import evaluators as evals
from deap_er import records
from deap_er import creator
from deap_er import base
import random
import numpy
import array


random.seed(1234)  # disables randomization


IND_SIZE = 30
MIN_VALUE = 4
MAX_VALUE = 5
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3


def gen_evo_strat(icls, scls):
    ind = icls(random.uniform(MIN_VALUE, MAX_VALUE) for _ in range(IND_SIZE))
    ind.strategy = scls(random.uniform(MIN_STRATEGY, MAX_STRATEGY) for _ in range(IND_SIZE))
    return ind


def check_strategy(strat):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            children = func(*args, **kwargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < strat:
                        child.strategy[i] = strat
            return children
        return wrapped
    return wrapper


def setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("individual", gen_evo_strat, creator.Individual, creator.Strategy)
    toolbox.register("population", utils.init_repeat, list, toolbox.individual)
    toolbox.register("mate", ops.cx_es_blend, alpha=0.1)
    toolbox.register("mutate", ops.mut_es_log_normal, learn_rate=1.0, mut_prob=0.03)
    toolbox.register("select", ops.sel_tournament, contestants=3)
    toolbox.register("evaluate", evals.sphere)
    toolbox.decorate("mate", check_strategy(MIN_STRATEGY))
    toolbox.decorate("mutate", check_strategy(MIN_STRATEGY))

    stats = records.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, stats


def print_results(best_ind):
    if not best_ind.fitness.values < (0.5,):
        raise RuntimeError('Evolution failed to converge.')
    print(f'\nEvolution converged correctly.')


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=100)
    hof = records.HallOfFame(1)
    args = dict(
        toolbox=toolbox,
        population=pop,
        generations=500,
        offsprings=100,
        survivors=10,
        cx_prob=0.6,
        mut_prob=0.3,
        hof=hof,
        stats=stats,
        verbose=True  # prints stats
    )
    algos.ea_mu_comma_lambda(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
