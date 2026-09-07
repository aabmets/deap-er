import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    dim = 100
    strategy = tools.StrategySeparable(
        centroid=[3.0] * dim,
        sigma=3.0,
        low=-5.0,
        up=5.0,
    )
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values >= (1e-6,):
        raise RuntimeError("Evolution failed to converge.")
    print("\nEvolution converged correctly.")


def main():
    toolbox, stats = setup()
    hof = tools.HallOfFame(1)
    tools.ea_generate_update(
        toolbox,
        generations=80,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
