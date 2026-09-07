import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    dim = 10
    strategy = tools.Strategy(
        centroid=[2.0] * dim,
        sigma=1.5,
        low=-5.12,
        up=5.12,
        bound_mode="resample",
    )
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_rastrigin)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    return toolbox, stats


def print_results(best_ind):
    print(f"\nBest: {best_ind.fitness.values[0]:.6g}")
    print(f"Genes stay in [-5.12, 5.12]: {all(-5.12 <= g <= 5.12 for g in best_ind)}")


def main():
    toolbox, stats = setup()
    hof = tools.HallOfFame(1)
    tools.ea_generate_update(
        toolbox,
        generations=40,
        hof=hof,
        stats=stats,
        verbose=True,
        log_time=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
