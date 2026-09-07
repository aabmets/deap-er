import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    dim = 10
    strategy = tools.Strategy(
        centroid=[2.0] * dim,
        sigma=2.0,
        low=-5.0,
        up=5.0,
    )
    restart = tools.RestartStrategy(strategy, mode="bipop", budget=120_000)
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_rastrigin)
    toolbox.register("generate", restart.generate, creator.Individual)
    toolbox.register("update", restart.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    return toolbox, restart, stats


def main():
    toolbox, restart, stats = setup()
    hof = tools.HallOfFame(1)
    pop, logbook = tools.ea_generate_update_restarts(
        toolbox,
        restart,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print(f"\nRestarts: {restart.restart_count}, evals: {restart.evals_used}")
    print(f"Best: {hof[0].fitness.values}")
    return pop, logbook


if __name__ == "__main__":
    main()
