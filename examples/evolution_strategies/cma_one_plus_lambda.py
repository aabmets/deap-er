import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 10
GENERATIONS = 40


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)
    parent = creator.Individual(5.0 for _ in range(DIM))
    parent.fitness.values = tools.bm_sphere(parent)
    strategy = tools.StrategyOnePlusLambda(parent, sigma=2.0, offsprings=8)
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    return toolbox, stats, strategy, parent.fitness.values[0]


def print_results(best_ind, start_fit, sigma):
    if best_ind.fitness.values[0] >= start_fit:
        raise RuntimeError("One-plus-lambda CMA did not improve the parent.")
    print(f"\nStart fitness: {start_fit:.6f}")
    print(f"Best fitness: {best_ind.fitness.values[0]:.6f}")
    print(f"Final sigma: {sigma:.6f}")


def main():
    toolbox, stats, strategy, start_fit = setup()
    hof = tools.HallOfFame(1)
    tools.ea_generate_update(
        toolbox,
        generations=GENERATIONS,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0], start_fit, strategy.sigma)


if __name__ == "__main__":
    main()
