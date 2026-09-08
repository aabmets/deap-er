import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIMENSIONS = 10
BOUND_LOW, BOUND_UP = 0.0, 1.0
SURVIVORS = 40
GENERATIONS = 30


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0, -1.0))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_float,
        DIMENSIONS,
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register(
        "mate",
        tools.cx_simulated_binary_bounded,
        eta=20.0,
        low=BOUND_LOW,
        up=BOUND_UP,
    )
    toolbox.register(
        "mutate",
        tools.mut_polynomial_bounded,
        eta=20.0,
        low=BOUND_LOW,
        up=BOUND_UP,
        mut_prob=1.0 / DIMENSIONS,
    )
    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.register("select", tools.sel_spea_2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    return toolbox, stats


def print_results(population):
    hv = tools.hypervolume(population, [11.0, 11.0])
    if hv < 80.0:
        raise RuntimeError(f"Hypervolume {hv:.4f} is below the seed-fixed floor.")
    print(f"\nSPEA-II hypervolume: {hv:.4f}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=SURVIVORS)
    hof = tools.ParetoFront()
    tools.ea_mu_plus_lambda(
        toolbox,
        pop,
        generations=GENERATIONS,
        offsprings=SURVIVORS,
        survivors=SURVIVORS,
        cx_prob=0.7,
        mut_prob=0.3,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(pop)


if __name__ == "__main__":
    main()
