from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

OBJECTIVES = 3
DIMENSIONS = 12
BOUND_LOW, BOUND_UP = 0.0, 1.0
REF_PPO = 6
GENERATIONS = 40


def evaluate(individual):
    return tuple(tools.bm_dtlz_2(individual, OBJECTIVES))


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * OBJECTIVES)
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    ref_points = tools.uniform_reference_points(OBJECTIVES, ref_ppo=REF_PPO)
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
        eta=30.0,
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
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.SelNSGA3WithMemory(ref_points))
    return toolbox, len(ref_points)


def print_results(population, fronts):
    if not fronts or not fronts[-1]:
        raise RuntimeError("Evolution produced an empty Pareto front.")
    hv = tools.hypervolume(population, [1.1] * OBJECTIVES)
    if hv < 0.35:
        raise RuntimeError(f"Hypervolume {hv:.4f} is below the seed-fixed floor.")
    print(f"\nNSGA-III hypervolume: {hv:.4f}")
    print(f"Last generation front size: {len(fronts[-1])}")


def main():
    toolbox, survivors = setup()
    pop = toolbox.population(size=survivors)
    fronts = []
    hof = tools.ParetoFront()
    tools.ea_mu_plus_lambda(
        toolbox,
        pop,
        generations=GENERATIONS,
        offsprings=survivors,
        survivors=survivors,
        cx_prob=0.9,
        mut_prob=0.1,
        hof=hof,
        verbose=True,
        fronts=fronts,
    )
    print_results(pop, fronts)


if __name__ == "__main__":
    main()
