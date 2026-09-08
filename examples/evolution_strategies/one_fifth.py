from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 10
LAMBDA = 8
SIGMA0 = 0.5
GENERATIONS = 80


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin, sigma=SIGMA0)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.gauss, 0.0, 1.0)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_float,
        DIM,
    )
    toolbox.register("evaluate", tools.bm_sphere)
    return toolbox


def mutate(parent):
    child = creator.Individual(gene + parent.sigma * tools.rng.gauss(0.0, 1.0) for gene in parent)
    child.sigma = parent.sigma
    return child


def print_results(best_fit):
    if best_fit >= 1.0:
        raise RuntimeError("One-fifth ES failed to reach the seed-fixed floor.")
    print(f"\nBest sphere fitness: {best_fit:.6f}")


def main():
    toolbox = setup()
    parent = toolbox.individual()
    parent.sigma = SIGMA0
    parent.fitness.values = toolbox.evaluate(parent)
    best = parent.fitness.values[0]

    for _ in range(GENERATIONS):
        successes = 0
        challenger = parent
        for _ in range(LAMBDA):
            child = mutate(parent)
            child.fitness.values = toolbox.evaluate(child)
            if child.fitness.values[0] < best:
                best = child.fitness.values[0]
            if child.fitness.values[0] >= parent.fitness.values[0]:
                continue
            successes += 1
            if child.fitness.values[0] < challenger.fitness.values[0]:
                challenger = child
        if successes > 0:
            parent = challenger
        rate = successes / LAMBDA
        if rate > 0.2:
            parent.sigma *= 1.2
        elif rate < 0.2:
            parent.sigma /= 1.2

    print_results(best)


if __name__ == "__main__":
    main()
