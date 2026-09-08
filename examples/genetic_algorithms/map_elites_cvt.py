import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 6
LOW, UP = -5.12, 5.12
CELLS = 16


def evaluate(individual):
    (score,) = tools.bm_rastrigin(individual)
    return (-score,)


def descriptor(individual):
    return (float(individual[0]), float(individual[1]))


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register("attr_float", tools.rng.uniform, LOW, UP)
    toolbox.register(
        "individual",
        tools.init_repeat,
        creator.Individual,
        toolbox.attr_float,
        DIM,
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_blend_bounded, alpha=0.5, low=LOW, up=UP)
    toolbox.register(
        "mutate",
        tools.mut_polynomial_bounded,
        low=LOW,
        up=UP,
        eta=20.0,
        mut_prob=1.0 / DIM,
    )
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return toolbox, stats


def print_results(archive):
    if archive.stats.num_elites == 0:
        raise RuntimeError("CVT archive stored no elites.")
    print(f"\nCVT elites: {archive.stats.num_elites}")
    print(f"QD score: {archive.stats.qd_score:.3f}")


def main():
    toolbox, stats = setup()
    samples = [[tools.rng.uniform(LOW, UP), tools.rng.uniform(LOW, UP)] for _ in range(80)]
    archive = tools.CvtArchive.from_samples(samples, k=CELLS)
    initial = toolbox.population(size=40)
    tools.ea_map_elites(
        toolbox,
        archive,
        descriptor,
        initial,
        generations=12,
        batch_size=20,
        cx_prob=0.5,
        mut_prob=0.4,
        stats=stats,
        verbose=True,
    )
    print_results(archive)


if __name__ == "__main__":
    main()
