import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 6
LOW, UP = -5.12, 5.12
BINS = 8


def evaluate(individual):
    (score,) = tools.bm_rastrigin(individual)
    return (-score,)  # maximize; GridArchive qd_score sums wvalues


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

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)
    return toolbox, stats


def print_results(archive):
    summary = archive.stats
    elites = list(archive)
    best = max(elites, key=lambda ind: ind.fitness.values) if elites else None
    print(f"\nCoverage: {summary.coverage:.3f} ({summary.num_elites}/{summary.num_cells})")
    print(f"QD score: {summary.qd_score:.3f}")
    if best is not None:
        print(f"Best elite fitness: {best.fitness.values[0]:.4f}")
        print(f"Best elite behavior: {descriptor(best)}")


def main():
    toolbox, stats = setup()
    archive = tools.GridArchive(ranges=[(LOW, UP), (LOW, UP)], bins=BINS)
    initial = toolbox.population(size=40)
    tools.ea_map_elites(
        toolbox,
        archive,
        descriptor,
        initial,
        generations=15,
        batch_size=20,
        cx_prob=0.5,
        mut_prob=0.4,
        stats=stats,
        verbose=True,
        log_time=True,
    )
    print_results(archive)


if __name__ == "__main__":
    main()
