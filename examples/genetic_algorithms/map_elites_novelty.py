import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

DIM = 4
LOW, UP = 0.0, 1.0


def evaluate(individual):
    return (sum(individual),)


def descriptor(individual):
    return (float(individual[0]), float(individual[1]))


def setup(archive):
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
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    def mutate(individual):
        donor = archive.random_elites(1)[0] if len(archive) else individual
        return tools.mut_iso_line(individual, donor, iso=0.05, sigma=0.08, low=LOW, up=UP)

    toolbox.register("mutate", mutate)
    toolbox.register(
        "select",
        tools.sel_novelty,
        archive=archive,
        descriptor_fn=descriptor,
        k=2,
    )

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    return toolbox, stats


def print_results(archive):
    if archive.stats.num_elites <= 1:
        raise RuntimeError("Novelty MAP-Elites failed to illuminate more than one elite.")
    print(f"\nUnstructured elites: {archive.stats.num_elites}")
    print(f"QD score: {archive.stats.qd_score:.3f}")


def main():
    archive = tools.UnstructuredArchive(2, min_distance=0.12, max_elites=20)
    toolbox, stats = setup(archive)
    initial = toolbox.population(size=24)
    tools.ea_map_elites(
        toolbox,
        archive,
        descriptor,
        initial,
        generations=12,
        batch_size=16,
        cx_prob=0.3,
        mut_prob=0.7,
        stats=stats,
        verbose=True,
    )
    print_results(archive)


if __name__ == "__main__":
    main()
