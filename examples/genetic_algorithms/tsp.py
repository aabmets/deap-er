import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

N_CITIES = 10
GENERATIONS = 40


def city_coords():
    return [(tools.rng.random(), tools.rng.random()) for _ in range(N_CITIES)]


CITIES = city_coords()


def tour_length(tour):
    dist = 0.0
    for i, src in enumerate(tour):
        dst = tour[(i + 1) % len(tour)]
        x0, y0 = CITIES[src]
        x1, y1 = CITIES[dst]
        dist += ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    return dist


def random_tour_baseline():
    tour = list(range(N_CITIES))
    tools.rng.shuffle(tour)
    return tour_length(tour)


BASELINE = random_tour_baseline()


def evaluate(individual):
    return (tour_length(individual),)


def setup():
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("indices", tools.rng.sample, range(N_CITIES), N_CITIES)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", tools.cx_partially_matched)
    toolbox.register("mutate", tools.mut_shuffle_indexes, mut_prob=0.2)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("clone", tools.clone_individual)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    return toolbox, stats


def print_results(best_ind):
    genes = list(best_ind)
    if sorted(genes) != list(range(N_CITIES)):
        raise RuntimeError("Best tour is not a permutation of 0..n-1.")
    length = tour_length(genes)
    if length >= BASELINE:
        raise RuntimeError(
            f"Best tour length {length:.4f} is not below the random baseline {BASELINE:.4f}."
        )
    print(f"\nBest tour length: {length:.4f} (random baseline {BASELINE:.4f})")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=GENERATIONS,
        cx_prob=0.7,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
