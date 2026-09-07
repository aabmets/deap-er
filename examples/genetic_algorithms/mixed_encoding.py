import numpy
from deap_er import Fitness, Toolbox, creator, tools

tools.rng.seed(1234)  # disables randomization

# Mixed genome: on/off flag, integer count, material multiplier, two boxed reals.
MATERIALS = (0.5, 1.0, 1.5)
N_DISCRETE = 3
LOW, UP = 0.0, 1.0


def attr_flag():
    return tools.rng.randint(0, 1)


def attr_qty():
    return tools.rng.randint(1, 8)


def attr_material():
    return tools.rng.choice(MATERIALS)


def attr_real():
    return tools.rng.uniform(LOW, UP)


def flip_bit(gene):
    return 1 - int(gene)


def nudge_qty(gene):
    return int(min(8, max(1, gene + tools.rng.choice((-1, 1)))))


def redraw_material(_gene):
    return tools.rng.choice(MATERIALS)


def nudge_real(gene):
    return min(UP, max(LOW, gene + tools.rng.gauss(0.0, 0.15)))


MUTATORS = (flip_bit, nudge_qty, redraw_material, nudge_real, nudge_real)


def mate(ind1, ind2):
    for i in range(N_DISCRETE):
        if tools.rng.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    floats1 = list(ind1[N_DISCRETE:])
    floats2 = list(ind2[N_DISCRETE:])
    tools.cx_blend_bounded(floats1, floats2, alpha=0.5, low=LOW, up=UP)
    ind1[N_DISCRETE:] = floats1
    ind2[N_DISCRETE:] = floats2
    return ind1, ind2


def evaluate(individual):
    flag, qty, material, quality, waste = individual
    value = (1.0 + 0.5 * flag) * qty * material * quality - waste
    return (value,)


def setup():
    creator.create_type("FitnessMax", Fitness, weights=(1.0,))
    creator.create_type("Individual", list, fitness=creator.FitnessMax)

    toolbox = Toolbox()
    toolbox.register(
        "individual",
        tools.init_cycle,
        creator.Individual,
        (attr_flag, attr_qty, attr_material, attr_real, attr_real),
    )
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", tools.mut_heterogeneous, mutators=MUTATORS, mut_prob=0.3)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("evaluate", evaluate)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_keys = tools.Statistics(lambda ind: tuple(ind))
    mstats = tools.MultiStatistics(fitness=stats_fit, variety=stats_keys)
    mstats.register("avg", numpy.mean, chapters="fitness")
    mstats.register("max", numpy.max, chapters="fitness")
    mstats.register("dups", tools.duplicate_count, chapters="variety")

    return toolbox, mstats


def print_results(best_ind):
    print(f"\nBest widget: {best_ind}")
    print(f"Value: {best_ind.fitness.values[0]:.4f}")


def main():
    toolbox, mstats = setup()
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=20,
        cx_prob=0.5,
        mut_prob=0.4,
        hof=hof,
        stats=mstats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
