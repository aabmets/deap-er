import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

COLUMNS = ["first", "second"]
ROWS = 48


def make_columns():
    steps = numpy.linspace(0.0, 4.0, ROWS)
    first = numpy.sin(steps)
    second = numpy.cos(steps * 0.7)
    return tuple(numpy.ascontiguousarray(c, dtype=numpy.float64) for c in (first, second))


def evaluate(individual, toolbox, columns):
    func = toolbox.compile(expr=individual)
    predicted = numpy.asarray(func(*columns), dtype=numpy.float64)
    target = columns[0] + columns[1]
    valid = numpy.isfinite(predicted) & numpy.isfinite(target)
    if valid.sum() < ROWS // 2:
        return (1.0e6,)
    return (float(numpy.mean((predicted[valid] - target[valid]) ** 2)),)


def setup(columns):
    pset = gp.columnar_pset(COLUMNS, window=(2, 8))
    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    gp.register_gp(
        toolbox,
        pset,
        individual=creator.Individual,
        min_depth=1,
        max_depth=3,
        height_limit=8,
        select=False,
    )
    toolbox.register("mate", gp.cx_homologous)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, columns=columns)
    toolbox.register("select", tools.sel_tournament, contestants=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    return toolbox, stats


def print_results(best_ind):
    if best_ind.fitness.values[0] >= 0.05:
        raise RuntimeError("Homologous crossover failed to reach the seed-fixed MSE floor.")
    print(f"\nBest program: {best_ind}")
    print(f"MSE: {best_ind.fitness.values[0]:.6g}")


def main():
    columns = make_columns()
    toolbox, stats = setup(columns)
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=20,
        cx_prob=0.6,
        mut_prob=0.25,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0])


if __name__ == "__main__":
    main()
