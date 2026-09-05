import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

COLUMNS = ["level", "flow", "noise"]


def make_columns(size=512):
    steps = numpy.linspace(0.0, 12.0, size)
    level = numpy.sin(steps) + 0.25 * steps
    flow = numpy.cos(steps * 0.7) * 2.0
    noise = numpy.abs(numpy.sin(steps * 3.1)) + 0.5
    return tuple(numpy.ascontiguousarray(c, dtype=numpy.float64) for c in (level, flow, noise))


def make_target(columns):
    # The program the search is expected to rediscover. It is causal,
    # so its first samples are nan and the fitness must ignore them.
    level, flow, _noise = columns
    return gp.rolling_mean(level, 4) - gp.delay(flow, 2)


def evaluate(individual, toolbox, columns, target):
    func = toolbox.compile(expr=individual)
    predicted = numpy.broadcast_to(numpy.asarray(func(*columns), dtype=numpy.float64), target.shape)
    valid = numpy.isfinite(predicted) & numpy.isfinite(target)
    if valid.sum() < target.size // 2:
        return (1.0e6,)  # too much warmup or too many holes to judge
    return (float(numpy.mean((predicted[valid] - target[valid]) ** 2)),)


def setup():
    columns = make_columns()
    target = make_target(columns)

    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "window", 2, 8)

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("clone", tools.clone_individual)  # GP nodes are immutable
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, columns=columns, target=target)
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)

    toolbox.decorate("mate", gp.static_limit(lambda ind: ind.height, 8))
    toolbox.decorate("mutate", gp.static_limit(lambda ind: ind.height, 8))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    return toolbox, stats


def print_results(best_ind):
    print(f"\nBest program: {best_ind}")
    print(f"Mean squared error: {best_ind.fitness.values[0]:.6g}")


def main():
    toolbox, stats = setup()
    pop = toolbox.population(size=300)
    hof = tools.HallOfFame(1)
    args = {
        "toolbox": toolbox,
        "population": pop,
        "generations": 40,
        "cx_prob": 0.5,
        "mut_prob": 0.2,
        "hof": hof,
        "stats": stats,
        "verbose": True,  # prints stats
    }
    tools.ea_simple(**args)
    print_results(hof[0])


if __name__ == "__main__":
    main()
