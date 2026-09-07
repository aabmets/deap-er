import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

COLUMNS = ["level", "flow"]
CASES = [(0, 256), (256, 512)]
N_ROWS = 512


def make_columns(size=N_ROWS):
    steps = numpy.linspace(0.0, 12.0, size)
    level = numpy.sin(steps) + 0.25 * steps
    flow = numpy.cos(steps * 0.7) * 2.0
    return tuple(numpy.ascontiguousarray(c, dtype=numpy.float64) for c in (level, flow))


def make_target(columns):
    level, flow = columns
    # Pair-window beta and a time-series rank — both causal.
    return gp.rolling_beta(level, flow, 4) + gp.ts_rank(level, 6)


def evaluate_batch(individuals, pset, matrix, target):
    if not individuals:
        return []
    unique = {}
    tapes = []
    index = []
    for individual in individuals:
        key = str(individual)
        slot = unique.get(key)
        if slot is None:
            unique[key] = slot = len(tapes)
            tapes.append(gp.lower_tree(individual, pset))
        index.append(slot)
    predicted = gp.interpret_tapes(tapes, matrix, backend="opcode")
    return [score_prediction(predicted[i], target) for i in index]


def score_prediction(predicted, target):
    errors = tools.case_errors(predicted, target, CASES)
    return (float(numpy.mean(errors)),)


def setup():
    columns = make_columns()
    target = make_target(columns)
    matrix = numpy.column_stack(columns)

    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_pair_window_primitives(pset)
    gp.add_ts_primitives(pset)
    gp.add_window_ephemeral(pset, "window", 2, 8)

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,))
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=3)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("clone", tools.clone_individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset, backend="opcode")
    toolbox.register(
        "evaluate_batch",
        evaluate_batch,
        pset=pset,
        matrix=matrix,
        target=target,
    )
    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.decorate("mate", gp.static_limit(lambda ind: ind.height, 8))
    toolbox.decorate("mutate", gp.static_limit(lambda ind: ind.height, 8))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    return toolbox, stats, pset, matrix, target


def print_results(best_ind, toolbox, matrix, target):
    func = toolbox.compile(expr=best_ind)
    predicted = func(*[matrix[:, i] for i in range(matrix.shape[1])])
    errors = tools.case_errors(predicted, target, CASES)
    print(f"\nBest program: {best_ind}")
    print(f"Case MSE: {tuple(round(e, 6) for e in errors)}")
    print(f"Mean case MSE: {float(numpy.mean(errors)):.6g}")


def main():
    toolbox, stats, _pset, matrix, target = setup()
    pop = toolbox.population(size=80)
    hof = tools.HallOfFame(1)
    tools.ea_simple(
        toolbox,
        pop,
        generations=15,
        cx_prob=0.5,
        mut_prob=0.2,
        hof=hof,
        stats=stats,
        verbose=True,
    )
    print_results(hof[0], toolbox, matrix, target)


if __name__ == "__main__":
    main()
