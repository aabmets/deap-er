import math
import operator

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools

tools.rng.seed(1234)  # disables randomization

POINTS = [x / 10.0 for x in range(-10, 10)]
TARGET = [x**2 for x in POINTS]
N_CASES = len(POINTS)
TRAIN_CASES = list(range(16))
HELD_CASES = [16, 17, 18, 19]
HELD_USED = {"n": 0}


def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return tuple((func(x) - y) ** 2 for x, y in zip(POINTS, TARGET, strict=True))


def select(individuals, sel_count, pool):
    generation = select.generation
    select.generation += 1
    matrix = tools.fitness_case_matrix(individuals)
    cases = tools.next_downsample_cases(
        individuals,
        6,
        generation,
        mode="informed",
        held_out=pool.held_out,
        matrix=matrix,
    )
    informed = tools.sample_informed_cases(individuals, 6, matrix=matrix)
    if not informed:
        raise RuntimeError("sample_informed_cases returned no cases.")
    return tools.sel_lexicase(individuals, sel_count, cases=cases, matrix=matrix)


def setup(pool):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(safe_div, 2)
    pset.add_ephemeral_constant("rand101", lambda: tools.rng.randint(-1, 1))
    pset.rename_arguments(ARG0="x")

    creator.create_type("FitnessMin", Fitness, weights=(-1.0,) * N_CASES)
    creator.create_type("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = Toolbox()
    toolbox.register("expr", gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    toolbox.register("individual", tools.init_iterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile_tree, prim_set=pset)
    toolbox.register("mate", gp.cx_one_point)
    toolbox.register("expr_mut", gp.gen_full, min_depth=0, max_depth=2)
    toolbox.register("mutate", gp.mut_uniform, expr=toolbox.expr_mut, prim_set=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    select.generation = 0
    toolbox.register("select", select, pool=pool)
    toolbox.decorate("mate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.static_limit(limiter=operator.attrgetter("height"), max_value=8))

    stats = tools.Statistics(lambda ind: math.fsum(ind.fitness.values) / N_CASES)
    stats.register("min", numpy.min)
    return toolbox, stats


def score_held_out(individual, toolbox, pool):
    HELD_USED["n"] += 1
    cases = pool.held_out.as_cases(N_CASES)
    values = toolbox.evaluate(individual)
    return math.fsum(values[i] for i in cases) / len(cases)


def print_results(best_ind, toolbox, pool):
    held = score_held_out(best_ind, toolbox, pool)
    if HELD_USED["n"] == 0:
        raise RuntimeError("Held-out exam was never scored.")
    train = [best_ind.fitness.values[i] for i in TRAIN_CASES]
    train_mse = math.fsum(train) / len(train)
    if train_mse >= 0.25:
        raise RuntimeError("Training error is above the seed-fixed floor.")
    print(f"\nTrain MSE: {train_mse:.4f}")
    print(f"Held-out MSE: {held:.4f}")


def main():
    pool = tools.CaseExamPool(
        [tools.CaseExam.from_cases(TRAIN_CASES, N_CASES)],
        held_out=tools.CaseExam.from_cases(HELD_CASES, N_CASES),
    )
    toolbox, stats = setup(pool)
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
    print_results(hof[0], toolbox, pool)


if __name__ == "__main__":
    main()
